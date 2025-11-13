import copy
import glob
import hashlib
import json
import logging
import os
from typing import Any

import awkward as ak
import h5py
import hydra
import numpy as np
from f9columnar.dataset_builder import RootPhysicsDataset
from f9columnar.ml.dataloader_helpers import get_hdf5_writer_branches
from f9columnar.ml.hdf5_parallel_writer import Hdf5WriterProcessor
from f9columnar.processors import (
    CheckpointProcessor,
    Processor,
    ProcessorsGraph,
)
from f9columnar.processors_collection import ProcessorsCollection
from f9columnar.run import ColumnarEventLoop
from f9columnar.utils.helpers import dump_json, load_json, open_root_file
from omegaconf import DictConfig

from seesaw.utils.loggers import setup_logger


class BranchesProcessor(Processor):
    def __init__(self, analysis_dir: str, config_branches: list[str], labels: list[str]) -> None:
        super().__init__(name="branchesProcessor")
        self.analysis_dir = analysis_dir
        self.labels = labels

        self._validate_input_branches(config_branches)

        self.branch_name = config_branches

    def _validate_input_branches(self, branches: list[str]) -> None:
        label_id = hashlib.md5("".join(sorted(self.labels)).encode()).hexdigest()
        branches_file_name = f"{self.analysis_dir}/{label_id}_branches.json"

        test_branches = load_json(branches_file_name)["branches"]

        for branch in branches:
            if branch not in test_branches:
                raise ValueError(f"Branch {branch} not found in {branches_file_name}!")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        return {"arrays": arrays}


class WeightsProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="weightsProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        arrays["weights"] = arrays["eventWeight"]

        arrays = ak.without_field(arrays, "eventWeight")

        return {"arrays": arrays}


class SignalMassProcessor(Processor):
    def __init__(self, signal_label: str | None, ml_mass_branches: list[str] | None) -> None:
        super().__init__(name="signalMassProcessor")
        if signal_label is None:
            logging.warning("No signal name provided, sig_type will be 0.0!")
        else:
            logging.info(f"Signal name set to {signal_label}.")

        if ml_mass_branches is None:
            logging.warning("No MLMass branches provided, sig_type will be 0.0!")
        else:
            logging.info(f"MLMass branches set to {ml_mass_branches}.")

        self.signal_label = signal_label
        self.ml_mass_branches = ml_mass_branches

    def _get_ml_mass(self, arrays: ak.Array) -> float:
        if self.ml_mass_branches is not None:
            for i, ml_mass_branch in enumerate(self.ml_mass_branches):
                if i == 0:
                    ml_mass = arrays[ml_mass_branch][0]
                else:
                    ml_mass += arrays[ml_mass_branch][0]

            return ml_mass

        return 0.0

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.signal_label is None:
            arrays["sig_type"] = 0.0
        elif self.signal_label in self.reports["file_path"]:
            arrays["sig_type"] = self._get_ml_mass(arrays)
        else:
            arrays["sig_type"] = 0.0

        return {"arrays": arrays}


class LabelsProcessor(Processor):
    def __init__(self, labels: list[str], signal_label: str | None = None, other_label: bool = False) -> None:
        super().__init__(name="labelsProcessor")
        self.labels = copy.deepcopy(labels)
        self.signal_label = signal_label
        self.other_label = other_label

        self.is_binary = len(labels) == 0

        if self.is_binary:
            logging.info("Binary classification mode enabled. Background will be labeled as 0.")

        if self.is_binary and self.signal_label is None:
            raise ValueError("If labels are empty, signal_label must be provided!")

        if self.signal_label is not None:
            self.offset = 2
        else:
            self.offset = 1

        if not self.other_label:
            self.offset = self.offset - 1

        if self.signal_label is not None:
            self.signal_int_label = 1 if self.other_label or self.is_binary else 0
            logging.info(f"Using signal {self.signal_label} with label {self.signal_int_label}.")

        if "data" in self.labels:
            self.write_data = True
            self.data_int_label = len(self.labels) + self.offset - 1
            self.labels.remove("data")
            logging.info(f"[yellow]Data mode enabled. Data will be labeled as {self.data_int_label}.")
        else:
            self.write_data = False

        if not self.is_binary:
            _label_dct = {label: i + self.offset for i, label in enumerate(self.labels)}
            logging.info(f"Multi-class classification mode enabled with labels: {_label_dct}.")

            if self.other_label:
                logging.info("Label 0 will be assigned to events that do not match any label.")

    def _get_label(self) -> int:
        current_file = self.reports["file_path"]

        if self.write_data and "data.Nominal" in current_file or "_Nominal_data_" in current_file:
            return self.data_int_label

        if self.signal_label is not None and self.signal_label in current_file:
            return self.signal_int_label

        if self.is_binary:
            return 0

        for i, label in enumerate(self.labels):
            if label in current_file:
                return i + self.offset

        if not self.other_label:
            raise RuntimeError(f"Label not found in file {current_file}!")

        return 0

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        arrays["label_type"] = self._get_label()
        return {"arrays": arrays}


class UnknownLeptonProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="unknownLeptonProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if "el_type" in arrays.fields:
            unknown_mask = arrays["el_type"] != 3

            for field in arrays.fields:
                if field.startswith("el_"):
                    arrays[field] = arrays[field][unknown_mask]

            arrays = ak.without_field(arrays, "el_type")

        if "mu_type" in arrays.fields:
            unknown_mask = arrays["mu_type"] != 3

            for field in arrays.fields:
                if field.startswith("mu_"):
                    arrays[field] = arrays[field][unknown_mask]

            arrays = ak.without_field(arrays, "mu_type")

        return {"arrays": arrays}


class ScaleBranchProcessor(Processor):
    def __init__(self, scale_dct: dict[str, float]) -> None:
        super().__init__(name="scaleBranchProcessor")
        self.scale_dct = scale_dct

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        for branch, scale in self.scale_dct.items():
            if branch in arrays.fields:
                elements_type = str(ak.type(arrays[branch])).split("*")[-1].strip()
                arrays[branch] = arrays[branch] * getattr(np, elements_type)(scale)
            else:
                raise RuntimeError(f"Branch {branch} not found in arrays, cannot scale!")

        return {"arrays": arrays}


class ScaleWZjetsProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="scaleWjetsProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if "Wjets" in self.reports["file_path"] or "Zjets" in self.reports["file_path"]:
            arrays["weights"] = arrays["weights"] * 0.95

        return {"arrays": arrays}


class SameSignProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="sameSignProcessor")

    @staticmethod
    def same_sign_pairs_mask(lep1: ak.Array, lep2: ak.Array) -> ak.Array:
        charges = ak.concatenate((lep1, lep2), axis=1)

        # count number of +1 and -1
        count_pos = ak.sum(charges == 1, axis=1)
        count_neg = ak.sum(charges == -1, axis=1)

        # count how many full same-sign pairs we can form
        same_sign_pairs = (count_pos // 2) + (count_neg // 2)

        # total number of full pairs we can possibly form from all charges
        total_pairs = ak.num(charges) // 2

        # it's valid only if all pairs we can form are same-sign pairs
        return same_sign_pairs == total_pairs

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if "el_charge" in arrays.fields and "mu_charge" in arrays.fields:
            ss_pair_mask = self.same_sign_pairs_mask(arrays["el_charge"], arrays["mu_charge"])
            arrays = arrays[ss_pair_mask]

        return {"arrays": arrays}


def build_datasets(
    input_dir: str,
    max_files: int | None,
    dataset_name: str,
    filter_labels: list[str] | None = None,
    disable_other_label: bool = False,
) -> tuple[list[RootPhysicsDataset], bool]:
    files = glob.glob(f"{input_dir}/**/*.root", recursive=True)

    if len(files) == 0:
        raise RuntimeError(f"No files found in {input_dir}!")

    logging.info(f"Recursively found {len(files)} root files in {input_dir}.")

    other_label = False

    if filter_labels is not None:
        filter_files = []
        for file in files:
            base_name = os.path.basename(file).split(".")[0]
            for label in filter_labels:
                if label == base_name:
                    filter_files.append(file)
                    break

        other_label_files = set(files) - set(filter_files)

        if len(other_label_files) != 0:
            if disable_other_label:
                logging.info("Disabling other label, not using files that do not match any labels.")
                files = filter_files
            else:
                logging.info(f"Using label 0 for files: {list(other_label_files)}.")
                other_label = True

    if max_files is None:
        return [RootPhysicsDataset(f"{dataset_name}", files, is_data=False)], other_label

    if max_files > len(files):
        raise ValueError("max_files should be less than the number of files!")

    n = len(files) // max_files
    if len(files) % max_files != 0:
        n += 1

    datasets = []
    for i in range(n):
        name = f"{dataset_name}_{i}"
        datasets.append(RootPhysicsDataset(name, files[i * max_files : (i + 1) * max_files], is_data=False))

    return datasets, other_label


def build_hdf_writer_analysis(
    analysis_dir: str,
    contents_config: DictConfig,
    output_file: str,
    n_piles: int,
    pile_assignment: str,
    chunk_shape: int,
    labels: list[str],
    custom_chunk_shapes_dct: dict[str, int] | None = None,
    signal_label: str | None = None,
    other_label: bool = False,
    ml_mass_branches: list[str] | None = None,
    enforced_types_dct: dict[str, str] | None = None,
    scale_dct: dict[str, float] | None = None,
    same_sign_leptons: bool = False,
    scale_wzjets: bool = False,
    num_workers: int = 1,
) -> tuple[ProcessorsCollection, ProcessorsGraph]:
    analysis_collection = ProcessorsCollection("analysisCollection")

    flat_branches, jagged_branches, max_lengths, pad_values = get_hdf5_writer_branches(dict(contents_config))

    input_branches: list[str] = []

    if len(flat_branches) != 0:
        input_branches += flat_branches["output"] + flat_branches["extra_input"]

    for object_dct in jagged_branches.values():
        input_branches += object_dct["output"] + object_dct["extra_input"]

    analysis_collection += BranchesProcessor(analysis_dir, input_branches, labels)

    if len(flat_branches) != 0:
        analysis_collection += WeightsProcessor()
        analysis_collection += SignalMassProcessor(signal_label, ml_mass_branches)
        analysis_collection += LabelsProcessor(labels, signal_label=signal_label, other_label=other_label)

    if len(jagged_branches) != 0:
        analysis_collection += UnknownLeptonProcessor()
        if same_sign_leptons and "el_charge" in input_branches and "mu_charge" in input_branches:
            logging.info("Same sign leptons enabled.")
            analysis_collection += SameSignProcessor()
        elif same_sign_leptons and ("el_charge" not in input_branches or "mu_charge" not in input_branches):
            logging.warning("Same sign leptons enabled, but charge branches are not present in the input data.")
        else:
            logging.info("No restriction on lepton charges.")

    if scale_dct is not None:
        analysis_collection += ScaleBranchProcessor(scale_dct)

    if scale_wzjets:
        logging.info("Scaling Wjets and Zjets events by k-factor.")
        analysis_collection += ScaleWZjetsProcessor()

    if len(flat_branches) != 0:
        flat_output_columns = flat_branches["output"] + flat_branches["extra_output"]
    else:
        flat_output_columns = []

    jagged_object_output_columns = {}
    for key, object_dct in jagged_branches.items():
        jagged_object_output_columns[key] = object_dct["output"] + object_dct["extra_output"]

    hdf5_writer = Hdf5WriterProcessor(
        file_path=output_file,
        flat_column_names=flat_output_columns,
        jagged_column_names=jagged_object_output_columns,
        default_chunk_shape=chunk_shape,
        custom_chunk_shape_dct=custom_chunk_shapes_dct,
        max_lengths=max_lengths,
        pad_values=pad_values,
        n_piles=n_piles,
        pile_assignment=pile_assignment,
        enforce_dtypes=enforced_types_dct,
        max_workers=num_workers,
    )
    analysis_collection += hdf5_writer

    analysis_graph = ProcessorsGraph()
    analysis_graph.add(
        CheckpointProcessor("input"),
        *analysis_collection.as_list(),
        CheckpointProcessor("output", save_arrays=True),
    )
    analysis_graph.chain()

    return analysis_collection, analysis_graph


def dump_branches(root_file: str, analysis_dir: str, key: str, labels: list[str]) -> list[str]:
    f_data = open_root_file(root_file, to_arrays=False, key=key)
    branches = f_data.keys()

    label_id = hashlib.md5("".join(sorted(labels)).encode()).hexdigest()
    file_name = os.path.join(analysis_dir, f"{label_id}_branches")

    dump_json({"branches": branches}, f"{file_name}.json")

    return branches


def convert_to_hdf5(
    *,
    analysis_dir: str,
    ntuples_dir: str,
    signal_label: str | None,
    labels: list[str],
    contents_config: DictConfig,
    dataloader_config: dict[str, Any],
    max_files: int | None,
    n_piles: int,
    pile_assignment: str,
    chunk_shape: int,
    custom_chunk_shapes_dct: dict[str, int] | None,
    enforced_types_dct: dict[str, str] | None,
    scale_dct: dict[str, float] | None,
    same_sign_leptons: bool = False,
    scale_wzjets: bool = False,
    disable_other_label: bool = False,
    output_file: str,
) -> bool:
    if signal_label and any(signal_label in label for label in labels):
        raise ValueError(f"Signal label '{signal_label}' cannot be part of labels: {labels}.")

    if len(labels) != 0 and signal_label is not None:
        filter_labels = [signal_label] + labels
    elif len(labels) != 0 and signal_label is None:
        filter_labels = labels
    else:
        filter_labels = None

    datasets, other_label = build_datasets(
        ntuples_dir,
        max_files,
        "ROOTDataset",
        filter_labels=filter_labels,
        disable_other_label=disable_other_label,
    )

    branches = dump_branches(
        datasets[0].root_files[0],
        analysis_dir,
        key=dataloader_config["key"],
        labels=labels,
    )

    ml_mass_branches = []
    for branch in branches:
        if "mlmass" in branch.lower():
            ml_mass_branches.append(branch)

    analysis_collection, analysis_graph = build_hdf_writer_analysis(
        analysis_dir,
        contents_config,
        output_file,
        n_piles,
        pile_assignment,
        chunk_shape,
        labels,
        custom_chunk_shapes_dct=custom_chunk_shapes_dct,
        signal_label=signal_label,
        other_label=other_label,
        ml_mass_branches=ml_mass_branches if len(ml_mass_branches) != 0 else None,
        enforced_types_dct=enforced_types_dct,
        scale_dct=scale_dct,
        same_sign_leptons=same_sign_leptons,
        scale_wzjets=scale_wzjets,
        num_workers=dataloader_config["num_workers"],
    )

    branch_filter = analysis_collection.branch_name_filter
    dataloader_config["filter_name"] = branch_filter

    num_entries = 0
    logging.info("[green]Initializing dataloaders...[/green]")
    for ds in datasets:
        ds.setup_dataloader(**dataloader_config)
        ds.init_dataloader(processors=analysis_graph)
        num_entries += ds.num_entries

    if n_piles is not None:
        exepected_events = chunk_shape * n_piles

        if exepected_events > num_entries:
            logging.critical("Decrease chunk shape or number of piles to avoid empty piles.")
            raise ValueError(f"HDF5 expected events ({exepected_events}) > events ({num_entries}).")

    event_loop = ColumnarEventLoop(
        mc_datasets=None,
        data_datasets=datasets,
        postprocessors_graph=None,
        fit_postprocessors=False,
        cut_flow=False,
    )
    event_loop.run(data_only=True)

    return other_label


def add_labels_to_metadata(
    file_path: str,
    labels: list[str],
    merge_piles: bool,
    signal_label: str | None,
    other_label: bool = False,
) -> None:
    logging.info("Adding labels to metadata.")

    if signal_label is None:
        offset = 1
    else:
        offset = 2

    labels_dct: dict[str, int] = {}

    if other_label:
        labels_dct["other"] = 0
    else:
        offset = offset - 1

    if signal_label is None:
        labels_dct = labels_dct | {label: i + offset for i, label in enumerate(labels)}
    elif len(labels) == 0:
        labels_dct = {"background": 0, signal_label: 1}
    else:
        labels_dct = labels_dct | {signal_label: 1 if other_label else 0}
        labels_dct = labels_dct | {label: i + offset for i, label in enumerate(labels)}

    if "data" in labels_dct:
        _labels_dct: dict[str, int] = {}
        i = 0
        for k in labels_dct.keys():
            if k != "data":
                _labels_dct[k] = i
                i += 1

        _labels_dct["data"] = len(labels_dct) + offset - 1
        labels_dct = _labels_dct

    logging.info(f"Labels dictionary: {labels_dct}")

    if merge_piles:
        with h5py.File(file_path, "r+") as f:
            metadata = json.loads(f["metadata"][()])
            metadata["labels"] = labels_dct

            del f["metadata"]
            f.create_dataset("metadata", data=json.dumps(metadata))

        return None

    base_path = os.path.dirname(file_path)
    pile_files = glob.glob(os.path.join(base_path, "p*.hdf5"))

    if len(pile_files) == 0:
        raise RuntimeError("No piles found in the output directory!")

    for pile_file in pile_files:
        with h5py.File(pile_file, "r+") as f:
            metadata = json.loads(f["metadata"][()])
            metadata["labels"] = labels_dct

            del f["metadata"]
            f.create_dataset("metadata", data=json.dumps(metadata))


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="convert_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(min_level=config.min_logging_level)

    convert_config = config.convert_config
    ml_data_dir = os.environ["ANALYSIS_ML_DATA_DIR"]

    dataloader_config = {
        "step_size": convert_config.step_size,
        "key": convert_config.key,
        "num_workers": convert_config.num_workers,
        "partition_size": config.convert_config.n_piles,
        "dataloader_kwargs": {"multiprocessing_context": convert_config.multiprocessing_context},
    }

    signal_label = config.hdf5_config.get("signal_label", None)
    labels = list(config.hdf5_config.get("labels", []))

    ntuples_dir = convert_config.ntuples_dir
    if ntuples_dir is None:
        ntuples_dir = os.environ["ANALYSIS_ML_NTUPLES_DIR"]
        logging.info(f"Using default ntuples dir: {ntuples_dir}.")

    output_dir = convert_config.output_dir
    if output_dir is None:
        output_dir = ml_data_dir
        logging.info(f"Using default output dir: {output_dir}. Make sure to have write access.")

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist!")

    if "/eos/" in ntuples_dir:
        ntuples_dir = ml_data_dir
        logging.info(f"Detected eos, changed ntuples dir: {ntuples_dir}.")

    output_file = os.path.join(output_dir, convert_config.output_file)
    merge_piles = convert_config.get("merge_piles", False)
    custom_chunk_shapes_dct = config.hdf5_config.get("custom_chunk_shapes", None)
    enforced_types_dct = config.hdf5_config.get("enforce_types", None)
    scale_dct = config.hdf5_config.get("scale", None)

    other_label = convert_to_hdf5(
        analysis_dir=ml_data_dir,
        ntuples_dir=ntuples_dir,
        signal_label=signal_label,
        labels=labels,
        contents_config=config.hdf5_config,
        dataloader_config=dataloader_config,
        max_files=convert_config.max_files,
        n_piles=convert_config.get("n_piles", None),
        pile_assignment=convert_config.get("pile_assignment", "random"),
        chunk_shape=convert_config.chunk_shape,
        custom_chunk_shapes_dct=custom_chunk_shapes_dct,
        enforced_types_dct=enforced_types_dct,
        scale_dct=scale_dct,
        same_sign_leptons=convert_config.get("same_sign_leptons", False),
        scale_wzjets=convert_config.get("scale_wzjets", False),
        disable_other_label=convert_config.get("disable_other_label", False),
        output_file=os.path.join(output_dir, convert_config.output_file),
    )

    add_labels_to_metadata(output_file, labels, merge_piles, signal_label, other_label)


if __name__ == "__main__":
    main()
