import glob
import logging
import os
from typing import Any

import awkward as ak
import hydra
import numpy as np
from f9columnar.dataset_builder import RootPhysicsDataset
from f9columnar.ml.hdf5_writer import Hdf5WriterPostprocessor
from f9columnar.processors import (
    CheckpointPostprocessor,
    CheckpointProcessor,
    PostprocessorsGraph,
    Processor,
    ProcessorsGraph,
)
from f9columnar.processors_collection import ProcessorsCollection
from f9columnar.run import ColumnarEventLoop
from f9columnar.utils.helpers import dump_json, load_json, open_root_file
from omegaconf import DictConfig

from seesaw.utils.loggers import setup_logger


class BranchesProcessor(Processor):
    def __init__(self, analysis_dir: str, contents_config: DictConfig) -> None:
        super().__init__(name="branchesProcessor")
        self.analysis_dir = analysis_dir

        self.particle_type = contents_config.particle_type

        input_data_branches, input_mc_branches = self._get_hdf_inputs(contents_config)

        self.branch_name = list(set(input_data_branches + input_mc_branches))

    def _get_hdf_inputs(self, contents_config) -> tuple[list[str], list[str]]:
        input_data_branches = contents_config["data"]
        input_mc_branches = contents_config["mc"]

        self._validate_input_branches(input_data_branches, input_mc_branches)

        return input_data_branches, input_mc_branches

    def _validate_input_branches(self, input_data_branches: list[str], input_mc_branches: list[str]) -> None:
        data_file_name = f"{self.analysis_dir}/data_branches"
        mc_file_name = f"{self.analysis_dir}/mc_branches"

        data_branches = load_json(f"{data_file_name}.json")["branches"]
        mc_branches = load_json(f"{mc_file_name}.json")["branches"]

        if not set(input_data_branches).issubset(set(data_branches)):
            raise ValueError("Mismatch in data branches!")

        if not set(input_mc_branches).issubset(mc_branches):
            raise ValueError("Mismatch in mc branches!")

        if self.particle_type == "el":
            if not len([b for b in input_data_branches if "mu_" in b]) == 0:
                raise ValueError("Invalid config!")

            if not len([b for b in input_mc_branches if "mu_" in b]) == 0:
                raise ValueError("Invalid config!")

        if self.particle_type == "mu":
            if not len([b for b in input_data_branches if "el_" in b]) == 0:
                raise ValueError("Invalid config!")

            if not len([b for b in input_mc_branches if "el_" in b]) == 0:
                raise ValueError("Invalid config!")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        return {"arrays": arrays}


class WeightProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="weightProcessor")
        self.data_weights = ["correctionWeight"]
        self.mc_weights = [
            "correctionWeight",
            "generatorWeight",
            "pileupWeight",
            "luminosityWeight",
            "weight_beamspot",
        ]

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.is_data:
            weight_names = self.data_weights
        else:
            weight_names = self.mc_weights

        weight_prod = ak.ones_like(arrays[self.data_weights[0]])
        for weight_name in weight_names:
            weight_prod = weight_prod * arrays[weight_name]

        arrays["weights"] = weight_prod

        for weight_name in weight_names:
            arrays = ak.without_field(arrays, weight_name)

        return {"arrays": arrays}


class AbsEtaProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="absEtaProcessor")
        self.abs_eta_branch_name: str | None = None

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        for branch in arrays.fields:
            if "eta" in branch:
                split_branch_name = branch.split("_")
                self.abs_eta_branch_name = split_branch_name[0] + "_abs_" + "_".join(split_branch_name[1:])
                arrays[self.abs_eta_branch_name] = abs(arrays[branch])

        return {"arrays": arrays}


class CrackVetoProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="crackVetoProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        abs_eta_branch_name = self.previous_processors["absEtaProcessor"].abs_eta_branch_name

        abs_eta = arrays[abs_eta_branch_name]
        mask = (abs_eta < 1.37) | (abs_eta > 1.52)
        arrays = arrays[mask]

        return {"arrays": arrays}


class NJetsProcessor(Processor):
    def __init__(self, njets_cut: tuple[int, int | str] | None = None) -> None:
        super().__init__(name="nJetsProcessor")
        if njets_cut is not None and len(njets_cut) != 2:
            raise ValueError("njets_cut should be a tuple of two values (min, max)!")

        self.njets_cut: tuple[float, float] | None = None

        if njets_cut is not None:
            if njets_cut[1] == "inf":
                self.njets_cut = (float(njets_cut[0]), float("inf"))
            else:
                self.njets_cut = (float(njets_cut[0]), float(njets_cut[1]))

            logging.info(f"Number of jets cut set to {self.njets_cut}.")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        arrays["njets"] = ak.num(arrays["jet_selected"])

        arrays = ak.without_field(arrays, "jet_selected")

        if self.njets_cut is not None:
            mask = (arrays["njets"] >= self.njets_cut[0]) & (arrays["njets"] <= self.njets_cut[1])
            arrays = arrays[mask]

        return {"arrays": arrays}


class METProcessor(Processor):
    def __init__(self, met_cut: tuple[float, float | str]) -> None:
        super().__init__(name="metProcessor")
        if len(met_cut) != 2:
            raise ValueError("met_cut should be a tuple of two values (min, max)!")

        if met_cut[1] == "inf":
            self.met_cut = (float(met_cut[0]), float("inf"))
        else:
            self.met_cut = (float(met_cut[0]), float(met_cut[1]))

        logging.info(f"MET cut set to {self.met_cut}.")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if "met" not in arrays.fields:
            raise ValueError("MET branch not found in arrays!")

        mask = (arrays["met"] * 1e-3 >= self.met_cut[0]) & (arrays["met"] * 1e-3 <= self.met_cut[1])
        arrays = arrays[mask]

        return {"arrays": arrays}


class RavelProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="ravelProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        for branch in arrays.fields:
            arrays[branch] = ak.ravel(arrays[branch])

        return {"arrays": arrays}


class DataMCProcessor(Processor):
    def __init__(self):
        super().__init__(name="dataMCProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.is_data:
            arrays["data_type"] = ak.ones_like(arrays["weights"], dtype=np.int64)
        else:
            arrays["data_type"] = ak.zeros_like(arrays["weights"], dtype=np.int64)

        return {"arrays": arrays}


class ScaleWZjetsProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(name="scaleWZjetsProcessor")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if "Wjets" in self.reports["file_path"] or "Zjets" in self.reports["file_path"]:
            arrays["weights"] = arrays["weights"] * 0.95

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


def _build_datasets_list(files: list[str], max_files: int | None, name: str, is_data: bool) -> list[RootPhysicsDataset]:
    name = f"MC{name}" if not is_data else f"Data{name}"

    if max_files is None:
        return [RootPhysicsDataset(name, files, is_data=is_data)]

    if max_files > len(files):
        raise ValueError("max_files should be less than the number of files!")

    n = len(files) // max_files
    if len(files) % max_files != 0:
        n += 1

    datasets = []
    for i in range(n):
        datasets.append(RootPhysicsDataset(f"{name}_{i}", files[i * max_files : (i + 1) * max_files], is_data=is_data))

    return datasets


def build_datasets(
    input_dir: str, max_files: int | None, name: str
) -> tuple[list[RootPhysicsDataset], list[RootPhysicsDataset]]:
    files = glob.glob(f"{input_dir}/**/*.root", recursive=True)

    if len(files) == 0:
        raise RuntimeError(f"No files found in {input_dir}!")

    logging.info(f"Recursively found {len(files)} root files in {input_dir}.")
    logging.info(f"Found {len(files)} files in {input_dir}.")

    data_files, mc_files = [], []

    for f in files:
        if "data.Nominal" in f or "_Nominal_data_" in f:
            data_files.append(f)
        else:
            mc_files.append(f)

    logging.info(f"Found {len(data_files)} data files and {len(mc_files)} mc files.")

    data_datasets = _build_datasets_list(data_files, max_files, name, is_data=True)
    mc_datasets = _build_datasets_list(mc_files, max_files, name, is_data=False)

    return data_datasets, mc_datasets


def build_hdf_writer_analysis(
    analysis_dir: str,
    contents_config: DictConfig,
    output_file: str,
    n_piles: int,
    pile_assignment: str,
    chunk_shape: int,
    merge_piles: bool,
    enforced_types_dct: dict[str, str] | None = None,
    scale_dct: dict[str, float] | None = None,
    scale_wzjets: bool = False,
    njets_cut: tuple[int, int | str] | None = None,
    met_cut: tuple[float, float | str] | None = None,
    apply_crack_veto: bool = True,
) -> tuple[ProcessorsCollection, ProcessorsGraph, PostprocessorsGraph]:
    analysis_collection = ProcessorsCollection("analysisCollection")

    analysis_collection += BranchesProcessor(analysis_dir, contents_config)
    analysis_collection += WeightProcessor()
    analysis_collection += AbsEtaProcessor()
    analysis_collection += NJetsProcessor(njets_cut)

    if met_cut is not None:
        analysis_collection += METProcessor(met_cut)

    analysis_collection += RavelProcessor()

    if apply_crack_veto:
        logging.info("Applying crack veto.")
        analysis_collection += CrackVetoProcessor()

    analysis_collection += DataMCProcessor()

    if scale_wzjets:
        logging.info("Scaling Wjets and Zjets events by k-factor.")
        analysis_collection += ScaleWZjetsProcessor()

    if scale_dct is not None:
        analysis_collection += ScaleBranchProcessor(scale_dct)

    analysis_graph = ProcessorsGraph()
    analysis_graph.add(
        CheckpointProcessor("input"),
        *analysis_collection.as_list(),
        CheckpointProcessor("output", save_arrays=True),
    )
    analysis_graph.chain()

    analysis_graph.draw(os.path.join(os.environ["ANALYSIS_ML_LOGS_DIR"], "hdf_analysis_graph.pdf"))

    hdf5_writer = Hdf5WriterPostprocessor(
        output_file,
        flat_column_names=list(contents_config.output),
        jagged_column_names=None,
        default_chunk_shape=chunk_shape,
        n_piles=n_piles,
        pile_assignment=pile_assignment,
        merge_piles=merge_piles,
        enforce_dtypes=enforced_types_dct,
        save_node="output",
    )

    postprocessors_graph = PostprocessorsGraph()
    postprocessors_graph.add(
        CheckpointPostprocessor("input"),
        hdf5_writer,
    )
    postprocessors_graph.chain()

    postprocessors_graph.draw(os.path.join(os.environ["ANALYSIS_ML_LOGS_DIR"], "hdf_post_analysis_graph.pdf"))

    return analysis_collection, analysis_graph, postprocessors_graph


def dump_branches(root_file: str, analysis_dir: str, is_data: bool, key: str) -> list[str]:
    name = "data" if is_data else "mc"

    f_data = open_root_file(root_file, to_arrays=False, key=key)
    branches = f_data.keys()

    file_name = os.path.join(analysis_dir, f"{name}_branches")

    dump_json({"branches": branches}, f"{file_name}.json")

    return branches


def convert_to_hdf5(
    *,
    analysis_dir: str,
    ntuples_dir: str,
    contents_config: DictConfig,
    dataloader_config: dict[str, Any],
    max_files: int | None,
    n_piles: int,
    pile_assignment: str,
    chunk_shape: int,
    merge_piles: bool,
    enforced_types_dct: dict[str, str] | None,
    scale_dct: dict[str, float] | None,
    scale_wzjets: bool,
    njets_cut: tuple[int, int | str] | None,
    met_cut: tuple[float, float | str] | None,
    apply_crack_veto: bool,
    output_file: str,
) -> None:
    particle_type = contents_config.particle_type

    if particle_type not in ["el", "mu"]:
        raise ValueError("Invalid particle type!")

    data_datasets, mc_datasets = build_datasets(ntuples_dir, max_files, "ROOTDataset")

    dump_branches(
        data_datasets[0].root_files[0],
        analysis_dir,
        is_data=True,
        key=dataloader_config["key"],
    )
    dump_branches(
        mc_datasets[0].root_files[0],
        analysis_dir,
        is_data=False,
        key=dataloader_config["key"],
    )

    analysis_collection, analysis_graph, postprocessors_graph = build_hdf_writer_analysis(
        analysis_dir,
        contents_config,
        output_file,
        n_piles,
        pile_assignment,
        chunk_shape,
        merge_piles,
        enforced_types_dct=enforced_types_dct,
        scale_dct=scale_dct,
        scale_wzjets=scale_wzjets,
        njets_cut=njets_cut,
        met_cut=met_cut,
        apply_crack_veto=apply_crack_veto,
    )

    branch_filter = analysis_collection.branch_name_filter
    dataloader_config["filter_name"] = branch_filter

    logging.info("[green]Setting up data dataloaders...[/green]")
    for data_dataset in data_datasets:
        data_dataset.setup_dataloader(**dataloader_config)
        data_dataset.init_dataloader(processors=analysis_graph)

    logging.info("[green]Setting up mc dataloaders...[/green]")
    for mc_dataset in mc_datasets:
        mc_dataset.setup_dataloader(**dataloader_config)
        mc_dataset.init_dataloader(processors=analysis_graph)

    event_loop = ColumnarEventLoop(
        mc_datasets=mc_datasets,
        data_datasets=data_datasets,
        postprocessors_graph=postprocessors_graph,
        fit_postprocessors=True,
        cut_flow=False,
    )
    event_loop.run()

    postprocessors_graph["hdf5WriterPostprocessor"].close()


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
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
        "dataloader_kwargs": {"multiprocessing_context": convert_config.multiprocessing_context},
    }

    ntuples_dir = convert_config.ntuples_dir
    if ntuples_dir is None:
        ntuples_dir = os.environ["ANALYSIS_ML_NTUPLES_DIR"]
        logging.info(f"Using default ntuples dir: {ntuples_dir}.")

    output_dir = convert_config.output_dir
    if output_dir is None:
        output_dir = ml_data_dir
        logging.info(f"Using default ntuples dir: {ntuples_dir}.")

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist!")

    if "/eos/" in ntuples_dir:
        ntuples_dir = ml_data_dir
        logging.info(f"Detected eos, changed ntuples dir: {ntuples_dir}.")

    output_file = os.path.join(output_dir, convert_config.output_file)
    merge_piles = convert_config.get("merge_piles", False)
    enforced_types_dct = config.hdf5_config.get("enforce_types", None)
    scale_dct = config.hdf5_config.get("scale", None)

    convert_to_hdf5(
        analysis_dir=ml_data_dir,
        ntuples_dir=ntuples_dir,
        contents_config=config.hdf5_config,
        dataloader_config=dataloader_config,
        max_files=convert_config.max_files,
        n_piles=convert_config.n_piles,
        pile_assignment=convert_config.get("pile_assignment", "random"),
        chunk_shape=convert_config.chunk_shape,
        merge_piles=merge_piles,
        enforced_types_dct=enforced_types_dct,
        scale_dct=scale_dct,
        scale_wzjets=convert_config.get("scale_wzjets", False),
        njets_cut=convert_config.get("njets_cut", None),
        met_cut=convert_config.get("met_cut", None),
        apply_crack_veto=convert_config.get("crack_veto", True),
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
