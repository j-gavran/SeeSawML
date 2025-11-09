import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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
from f9columnar.utils.helpers import load_json
from omegaconf import DictConfig

from seesaw.fakes.dataset.hdf5_converter import AbsEtaProcessor, BranchesProcessor, CrackVetoProcessor, dump_branches
from seesaw.utils.dsid_store import DSIDStore
from seesaw.utils.loggers import setup_logger


@dataclass
class OpendataMetadata:
    path: Path
    metadata_dct: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata_files = list(self.path.glob("*.metadata.json"))
        for meta_file in metadata_files:
            meta_id = meta_file.stem.split(".")[0]
            meta_dct = load_json(meta_file)
            self.metadata_dct[meta_id] = meta_dct

        logging.info(f"Loaded metadata for {len(self.metadata_dct)} files from {self.path}.")

    def get(self, meta_id: str) -> dict[str, Any]:
        if meta_id not in self.metadata_dct:
            raise KeyError(f"Metadata for id {meta_id} not found!")

        return self.metadata_dct[meta_id]

    def __getitem__(self, meta_id: str) -> dict[str, Any]:
        return self.get(meta_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path}, num_metadata={len(self.metadata_dct)})"

    def __str__(self) -> str:
        ids_preview = ", ".join(list(self.metadata_dct.keys())[:3])
        if len(self.metadata_dct) > 3:
            ids_preview += ", ..."
        return f"OpendataMetadata({len(self.metadata_dct)} files, ids=[{ids_preview}])"

    def save_as_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json_dct = {k: v for k, v in self.metadata_dct.items()}
            json.dump(json_dct, f, indent=4)

        logging.info(f"Saved metadata to {output_path}.")


class WeightProcessor(Processor):
    def __init__(self, metadata: OpendataMetadata) -> None:
        super().__init__(name="weightProcessor")
        self.metadata = metadata

        self.data_weights: list[str] = []
        self.mc_weights = [
            "mcWeight",
            "ScaleFactor_PILEUP",
        ]

        self.luminosity = 36.0e3  # pb-1

    def _luminosity_weight(self) -> float:
        dsid = self.reports["file"].split(".")[0].split("_")[-1]
        m = self.metadata[dsid]
        return self.luminosity * m["cross_section_pb"] * m["genFiltEff"] * m["kFactor"] / m["sumOfWeights"]

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.is_data:
            weight_names = self.data_weights
        else:
            weight_names = self.mc_weights

        seed = "lep_pt" if "lep_pt" in arrays.fields else arrays.fields[0]
        weight_prod = ak.ones_like(arrays[seed])

        for weight_name in weight_names:
            weight_prod = weight_prod * arrays[weight_name]

        if not self.is_data:
            weight_prod = weight_prod * self._luminosity_weight()

        arrays["weights"] = weight_prod

        for weight_name in weight_names:
            arrays = ak.without_field(arrays, weight_name)

        return {"arrays": arrays}


class LeptonProcessor(Processor):
    def __init__(self, particle_type: str) -> None:
        super().__init__(name="leptonProcessor")

        self.particle_type = particle_type

        if self.particle_type == "el":
            self.particle_pdgid = 11
            self.lep_sf_branch = "ScaleFactor_ELE"
        elif self.particle_type == "mu":
            self.particle_pdgid = 13
            self.lep_sf_branch = "ScaleFactor_MUON"

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        # Keep single-lepton events
        arrays = arrays[arrays["lep_n"] == 1]
        arrays = ak.without_field(arrays, "lep_n")

        # Only keep events with the correct lepton
        arrays = arrays[ak.flatten(arrays["lep_type"] == self.particle_pdgid)]
        arrays = ak.without_field(arrays, "lep_type")

        # Apply lepton scale factors
        if not self.is_data:
            arrays["weights"] = arrays["weights"] * arrays[self.lep_sf_branch]
            arrays = ak.without_field(arrays, self.lep_sf_branch)

        # Rename lepton branches
        for branch in ["pt", "eta", "phi", "charge"]:
            arrays[f"{self.particle_type}_{branch}"] = arrays[f"lep_{branch}"]
            arrays = ak.without_field(arrays, f"lep_{branch}")

        return {"arrays": arrays}


class LeptonTypeProcessor(Processor):
    def __init__(self, particle_type: str, id_wp: str = "tight", iso_wp: str = "loose") -> None:
        super().__init__(name="leptonTypeProcessor")
        if id_wp == "tight":
            self.id_wp = "lep_isTightID"
        elif id_wp == "medium":
            self.id_wp = "lep_isMediumID"
        elif id_wp == "loose":
            self.id_wp = "lep_isLooseID"
        else:
            raise ValueError(f"Invalid id_wp: {id_wp}")

        if iso_wp == "tight":
            self.iso_wp = "lep_isTightIso"
        elif iso_wp == "loose":
            self.iso_wp = "lep_isLooseIso"
        else:
            raise ValueError(f"Invalid iso_wp: {iso_wp}")

        self.name_type = f"{particle_type}_type"

        logging.info(f"Using {particle_type} ID: {self.id_wp} and isolation: {self.iso_wp}.")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        id_mask = ak.values_astype(arrays[self.id_wp], np.bool_)
        iso_mask = ak.values_astype(arrays[self.iso_wp], np.bool_)

        type_mask = id_mask & iso_mask

        particle_type = ak.zeros_like(type_mask, dtype=np.int64)  # 0 for tight
        particle_type = ak.where(~type_mask, 2, particle_type)  # 2 for loose not tight

        arrays[self.name_type] = particle_type

        ak.without_field(arrays, self.id_wp)
        ak.without_field(arrays, self.iso_wp)

        return {"arrays": arrays}


class PreselectionProcessor(Processor):
    def __init__(
        self,
        particle_type: str,
        pt_min: float = 10.0,
        bjet_veto: bool = False,
        btag_quantile_working_point: int = 3,
    ) -> None:
        super().__init__(name="preselectionProcessor")
        self.pt_branch = f"{particle_type}_pt"
        self.pt_min = pt_min
        self.bjet_veto = bjet_veto

        self.btag_quantile_working_point = int(btag_quantile_working_point)

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.pt_branch not in arrays.fields:
            raise RuntimeError(f"Preselection: pt_branch: '{self.pt_branch}' not found in arrays")

        pt_scalar = ak.values_astype(ak.fill_none(ak.firsts(arrays[self.pt_branch]), 0.0), np.float64)
        mask = pt_scalar > self.pt_min

        if self.bjet_veto:
            bjet_veto_field = ak.ones_like(mask, dtype=np.bool_)

            jet_btag_quantile = ak.fill_none(arrays["jet_btag_quantile"], -99)
            bjet_veto_field = ak.all(jet_btag_quantile >= self.btag_quantile_working_point, axis=-1)

            mask = mask & bjet_veto_field
            arrays = arrays[mask]

            arrays = ak.without_field(arrays, "jet_btag_quantile")

            if self.btag_quantile_working_point == 3:
                btag_sf = arrays["ScaleFactor_BTAG"]
                arrays["weights"] = arrays["weights"] * btag_sf
                arrays = ak.without_field(arrays, "ScaleFactor_BTAG")
        else:
            arrays = arrays[mask]

        return {"arrays": arrays}


class TriggerProcessor(Processor):
    def __init__(self, particle_type: str) -> None:
        super().__init__(name="triggerProcessor")

        if particle_type == "el":
            self.trig_passed_branch = "trigE"  # Single-electron trigger
            self.trig_sf_branch = "ScaleFactor_ElTRIGGER"
        elif particle_type == "mu":
            self.trig_passed_branch = "trigM"  # Single-muon trigger
            self.trig_sf_branch = "ScaleFactor_MuTRIGGER"

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        # Keep events for which the trigger fired
        arrays = arrays[arrays[self.trig_passed_branch]]
        arrays = ak.without_field(arrays, self.trig_passed_branch)

        # Apply trigger matching
        arrays = arrays[ak.flatten(arrays["lep_isTrigMatched"])]
        arrays = ak.without_field(arrays, "lep_isTrigMatched")

        # Apply trigger scale factors
        if not self.is_data:
            arrays["weights"] = arrays["weights"] * arrays[self.trig_sf_branch]
            arrays = ak.without_field(arrays, self.trig_sf_branch)

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
        arrays["njets"] = arrays["jet_n"]

        arrays = ak.without_field(arrays, "jet_n")

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

        mask = (arrays["met"] >= self.met_cut[0]) & (arrays["met"] <= self.met_cut[1])
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
    def __init__(self, dsid_store: DSIDStore) -> None:
        super().__init__(name="scaleWZjetsProcessor")
        self.dsid_store = dsid_store

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.is_data:
            return {"arrays": arrays}

        dsid = int(self.reports["file"].split(".")[0].split("_")[-1])

        sample_name = self.dsid_store.get_sample(dsid)

        if sample_name == "W+jets" or sample_name == "Z+jets":
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


class TransverseMassProcessor(Processor):
    def __init__(self, mt_cut: tuple[float, float] | None = None) -> None:
        super().__init__(name="transverseMassProcessor")

        self.mt_cut = mt_cut

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        arrays["m_t"] = np.sqrt(
            2.0 * arrays["el_pt"] * arrays["met"] * (1.0 - np.cos(arrays["el_phi"] - arrays["met_phi"]))
        )

        if self.mt_cut is not None:
            mask = (arrays["m_t"] >= self.mt_cut[0]) & (arrays["m_t"] <= self.mt_cut[1])
            arrays = arrays[mask]

        return {"arrays": arrays}


def build_datasets(
    input_dir: Path, name: str, samples: list[str] | None = None
) -> tuple[RootPhysicsDataset, RootPhysicsDataset, DSIDStore | None]:
    if samples is not None:
        dsid_store = DSIDStore(samples)
    else:
        dsid_store = None

    data_files, mc_files = [], []
    accepted_mc_dct: dict[str, list[int]] = {}

    accept, total = 0, 0
    for file in input_dir.glob("*.root"):
        if "data" in file.name:
            data_files.append(file)
        else:
            total += 1
            dsid = int(str(file).split(".")[0].split("_")[-1])
            if dsid_store is not None:
                if dsid not in dsid_store:
                    continue

                sample_name = dsid_store.get_sample(dsid)
                if sample_name not in accepted_mc_dct:
                    accepted_mc_dct[sample_name] = []
                accepted_mc_dct[sample_name].append(dsid)

            mc_files.append(file)

            accept += 1

    logging.info(f"Accepted {accept}/{total} MC files based on provided samples.")

    accepted_str = "MC samples:\n"
    for sample_name, dsids in accepted_mc_dct.items():
        accepted_str += f" {sample_name}:\n"
        for dsid in dsids:
            accepted_str += f"  {dsid}\n"

    logging.info(accepted_str)

    files_found = len(data_files) + len(mc_files)

    if files_found == 0:
        raise RuntimeError(f"No files found in {input_dir}!")

    logging.info(f"Found {len(data_files)} data files and {len(mc_files)} MC files (total {files_found}).")

    data_dataset = RootPhysicsDataset(f"Data{name}", [str(f) for f in data_files], is_data=True)
    mc_dataset = RootPhysicsDataset(f"MC{name}", [str(f) for f in mc_files], is_data=False)

    return data_dataset, mc_dataset, dsid_store


def build_hdf_writer_analysis(
    analysis_dir: Path,
    ntuples_dir: Path,
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
    mt_cut: tuple[float, float] | None = None,
    dsid_store: DSIDStore | None = None,
) -> tuple[ProcessorsCollection, ProcessorsGraph, PostprocessorsGraph]:
    metadata = OpendataMetadata(ntuples_dir)
    metadata.save_as_json(analysis_dir / "metadata_summary.json")

    analysis_collection = ProcessorsCollection("analysisCollection")

    analysis_collection += BranchesProcessor(str(analysis_dir), contents_config)
    analysis_collection += WeightProcessor(metadata)
    analysis_collection += LeptonProcessor(contents_config.particle_type)
    analysis_collection += LeptonTypeProcessor(contents_config.particle_type)
    analysis_collection += PreselectionProcessor(
        contents_config.particle_type,
        pt_min=10.0,
        bjet_veto=True,
        btag_quantile_working_point=3,
    )
    analysis_collection += TriggerProcessor(contents_config.particle_type)
    analysis_collection += AbsEtaProcessor()
    analysis_collection += NJetsProcessor(njets_cut)

    if met_cut is not None:
        analysis_collection += METProcessor(met_cut)

    analysis_collection += RavelProcessor()
    analysis_collection += CrackVetoProcessor()
    analysis_collection += DataMCProcessor()

    if scale_wzjets:
        logging.info("Scaling Wjets and Zjets events by k-factor.")
        if dsid_store is None:
            raise ValueError("dsid_store must be provided to scale W/Z+jets samples!")
        analysis_collection += ScaleWZjetsProcessor(dsid_store)

    if scale_dct is not None:
        analysis_collection += ScaleBranchProcessor(scale_dct)

    if mt_cut is not None:
        logging.info(f"Applying transverse mass cut: {mt_cut}.")

    analysis_collection += TransverseMassProcessor(mt_cut)

    analysis_graph = ProcessorsGraph()
    analysis_graph.add(
        CheckpointProcessor("input"),
        *analysis_collection.as_list(),
        CheckpointProcessor("output", save_arrays=True),
    )
    analysis_graph.chain()

    analysis_graph.draw(Path(os.environ["ANALYSIS_ML_LOGS_DIR"]) / "hdf_analysis_graph.pdf")

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

    postprocessors_graph.draw(Path(os.environ["ANALYSIS_ML_LOGS_DIR"]) / "hdf_post_analysis_graph.pdf")

    return analysis_collection, analysis_graph, postprocessors_graph


def convert_to_hdf5(
    *,
    analysis_dir: Path,
    ntuples_dir: Path,
    contents_config: DictConfig,
    dataloader_config: dict[str, Any],
    n_piles: int,
    pile_assignment: str,
    chunk_shape: int,
    merge_piles: bool,
    enforced_types_dct: dict[str, str] | None,
    scale_dct: dict[str, float] | None,
    scale_wzjets: bool,
    njets_cut: tuple[int, int | str] | None,
    met_cut: tuple[float, float | str] | None,
    mt_cut: tuple[float, float] | None,
    samples: list[str] | None,
    output_file: str,
) -> None:
    particle_type = contents_config.particle_type

    if particle_type not in ["el", "mu"]:
        raise ValueError(f'Invalid particle type "{particle_type}"')

    data_dataset, mc_dataset, dsid_store = build_datasets(ntuples_dir, "OpenDataDataset", samples=samples)

    dump_branches(
        data_dataset.root_files[0],
        str(analysis_dir),
        is_data=True,
        key=dataloader_config["key"],
    )
    dump_branches(
        mc_dataset.root_files[0],
        str(analysis_dir),
        is_data=False,
        key=dataloader_config["key"],
    )

    analysis_collection, analysis_graph, postprocessors_graph = build_hdf_writer_analysis(
        analysis_dir,
        ntuples_dir,
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
        mt_cut=mt_cut,
        dsid_store=dsid_store,
    )

    branch_filter = analysis_collection.branch_name_filter
    dataloader_config["filter_name"] = branch_filter

    logging.info("[green]Setting up data dataloader ...[/green]")
    data_dataset.setup_dataloader(**dataloader_config)
    data_dataset.init_dataloader(processors=analysis_graph)

    logging.info("[green]Setting up MC dataloader ...[/green]")
    mc_dataset.setup_dataloader(**dataloader_config)
    mc_dataset.init_dataloader(processors=analysis_graph)

    event_loop = ColumnarEventLoop(
        mc_datasets=[mc_dataset],
        data_datasets=[data_dataset],
        postprocessors_graph=postprocessors_graph,
        fit_postprocessors=True,
        cut_flow=False,
    )
    event_loop.run()

    postprocessors_graph["hdf5WriterPostprocessor"].close()


@hydra.main(
    config_path=str(Path(os.environ["ANALYSIS_ML_CONFIG_DIR"]) / "fakes"),
    config_name="convert_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(min_level=config.min_logging_level)

    convert_config = config.convert_config

    dataloader_config = {
        "step_size": convert_config.step_size,
        "key": convert_config.key,
        "num_workers": convert_config.num_workers,
        "dataloader_kwargs": {"multiprocessing_context": convert_config.multiprocessing_context},
    }

    if convert_config.ntuples_dir is not None:
        ntuples_dir = Path(convert_config.ntuples_dir)
    else:
        ntuples_dir = Path(os.environ["ANALYSIS_ML_NTUPLES_DIR"])

    if not ntuples_dir.exists():
        raise FileNotFoundError(f"Ntuples directory {ntuples_dir} does not exist!")

    logging.info(f"Using ntuples from {ntuples_dir}")

    if convert_config.output_dir is not None:
        output_dir = Path(convert_config.output_dir)
    else:
        output_dir = Path(os.environ["ANALYSIS_ML_DATA_DIR"])

    logging.info(f"Output dir: {output_dir}")

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist!")

    convert_to_hdf5(
        analysis_dir=output_dir,
        ntuples_dir=ntuples_dir,
        contents_config=config.hdf5_config,
        dataloader_config=dataloader_config,
        n_piles=convert_config.n_piles,
        pile_assignment=convert_config.get("pile_assignment", "random"),
        chunk_shape=convert_config.chunk_shape,
        merge_piles=convert_config.get("merge_piles", False),
        enforced_types_dct=config.hdf5_config.get("enforce_types", None),
        scale_dct=config.hdf5_config.get("scale", None),
        scale_wzjets=convert_config.get("scale_wzjets", False),
        njets_cut=convert_config.get("njets_cut", None),
        met_cut=convert_config.get("met_cut", None),
        mt_cut=convert_config.get("mT_cut", None),
        samples=config.hdf5_config.get("samples", None),
        output_file=output_dir / convert_config.output_file,
    )


if __name__ == "__main__":
    main()
