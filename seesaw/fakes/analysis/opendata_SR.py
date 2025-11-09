import copy
import logging
import os
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import hydra
import mplhep as hep
import numpy as np
import plothist
import torch
from f9columnar.dataset_builder import RootPhysicsDataset
from f9columnar.histograms import HistogramProcessor, NtupleHistogramMerger
from f9columnar.ml.scalers import CategoricalFeatureScaler, NumericalFeatureScaler
from f9columnar.processors import (
    CheckpointPostprocessor,
    CheckpointProcessor,
    PostprocessorsGraph,
    Processor,
    ProcessorsGraph,
)
from f9columnar.processors_collection import ProcessorsCollection
from f9columnar.run import ColumnarEventLoop
from f9columnar.utils.helpers import load_pickle
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from seesaw.fakes.analysis.ratio_closure import BinnedFakeFactorIndexer
from seesaw.fakes.dataset.hdf5_converter import AbsEtaProcessor, CrackVetoProcessor
from seesaw.fakes.dataset.hdf5_opendata_converter import (
    LeptonProcessor,
    METProcessor,
    NJetsProcessor,
    OpendataMetadata,
    PreselectionProcessor,
    RavelProcessor,
    ScaleWZjetsProcessor,
    TransverseMassProcessor,
    TriggerProcessor,
    WeightProcessor,
)
from seesaw.fakes.models.dre_classifiers import RatioClassifier
from seesaw.models.utils import load_model_from_config
from seesaw.utils.dsid_store import DSIDStore
from seesaw.utils.helpers import load_dataset_column_from_config, setup_analysis_dirs
from seesaw.utils.loggers import setup_logger
from seesaw.utils.plots_utils import get_color


class BranchesProcessor(Processor):
    def __init__(self, contents_config: DictConfig) -> None:
        super().__init__(name="branchesProcessor")

        self.particle_type = contents_config.particle_type

        input_data_branches, input_mc_branches = self._get_hdf_inputs(contents_config)

        self.branch_name = list(set(input_data_branches + input_mc_branches))

    def _get_hdf_inputs(self, contents_config) -> tuple[list[str], list[str]]:
        input_data_branches = contents_config["data"]
        input_mc_branches = contents_config["mc"]

        self._validate_input_branches(input_data_branches, input_mc_branches)

        return input_data_branches, input_mc_branches

    def _validate_input_branches(self, input_data_branches: list[str], input_mc_branches: list[str]) -> None:
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


class LeptonTypeProcessor(Processor):
    def __init__(
        self, name: str, particle_type: str, return_selection: str, id_wp: str = "tight", iso_wp: str = "loose"
    ) -> None:
        super().__init__(name=name)
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

        if return_selection not in ["tight", "sloose"]:
            raise ValueError(f"Invalid return_selection: {return_selection}")

        self.return_selection = return_selection

        logging.info(f"Using {particle_type} ID: {self.id_wp} and isolation: {self.iso_wp}.")

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        id_mask = ak.values_astype(arrays[self.id_wp], np.bool_)
        iso_mask = ak.values_astype(arrays[self.iso_wp], np.bool_)

        type_mask = id_mask & iso_mask

        for branch in ["lep_isTightID", "lep_isMediumID", "lep_isLooseID", "lep_isTightIso", "lep_isLooseIso"]:
            arrays = ak.without_field(arrays, branch)

        return {"arrays": arrays[type_mask] if self.return_selection == "tight" else arrays[~type_mask]}


class FakeFactorProcessor(Processor):
    def __init__(self, ml_ff: bool, binned_ff_path: Path | None, config: DictConfig) -> None:
        super().__init__(name="fakeFactorProcessor")

        self.ml_ff = ml_ff

        if self.ml_ff:
            logging.info("[yellow]Using ML-based fake factor.")
            self.model = None
            events_column = load_dataset_column_from_config(config, "events")
            self.offset_used_columns = events_column.offset_used_columns
            self.numer_colums = events_column.numer_columns
            self.categ_colums = events_column.categ_columns
            self.numer_scaler = self._get_numer_scaler(
                config.dataset_config, self.numer_colums, extra_hash=config.dataset_config.files
            )
            self.categ_scaler = self._get_categ_scaler(
                config.dataset_config, self.categ_colums, extra_hash=config.dataset_config.files
            )
        else:
            if binned_ff_path is None:
                raise ValueError("binned_ff_path needs to be provided")

            logging.info("[yellow]Using binned fake factor.")
            self.binned_ff = BinnedFakeFactorIndexer(**load_pickle(str(binned_ff_path)))

    def _get_numer_scaler(
        self, dataset_config: DictConfig, numer_column_names: list[str], extra_hash: str = ""
    ) -> NumericalFeatureScaler | None:
        scaler_type = dataset_config.feature_scaling.scaler_type
        scaler_path = dataset_config.feature_scaling.save_path

        numer_scaler = NumericalFeatureScaler(scaler_type, save_path=scaler_path).load(
            column_names=numer_column_names, postfix="events_0", extra_hash=extra_hash
        )

        if numer_scaler is None:
            return None

        return numer_scaler

    def _get_categ_scaler(
        self, dataset_config: DictConfig, numer_column_names: list[str], extra_hash: str = ""
    ) -> CategoricalFeatureScaler | None:
        scaler_path = dataset_config.feature_scaling.save_path

        categ_scaler = CategoricalFeatureScaler("categorical", save_path=scaler_path).load(
            column_names=numer_column_names, postfix="events_0", extra_hash=extra_hash
        )

        if categ_scaler is None:
            return None

        return categ_scaler

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        if self.ml_ff:
            numer_features = np.stack([arrays[col].to_numpy() for col in self.numer_colums], axis=-1)
            categ_features = np.stack([arrays[col].to_numpy() for col in self.categ_colums], axis=-1)

            if self.numer_scaler is not None:
                numer_features = self.numer_scaler.transform(numer_features)
            if self.categ_scaler is not None:
                categ_features = self.categ_scaler.transform(categ_features)

            features = np.concatenate([numer_features, categ_features], axis=-1, dtype=np.float32)

            with torch.no_grad():
                fake_factor = np.exp(self.model(torch.from_numpy(features)).squeeze().numpy())  # type: ignore[misc]

        else:
            fake_factor = self.binned_ff.get(arrays["el_pt"], arrays["el_abs_eta"], arrays["met"])[0]

        arrays["weights"] = arrays["weights"] * fake_factor

        if not self.is_data:
            arrays["weights"] = -arrays["weights"]

        return {"arrays": arrays}


class OpendataSRHistsProcessor(HistogramProcessor):
    def __init__(self, name: str, save_as_fakes: bool = False) -> None:
        super().__init__(name=name, as_data=True)

        self.save_as_fakes = save_as_fakes

        self.make_hist1d("el_pt", 500, 0.0, 500.0)
        self.make_hist1d("el_eta", 300, -2.5, 2.5)
        self.make_hist1d("m_t", 300, 0.0, 300.0)
        self.make_hist1d("met", 300, 0.0, 300.0)
        self.make_hist1d("njets", 11, -0.5, 10.5)

    def run(self, arrays: ak.Array) -> dict[str, ak.Array]:
        metadata = {
            "name": "fakes" if self.save_as_fakes else self.reports["name"],
        }

        for hist_name in self.hists.keys():
            if hist_name in arrays.fields:
                self.fill_hist1d(hist_name, arrays[hist_name], weight=arrays["weights"], metadata=metadata)

        return {"arrays": arrays}


def build_datasets(
    input_dir: Path, samples: list[str]
) -> tuple[RootPhysicsDataset, list[RootPhysicsDataset], DSIDStore | None]:
    dsid_store = DSIDStore(samples)

    data_files: list[Path] = []
    mc_files: dict[str, list[Path]] = {}
    accepted_mc_dct: dict[str, list[int]] = {}

    accept, total = 0, 0
    for file in input_dir.glob("*.root"):
        if "data" in file.name:
            data_files.append(file)
        else:
            total += 1

            dsid = int(str(file).split(".")[0].split("_")[-1])
            if dsid not in dsid_store:
                continue

            sample_name = dsid_store.get_sample(dsid)

            if sample_name not in accepted_mc_dct:
                accepted_mc_dct[sample_name] = []
            accepted_mc_dct[sample_name].append(dsid)

            if sample_name not in mc_files:
                mc_files[sample_name] = []
            mc_files[sample_name].append(file)
            accept += 1

    logging.info(f"Accepted {accept}/{total} MC files based on provided samples.")

    accepted_str = "MC samples:\n"
    for sample_name, dsids in accepted_mc_dct.items():
        accepted_str += f" {sample_name}:\n"
        for dsid in dsids:
            accepted_str += f"  {dsid}\n"

    logging.info(accepted_str)

    files_found = len(data_files) + accept

    if files_found == 0:
        raise RuntimeError(f"No files found in {input_dir}!")

    logging.info(f"Found {len(data_files)} data files and {accept} MC files (total {files_found}).")

    data_dataset = RootPhysicsDataset("data", [str(f) for f in data_files], is_data=True)
    mc_datasets = [
        RootPhysicsDataset(sample_name, [str(f) for f in files], is_data=False)
        for sample_name, files in mc_files.items()
    ]

    return data_dataset, mc_datasets, dsid_store


def build_opendata_SR_analysis(
    *,
    ntuples_dir: Path,
    output_dir: Path,
    contents_config: DictConfig,
    analysis_config: DictConfig,
    config: DictConfig,
    dsid_store: DSIDStore | None = None,
) -> tuple[ProcessorsCollection, ProcessorsGraph, PostprocessorsGraph]:
    metadata = OpendataMetadata(ntuples_dir)

    model = None
    if analysis_config.fakes_method == "ml":
        model, _ = load_model_from_config(config, RatioClassifier)
        model = model.cpu().eval()

    analysis_collection = ProcessorsCollection("analysisCollection")

    analysis_collection += BranchesProcessor(contents_config)
    analysis_collection += WeightProcessor(metadata)
    analysis_collection += LeptonProcessor(contents_config.particle_type)
    analysis_collection += PreselectionProcessor(
        contents_config.particle_type,
        pt_min=analysis_config.el_pt_min,
        bjet_veto=True,
        btag_quantile_working_point=3,
    )
    analysis_collection += TriggerProcessor(contents_config.particle_type)
    analysis_collection += AbsEtaProcessor()
    analysis_collection += NJetsProcessor(analysis_config.njets_cut)

    if analysis_config.met_cut is not None:
        analysis_collection += METProcessor(analysis_config.met_cut)

    analysis_collection += RavelProcessor()
    analysis_collection += CrackVetoProcessor()

    if analysis_config.scale_wzjets:
        logging.info("Scaling Wjets and Zjets events by k-factor.")
        if dsid_store is None:
            raise ValueError("dsid_store must be provided to scale W/Z+jets samples!")
        analysis_collection += ScaleWZjetsProcessor(dsid_store)

    analysis_collection += TransverseMassProcessor(mt_cut=analysis_config.mT_cut)

    analysis_collection += LeptonTypeProcessor("leptonTypeProcessor_tight", contents_config.particle_type, "tight")
    analysis_collection += LeptonTypeProcessor("leptonTypeProcessor_sloose", contents_config.particle_type, "sloose")

    analysis_collection += FakeFactorProcessor(
        ml_ff=True if analysis_config.fakes_method == "ml" else False,
        binned_ff_path=analysis_config.binned_ff_path,
        config=config,
    )

    analysis_collection += OpendataSRHistsProcessor("HistsProcessor_tight")
    analysis_collection += OpendataSRHistsProcessor("HistsProcessor_fakes", save_as_fakes=True)

    analysis_graph = ProcessorsGraph(
        global_attributes={
            "model": model,
        }
    )
    analysis_graph.add(
        CheckpointProcessor("input"),
        *analysis_collection.as_list(),
    )
    analysis_graph.connect(
        [
            ("input", "branchesProcessor"),
            ("branchesProcessor", "weightProcessor"),
            ("weightProcessor", "leptonProcessor"),
            ("leptonProcessor", "preselectionProcessor"),
            ("preselectionProcessor", "triggerProcessor"),
            ("triggerProcessor", "absEtaProcessor"),
            ("absEtaProcessor", "nJetsProcessor"),
            ("nJetsProcessor", "ravelProcessor"),
            ("ravelProcessor", "crackVetoProcessor"),
            ("crackVetoProcessor", "scaleWZjetsProcessor"),
            ("scaleWZjetsProcessor", "transverseMassProcessor"),
            ("transverseMassProcessor", "leptonTypeProcessor_tight"),
            ("transverseMassProcessor", "leptonTypeProcessor_sloose"),
            ("leptonTypeProcessor_tight", "HistsProcessor_tight"),
            ("leptonTypeProcessor_sloose", "fakeFactorProcessor"),
            ("fakeFactorProcessor", "HistsProcessor_fakes"),
        ]
    )
    analysis_graph.draw(output_dir / "opendata_SR_analysis_graph.pdf")

    hist_writer = NtupleHistogramMerger(
        name="histMerger_tight",
        save_path=str(output_dir / "hists.p"),
        data_hist_name="HistsProcessor_tight",
        mc_hist_name="HistsProcessor_tight",
        merge_years=False,
        merge_campaigns=False,
    )
    hist_writer_fakes = NtupleHistogramMerger(
        name="histMerger_fakes",
        save_path=str(output_dir / "hists_fakes.p"),
        data_hist_name="HistsProcessor_fakes",
        mc_hist_name="HistsProcessor_fakes",
        merge_years=False,
        merge_campaigns=False,
    )

    postprocessors_graph = PostprocessorsGraph()
    postprocessors_graph.add(
        CheckpointPostprocessor("input"),
        hist_writer,
        hist_writer_fakes,
    )
    postprocessors_graph.chain()

    return analysis_collection, analysis_graph, postprocessors_graph


def fill_SR_hists(
    *,
    ntuples_dir: Path,
    output_dir: Path,
    contents_config: DictConfig,
    dataloader_config: dict[str, Any],
    analysis_config: DictConfig,
    config: DictConfig,
    samples: list[str],
) -> None:
    particle_type = contents_config.particle_type

    if particle_type not in ["el", "mu"]:
        raise ValueError(f'Invalid particle type "{particle_type}"')

    data_dataset, mc_datasets, dsid_store = build_datasets(ntuples_dir, samples=samples)

    analysis_collection, analysis_graph, postprocessors_graph = build_opendata_SR_analysis(
        ntuples_dir=ntuples_dir,
        output_dir=output_dir,
        contents_config=contents_config,
        analysis_config=analysis_config,
        config=config,
        dsid_store=dsid_store,
    )

    branch_filter = analysis_collection.branch_name_filter
    dataloader_config["filter_name"] = branch_filter

    logging.info("[green]Setting up data dataloader ...[/green]")
    data_dataset.setup_dataloader(**dataloader_config)
    data_dataset.init_dataloader(processors=analysis_graph)

    logging.info("[green]Setting up MC dataloaders ...[/green]")
    for mc_dataset in mc_datasets:
        mc_dataset.setup_dataloader(**dataloader_config)
        mc_dataset.init_dataloader(processors=analysis_graph)

    event_loop = ColumnarEventLoop(
        mc_datasets=mc_datasets,
        data_datasets=[data_dataset],
        postprocessors_graph=postprocessors_graph,
        fit_postprocessors=True,
        cut_flow=False,
    )
    event_loop.run()


def prepare_hists(
    plotting_config: DictConfig, hists: dict[str, dict[str, hist.Hist]], fakes: dict[str, dict[str, hist.Hist]]
) -> tuple[dict[str, hist.Hist], dict[str, dict[str, hist.Hist]], list[str], list[tuple[float, float, float]]]:
    mc = {}

    for sample in plotting_config["mc_samples"].keys():
        # Fakes are read from a separate file
        if sample == "fakes":
            mc[sample] = fakes[sample]
            continue

        # Sum up the two singletop samples
        if sample == "singletop":
            mc[sample] = hists["singletop_inclusive_noWt"]
            for var, hist in mc[sample].items():
                hist += hists["singletop_inclusive_Wt_DR_dyn"][var]
            continue

        mc[sample] = hists[sample]

    mc_labels = [sample["label"] for sample in plotting_config.mc_samples.values()]
    mc_colors = [get_color(sample["color"]).rgb for sample in plotting_config.mc_samples.values()]

    return hists["data"], mc, mc_labels, mc_colors


def plot_SR_hists(save_dir: Path, plotting_config: DictConfig, mt_cut: list[float] | None = None) -> None:
    data, mc, mc_labels, mc_colors = prepare_hists(
        plotting_config, load_pickle(str(save_dir / "hists.p")), load_pickle(str(save_dir / "hists_fakes.p"))
    )

    figs = []

    for var, config in plotting_config.variables.items():
        # Adjust plotting range
        if "range" in config and config.range is not None:
            for h in [data] + [mc[sample] for sample in mc.keys()]:
                h[var] = h[var][hist.loc(config.range[0]) : hist.loc(config.range[1])]  # type: ignore

        # Rebin
        if "N_bins" in config and config.N_bins is not None:
            for h in [data] + [mc[sample] for sample in mc.keys()]:
                axis_edges = h[var].axes[0].edges
                hist_range = axis_edges[-1] - axis_edges[0]
                new_bin_width = hist_range / config.N_bins
                rebin = int(new_bin_width / h[var].axes[0].widths[0])
                h[var] = h[var][hist.rebin(rebin)]  # type: ignore

        # Plot
        fig, ax, ax_ratio = plothist.plot_data_model_comparison(
            data_hist=data[var],
            stacked_components=[mc[sample][var] for sample in mc.keys()][::-1],
            stacked_labels=mc_labels[::-1],
            stacked_colors=mc_colors[::-1],
            xlabel=config["xlabel"],
            ylabel="Events / Bin",
            comparison="split_ratio",
            model_uncertainty=True,
            data_uncertainty_type="symmetrical",
        )

        fig.set_size_inches(7, 7)

        ax.legend(loc="upper right", fontsize=11, ncol=2)
        ax.text(0.04, 0.91, r"$\sqrt{s}=13\,\mathrm{TeV}$, $36\,\mathrm{fb}^{-1}$", transform=ax.transAxes, fontsize=14)

        ax_ratio.set_ylim(0.8, 1.2)

        if ax.get_ylim()[0] < 0.0:
            ax.set_ylim(ymin=0.0)

        figs.append(copy.deepcopy(fig))

        ax.set_ylim(auto=True)
        ax.set_yscale("log")

        ax.set_ylim(ymax=ax.get_ylim()[1] * 10.0)

        if ax.get_ylim()[0] < 0.1:
            ax.set_ylim(ymin=0.1)

        figs.append(fig)

    save_name = "opendata_SR_region"
    if mt_cut is not None:
        save_name += f"_mT_{mt_cut[0]:.0f}"
        save_name += f"-{mt_cut[1]:.0f}"

    with PdfPages(save_dir / f"{save_name}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@hydra.main(
    config_path=str(Path(os.environ["ANALYSIS_ML_CONFIG_DIR"]) / "fakes"),
    config_name="opendata_SR",
    version_base=None,
)
def main(config: DictConfig) -> None:
    hep.style.use(hep.style.ATLAS)
    setup_logger(min_level=config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

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

    output_dir = (
        Path(convert_config.output_dir)
        if convert_config.output_dir is not None
        else Path(os.environ["ANALYSIS_ML_RESULTS_DIR"]) / "opendata_SR"
    )
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    if config.fill_histograms:
        fill_SR_hists(
            ntuples_dir=ntuples_dir,
            output_dir=output_dir,
            contents_config=config.hdf5_config,
            dataloader_config=dataloader_config,
            analysis_config=config.analysis_config,
            config=config,
            samples=config.hdf5_config.get("samples", None),
        )

    plot_SR_hists(output_dir, config.plotting_config, mt_cut=config.analysis_config.mT_cut)


if __name__ == "__main__":
    main()
