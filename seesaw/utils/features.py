import logging
from dataclasses import dataclass

import numpy as np

from seesaw.utils.helpers import bin_edges_from_centers, get_log_binning


@dataclass
class PhysicsFeature:
    name: str
    nbins: int = 10
    x_range: tuple[float, float] | None = None
    x_range_scaled: tuple[float, float] | None = None
    centers: np.ndarray | list[float | int] | None = None
    logx: bool = False
    latex_name: str | None = None
    unit: str | None = None

    def __post_init__(self):
        if self.x_range is not None and len(self.x_range) != 2:
            raise ValueError(f"x_range must be a tuple of two elements for feature '{self.name}'.")

        if self.centers is not None and self.x_range is not None:
            raise ValueError("Cannot provide both centers and x_range for the same feature.")

        if self.unit not in (None, "GeV", "TeV", "MeV", "keV"):
            raise ValueError(f"Invalid unit '{self.unit}' for feature '{self.name}'.")

    def _get_default_binning(self, scaled: bool) -> np.ndarray:
        strategy = DefaultBinningStrategy(self.name, self.nbins, self.logx)
        if scaled:
            return strategy.get_scaled_bins()
        else:
            return strategy.get_bins()

    def _get_provided_binning(self, scaled: bool) -> np.ndarray:
        if self.centers is not None:
            return bin_edges_from_centers(np.array(self.centers))

        if scaled:
            if self.x_range_scaled is None:
                raise ValueError(f"Scaled x_range is not provided for feature '{self.name}'.")

            if self.logx:
                raise ValueError("Logarithmic binning is not supported with scaled x_range.")

            x_min, x_max = self.x_range_scaled
        else:
            if self.x_range is None:
                raise ValueError(f"x_range is not provided for feature '{self.name}'.")

            x_min, x_max = self.x_range

        if self.logx and x_min <= 0:
            logging.warning(f"Logarithmic binning requires x_min > 0 for feature '{self.name}'. Using linear.")
            return np.linspace(x_min, x_max, self.nbins)

        if self.logx:
            return get_log_binning(x_min, x_max, self.nbins)
        else:
            return np.linspace(x_min, x_max, self.nbins)

    def binning(self, scaled: bool = False) -> np.ndarray:
        if self.x_range is None and self.centers is None:
            return self._get_default_binning(scaled)
        elif self.x_range is not None or self.x_range_scaled is not None or self.centers is not None:
            return self._get_provided_binning(scaled)
        else:
            raise ValueError(f"Cannot determine binning for feature '{self.name}'. ")

    def __str__(self) -> str:
        if self.latex_name is not None:
            if self.unit is not None:
                return f"{self.latex_name} [{self.unit}]"
            else:
                return self.latex_name
        else:
            return self.name

    def __repr__(self) -> str:
        return (
            f"PhysicsFeature(name={self.name}, nbins={self.nbins}, x_range={self.x_range}, "
            + f"logx={self.logx}, latex_name={self.latex_name})"
        )


class DefaultBinningStrategy:
    def __init__(self, column_name: str, nbins: int, logx: bool = False):
        self.column_name = column_name
        self.nbins = nbins
        self.logx = logx

    def get_bins(self) -> np.ndarray:
        x_range = self._get_range()

        if x_range is None:
            return self._get_categorical_range()

        x_min, x_max = x_range
        if self.logx and x_min > 0:
            return get_log_binning(x_min, x_max, self.nbins)

        return np.linspace(x_min, x_max, self.nbins)

    def get_scaled_bins(self) -> np.ndarray:
        x_range = self._get_scaled_range()

        if x_range is None:
            return self._get_categorical_scaled_range()

        x_min, x_max = x_range
        return np.linspace(x_min, x_max, self.nbins)

    def _get_range(self) -> tuple[float, float] | None:
        name = self.column_name

        if "deta" in name or "dphi" in name or "dR" in name:
            return 0.0, 5.0
        elif "eta" in name and "abs" in name:
            return 0.0, 2.47
        elif "eta" in name:
            return -2.47, 2.47
        elif "phi" in name:
            return -3.14, 3.14
        elif "mva_score" in name:
            return 0.0, 1.0
        elif "charge" in name or name.startswith("n"):
            return None
        else:
            return 10.0, 1000.0

    def _get_scaled_range(self) -> tuple[float, float] | None:
        name = self.column_name

        if any(k in name for k in ("deta", "dphi", "dR", "eta", "phi")):
            return -5.0, 5.0
        elif "mva_score" in name:
            return 0.0, 1.0
        elif "charge" in name or name.startswith("n"):
            return None
        else:
            return -2.0, 10.0

    def _get_categorical_range(self):
        name = self.column_name

        if "charge" in name:
            return bin_edges_from_centers(np.array([-1.0, 1.0]))
        else:
            return bin_edges_from_centers(np.arange(0, 10, 1))

    def _get_categorical_scaled_range(self) -> np.ndarray:
        name = self.column_name

        if "charge" in name:
            return bin_edges_from_centers(np.array([0.0, 1.0]))
        else:
            return bin_edges_from_centers(np.arange(0, 10, 1))


def get_feature(
    name: str,
    nbins: int = 10,
    x_range: tuple[float, float] | None = None,
    x_range_scaled: tuple[float, float] | None = None,
    centers: np.ndarray | list[float | int] | None = None,
    logx: bool = False,
    latex_name: str | None = None,
    unit: str | None = None,
) -> PhysicsFeature:
    feature = switch_on_feature(name, nbins)

    if x_range is not None:
        feature.x_range = x_range
    if x_range_scaled is not None:
        feature.x_range_scaled = x_range_scaled
    if centers is not None:
        feature.centers = centers
    if logx is not None:
        feature.logx = logx
    if latex_name is not None:
        feature.latex_name = latex_name
    if unit is not None:
        feature.unit = unit

    return feature


def switch_on_feature(name: str, nbins: int) -> PhysicsFeature:
    match name:
        case "pt":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(10.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$p_\mathrm{T}$",
                unit="GeV",
            )
        case "ptl1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Leading lepton $p_\mathrm{T}$",
                unit="GeV",
            )
        case "ptl2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Subleading lepton $p_\mathrm{T}$",
                unit="GeV",
            )
        case "eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"$\eta$",
            )
        case "etal1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Leading lepton $\eta$",
            )
        case "etal2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Subleading $\eta$",
            )
        case "phill":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Leading lepton $\phi$",
            )
        case "phil1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Leading lepton $\phi$",
            )
        case "phil2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Subleading lepton $\phi$",
            )
        case "pt2l":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Di-lepton $p_\mathrm{T}$",
                unit="GeV",
            )
        case "pt2lSS1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Di-lepton $p_\mathrm{T}$ SS lead. pair",
                unit="GeV",
            )
        case "pt2lOS1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Di-lepton $p_\mathrm{T}$ SS lead. pair",
                unit="GeV",
            )
        case "eta2l":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Di-lepton $\eta$",
            )
        case "phi2l":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Di-lepton $\phi$",
            )
        case "mll":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(100.0, 2100.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Di-lepton invariant mass $m_{\ell\ell}$",
                unit="GeV",
            )
        case "mllSS1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(100.0, 2100.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$m_{\ell\ell}$ SS lead. pair",
                unit="GeV",
            )
        case "mllOS1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(100.0, 2100.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$m_{\ell\ell}$ OS lead. pair",
                unit="GeV",
            )
        case "dR2l12":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 5.0),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"$\Delta R(\mathrm{lead. lep.}, \mathrm{sublead. lep.})$",
            )
        case "dR2lOS1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 5.0),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"$\Delta R(\mathrm{lead. lep.}, \mathrm{sublead. lep.})$ OS lead. pair",
            )
        case "deta2l12":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 5.0),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"$\Delta \eta(\mathrm{lead. lep.}, \mathrm{sublead. lep.})$",
            )
        case "dphi2l12":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"$\Delta \phi(\mathrm{lead. lep.}, \mathrm{sublead. lep.})$",
            )
        case "met":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(10.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Missing $E_\mathrm{T}$",
                unit="GeV",
            )
        case "metphi":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Missing $E_\mathrm{T}$ $\phi$",
            )
        case "metsig":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 100.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$E_\mathrm{T}^\mathrm{miss}$ significance",
            )
        case "htmet":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$H_\mathrm{T} + E_\mathrm{T}^\mathrm{miss}$",
                unit="GeV",
            )
        case "htlepmet":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Lepton $H_\mathrm{T} + E_\mathrm{T}^\mathrm{miss}$",
                unit="GeV",
            )
        case "ht":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 3000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$H_\mathrm{T}$",
                unit="GeV",
            )
        case "htlep":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Lepton $H_\mathrm{T}$",
                unit="GeV",
            )
        case "ptj1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Leading light jet $p_\mathrm{T}$",
                unit="GeV",
            )
        case "ptj2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Subleading light jet $p_\mathrm{T}$",
                unit="GeV",
            )
        case "ptj3":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Third light jet $p_\mathrm{T}$",
                unit="GeV",
            )
        case "etaj1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Leading light jet $\eta$",
            )
        case "etaj2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Subleading light jet $\eta$",
            )
        case "etaj3":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Third light jet $\eta$",
            )
        case "phij1":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Leading light jet $\phi$",
            )
        case "phij2":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Subleading light jet $\phi$",
            )
        case "phij3":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Third light jet $\phi$",
            )
        case "pt2j":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Light di-jet $p_\mathrm{T}$",
                unit="GeV",
            )
        case "eta2j":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Light di-jet $\eta$",
            )
        case "phi2j":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Light di-jet $\phi$",
            )
        case "mjj":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Light di-jet invariant mass $m_{jj}$",
                unit="GeV",
            )
        case "ht2j":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Light di-jet $H_\mathrm{T}$",
                unit="GeV",
            )
        case "m2j2l":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 4000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$m_{jjll}$",
                unit="GeV",
            )
        case "m2jll":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 3000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$m_{jjl_\mathrm{lead}}$",
                unit="GeV",
            )
        case "m2jsl":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(100.0, 3000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"$m_{jjl_\mathrm{sublead}}$",
                unit="GeV",
            )
        case "mt":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 3000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Transverse mass $m_T$",
                unit="GeV",
            )
        case "m_t":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 70.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Transverse mass $m_T$",
                unit="GeV",
            )
        case "mtlep":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Lepton transverse mass $m_T$",
                unit="GeV",
            )
        case "njets_light":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.arange(0, 10, 1),
                latex_name="Number of light jets",
            )
        case "njets":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.arange(0, 10, 1),
                latex_name="Number of jets",
            )
        case "nbjets":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.arange(0, 10, 1),
                latex_name="Number of b-jets",
            )
        case "nwbosons":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.arange(0, 10, 1),
                latex_name=r"$W$ boson multiplicity",
            )
        case "mlmass":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(500.0, 2100.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name="ML mass",
                unit="GeV",
            )
        case "secondary_mlmass":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name="Secondary ML mass",
                unit="GeV",
            )
        case "mva_score":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 1.0),
                x_range_scaled=(0.0, 1.0),
                latex_name="MVA binary score",
            )
        case "el_pt":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(10.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Electron $p_\mathrm{T}$",
                unit="GeV",
            )
        case "el_abs_caloCluster_eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Electron |$\eta$|",
            )
        case "el_caloCluster_eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Electron $\eta$",
            )
        case "el_eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Electron $\eta$",
            )
        case "el_phi":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Electron $\phi$",
            )
        case "el_charge":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.array([-1.0, 1.0]),
                latex_name=r"Electron charge",
            )
        case "mu_pt":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(10.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Muon $p_\mathrm{T}$",
                unit="GeV",
            )
        case "mu_eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-2.47, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Muon $\eta$",
            )
        case "mu_abs_eta":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(0.0, 2.47),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Muon |$\eta$|",
            )
        case "mu_phi":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(-3.14, 3.14),
                x_range_scaled=(-5.0, 5.0),
                latex_name=r"Muon $\phi$",
            )
        case "mu_charge":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                centers=np.array([-1.0, 1.0]),
                latex_name=r"Muon charge",
            )
        case "toy_pt":
            return PhysicsFeature(
                name=name,
                nbins=nbins,
                x_range=(10.0, 1000.0),
                x_range_scaled=(-2.0, 10.0),
                latex_name=r"Toy $p_\mathrm{T}$",
                unit="GeV",
            )
        case _:
            return PhysicsFeature(name=name, nbins=nbins)
