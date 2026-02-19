from dataclasses import dataclass

from seesawml.utils.plots_utils import get_color, get_uniform_color


@dataclass(frozen=True)
class Label:
    key: str
    value: str
    latex_value: str | None = None
    color: str | None = None
    is_signal: bool = False

    @property
    def latex_name(self) -> str:
        if self.latex_value is not None:
            return self.latex_value
        else:
            return self.value


def get_label(key: str) -> Label:
    return switch_on_label(key)


def switch_on_label(key: str) -> Label:
    if "typeIIseesaw" in key:
        seesaw_colors = get_uniform_color("cubehelix", 16)

    match key:
        case "typeIIseesaw_500":
            return Label(
                key,
                "Type-II Seesaw 500 GeV",
                r"$m_{H^{\pm\pm}}=500$ GeV",
                is_signal=True,
                color=seesaw_colors[0].hex,
            )
        case "typeIIseesaw_600":
            return Label(
                key,
                "Type-II Seesaw 600 GeV",
                r"$m_{H^{\pm\pm}}=600$ GeV",
                is_signal=True,
                color=seesaw_colors[1].hex,
            )
        case "typeIIseesaw_700":
            return Label(
                key,
                "Type-II Seesaw 700 GeV",
                r"$m_{H^{\pm\pm}}=700$ GeV",
                is_signal=True,
                color=seesaw_colors[2].hex,
            )
        case "typeIIseesaw_800":
            return Label(
                key,
                "Type-II Seesaw 800 GeV",
                r"$m_{H^{\pm\pm}}=800$ GeV",
                is_signal=True,
                color=seesaw_colors[3].hex,
            )
        case "typeIIseesaw_900":
            return Label(
                key,
                "Type-II Seesaw 900 GeV",
                r"$m_{H^{\pm\pm}}=900$ GeV",
                is_signal=True,
                color=seesaw_colors[4].hex,
            )
        case "typeIIseesaw_1000":
            return Label(
                key,
                "Type-II Seesaw 1000 GeV",
                r"$m_{H^{\pm\pm}}=1000$ GeV",
                is_signal=True,
                color=seesaw_colors[5].hex,
            )
        case "typeIIseesaw_1100":
            return Label(
                key,
                "Type-II Seesaw 1100 GeV",
                r"$m_{H^{\pm\pm}}=1100$ GeV",
                is_signal=True,
                color=seesaw_colors[6].hex,
            )
        case "typeIIseesaw_1200":
            return Label(
                key,
                "Type-II Seesaw 1200 GeV",
                r"$m_{H^{\pm\pm}}=1200$ GeV",
                is_signal=True,
                color=seesaw_colors[7].hex,
            )
        case "typeIIseesaw_1300":
            return Label(
                key,
                "Type-II Seesaw 1300 GeV",
                r"$m_{H^{\pm\pm}}=1300$ GeV",
                is_signal=True,
                color=seesaw_colors[8].hex,
            )
        case "typeIIseesaw_1400":
            return Label(
                key,
                "Type-II Seesaw 1400 GeV",
                r"$m_{H^{\pm\pm}}=1400$ GeV",
                is_signal=True,
                color=seesaw_colors[9].hex,
            )
        case "typeIIseesaw_1500":
            return Label(
                key,
                "Type-II Seesaw 1500 GeV",
                r"$m_{H^{\pm\pm}}=1500$ GeV",
                is_signal=True,
                color=seesaw_colors[10].hex,
            )
        case "typeIIseesaw_1600":
            return Label(
                key,
                "Type-II Seesaw 1600 GeV",
                r"$m_{H^{\pm\pm}}=1600$ GeV",
                is_signal=True,
                color=seesaw_colors[11].hex,
            )
        case "typeIIseesaw_1700":
            return Label(
                key,
                "Type-II Seesaw 1700 GeV",
                r"$m_{H^{\pm\pm}}=1700$ GeV",
                is_signal=True,
                color=seesaw_colors[12].hex,
            )
        case "typeIIseesaw_1800":
            return Label(
                key,
                "Type-II Seesaw 1800 GeV",
                r"$m_{H^{\pm\pm}}=1800$ GeV",
                is_signal=True,
                color=seesaw_colors[13].hex,
            )
        case "typeIIseesaw_1900":
            return Label(
                key,
                "Type-II Seesaw 1900 GeV",
                r"$m_{H^{\pm\pm}}=1900$ GeV",
                is_signal=True,
                color=seesaw_colors[14].hex,
            )
        case "typeIIseesaw_2000":
            return Label(
                key,
                "Type-II Seesaw 2000 GeV",
                r"$m_{H^{\pm\pm}}=2000$ GeV",
                is_signal=True,
                color=seesaw_colors[15].hex,
            )
        case "VH_inclusive":
            return Label(
                key,
                "VH Inclusive",
                r"$\mathrm{VH}$ Inclusive",
                color=get_color("RedDarker").hex,
            )
        case "ttbar_dilepton":
            return Label(
                key,
                "ttbar Dilepton",
                r"$\mathrm{t\bar{t}}$ Dilepton",
                color=get_color("AltDeepGreen").hex,
            )
        case "ttbar_inclusive":
            return Label(
                key,
                "ttbar Inclusive",
                r"$\mathrm{t\bar{t}}$ Inclusive",
                color=get_color("AltGreen").hex,
            )
        case "singletop_inclusive":
            return Label(
                key,
                "Single Top Inclusive",
                "Single Top Inclusive",
                color=get_color("GreenDarker").hex,
            )
        case "singletop_dilepton":
            return Label(
                key,
                "Single Top Dilepton",
                "Single Top Dilepton",
                color=get_color("PinkDarker").hex,
            )
        case "othertop":
            return Label(
                key,
                "Other Top",
                "Other Top",
                color=get_color("Red").hex,
            )
        case "Zjets":
            return Label(
                key,
                "Drell-Yan",
                "Drell-Yan",
                color=get_color("Yellow").hex,
            )
        case "Wjets":
            return Label(
                key,
                "W+jets",
                "W+jets",
                color=get_color("Orange").hex,
            )
        case "diboson":
            return Label(
                key,
                "Diboson",
                "Diboson",
                color=get_color("Blue").hex,
            )
        case "dijet":
            return Label(
                key,
                "QCD",
                "QCD",
                color=get_color("Purple").hex,
            )
        case "top":
            return Label(
                key,
                "Top",
                "Top",
                color=get_color("Red").hex,
            )
        case "background":
            return Label(
                key,
                "Background",
                "Background",
                color=get_color("Gray").hex,
            )
        case "data":
            return Label(key, "Data", "Data")
        case "signal":
            return Label(key, "Signal", "Signal", is_signal=True)
        case _:
            return Label(key, key)
