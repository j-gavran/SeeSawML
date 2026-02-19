from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from plothist import get_color_palette


def save_plot(fig: plt.Figure, save_path: str, use_format: str = "pdf", dpi: float = 300.0) -> None:
    fig.tight_layout()
    fig.savefig(save_path, format=use_format, dpi=dpi)
    plt.close(fig)


def make_subplots_grid(n_plots: int, ratio: float = 1 / 1.618) -> tuple[int, int]:
    n_cols = int(np.sqrt(n_plots / ratio))
    n_rows = int(n_plots / n_cols)

    if n_cols * n_rows < n_plots:
        n_rows += 1

    return n_rows, n_cols


def iqr_remove_outliers(data: np.ndarray, q1_set: float = 5.0, q3_set: float = 95.0) -> np.ndarray:
    q1 = np.percentile(data, q1_set)
    q3 = np.percentile(data, q3_set)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


def iqr_remove_outliers_mask(data: np.ndarray, q1_set: float = 5.0, q3_set: float = 95.0) -> np.ndarray:
    q1 = np.percentile(data, q1_set)
    q3 = np.percentile(data, q3_set)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data >= lower_bound) & (data <= upper_bound)


def atlas_label(
    ax: plt.Axes, loc: int = 0, llabel: str = "Internal", rlabel: str = "", fontsize: int = 12, **kwargs: Any
) -> None:
    hep.atlas.label(
        loc=loc,
        llabel=llabel,
        rlabel=rlabel,
        ax=ax,
        fontsize=fontsize,
        fontname="Latin Modern sans",
        **kwargs,
    )


@dataclass
class Color:
    name: str
    rgb_value: tuple[float, float, float] | None = None
    hex_value: str | None = None

    @property
    def hex(self) -> str:
        if self.hex_value is not None:
            return self.hex_value

        if self.rgb_value is None:
            raise ValueError("RGB value must be set to compute hex value!")

        return (
            f"#{int(self.rgb_value[0] * 255):02x}{int(self.rgb_value[1] * 255):02x}{int(self.rgb_value[2] * 255):02x}"
        )

    @property
    def rgb(self) -> tuple[float, float, float]:
        if self.rgb_value is not None:
            return self.rgb_value

        if self.hex_value is None:
            raise ValueError("Hex value must be set to compute RGB value!")

        hex_value = self.hex_value.lstrip("#")
        return tuple(int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


@dataclass
class UniformColor:
    name: str
    colors: list[Color]

    @property
    def rgb(self) -> list[tuple[float, float, float]]:
        return [color.rgb for color in self.colors]

    @property
    def hex(self) -> list[str]:
        return [color.hex for color in self.colors]

    def __getitem__(self, idx: int) -> Color:
        if idx < 0 or idx >= len(self.colors):
            raise IndexError("Index out of range for UniformColor colors list!")
        return self.colors[idx]


def get_color(name: str) -> Color:
    return switch_on_color(name)


def get_uniform_color(name: str, n) -> UniformColor:
    color_palette = get_color_palette(name, n)
    return UniformColor(name, [Color(name=f"{name}_{i}", rgb_value=color) for i, color in enumerate(color_palette)])


def switch_on_color(name: str) -> Color:
    match name:
        case "Black":
            return Color(name, rgb_value=(0, 0, 0))
        case "Green":
            return Color(name, rgb_value=(154 / 255, 245 / 255, 164 / 255))
        case "Yellow":
            return Color(name, rgb_value=(255 / 255, 255 / 255, 179 / 255))
        case "Purple":
            return Color(name, rgb_value=(190 / 255, 186 / 255, 218 / 255))
        case "Red":
            return Color(name, rgb_value=(251 / 255, 128 / 255, 114 / 255))
        case "Blue":
            return Color(name, rgb_value=(128 / 255, 177 / 255, 211 / 255))
        case "Orange":
            return Color(name, rgb_value=(250 / 255, 176 / 255, 124 / 255))
        case "Pink":
            return Color(name, rgb_value=(228 / 255, 170 / 255, 208 / 255))
        case "Cyan":
            return Color(name, rgb_value=(158 / 255, 207 / 255, 241 / 255))
        case "Gray":
            return Color(name, rgb_value=(215 / 255, 215 / 255, 215 / 255))
        case "GreenDarker":
            return Color(name, rgb_value=(116 / 255, 184 / 255, 124 / 255))
        case "YellowDarker":
            return Color(name, rgb_value=(217 / 255, 222 / 255, 122 / 255))
        case "PurpleDarker":
            return Color(name, rgb_value=(151 / 255, 132 / 255, 173 / 255))
        case "RedDarker":
            return Color(name, rgb_value=(209 / 255, 106 / 255, 94 / 255))
        case "BlueDarker":
            return Color(name, rgb_value=(100 / 255, 137 / 255, 163 / 255))
        case "OrangeDarker":
            return Color(name, rgb_value=(207 / 255, 146 / 255, 103 / 255))
        case "PinkDarker":
            return Color(name, rgb_value=(173 / 255, 130 / 255, 158 / 255))
        case "CyanDarker":
            return Color(name, rgb_value=(130 / 255, 167 / 255, 193 / 255))
        case "GrayDarker":
            return Color(name, rgb_value=(200 / 255, 200 / 255, 200 / 255))
        case "AltDeepRed":
            return Color(name, rgb_value=(215 / 255, 48 / 255, 39 / 255))
        case "AltRed":
            return Color(name, rgb_value=(252 / 255, 141 / 255, 89 / 255))
        case "AltPaleRed":
            return Color(name, rgb_value=(253 / 255, 219 / 255, 199 / 255))
        case "AltPaleYellow":
            return Color(name, rgb_value=(254 / 255, 224 / 255, 144 / 255))
        case "AltDeepBlue":
            return Color(name, rgb_value=(69 / 255, 117 / 255, 180 / 255))
        case "AltBlue":
            return Color(name, rgb_value=(145 / 255, 191 / 255, 219 / 255))
        case "AltPaleBlue":
            return Color(name, rgb_value=(224 / 255, 243 / 255, 248 / 255))
        case "AltDeepPurple":
            return Color(name, rgb_value=(118 / 255, 42 / 255, 131 / 255))
        case "AltPurple":
            return Color(name, rgb_value=(175 / 255, 141 / 255, 195 / 255))
        case "AltPalePurple":
            return Color(name, rgb_value=(231 / 255, 212 / 255, 232 / 255))
        case "AltDeepGreen":
            return Color(name, rgb_value=(27 / 255, 120 / 255, 55 / 255))
        case "AltGreen":
            return Color(name, rgb_value=(127 / 255, 191 / 255, 123 / 255))
        case "AltPaleGreen":
            return Color(name, rgb_value=(217 / 255, 240 / 255, 211 / 255))
        case "AltDeepRedDarker":
            return Color(name, rgb_value=(178 / 255, 24 / 255, 43 / 255))
        case "AltRedDarker":
            return Color(name, rgb_value=(239 / 255, 138 / 255, 98 / 255))
        case "AltPaleYellowDarker":
            return Color(name, rgb_value=(204 / 255, 183 / 255, 120 / 255))
        case "AltDeepBlueDarker":
            return Color(name, rgb_value=(33 / 255, 102 / 255, 172 / 255))
        case "AltBlueDarker":
            return Color(name, rgb_value=(103 / 255, 169 / 255, 207 / 255))
        case "AltPaleBlueDarker":
            return Color(name, rgb_value=(209 / 255, 229 / 255, 240 / 255))
        case "AltDeepGreenDarker":
            return Color(name, rgb_value=(1 / 255, 102 / 255, 94 / 255))
        case "AltGreenDarker":
            return Color(name, rgb_value=(90 / 255, 180 / 255, 172 / 255))
        case "AltPaleGreenDarker":
            return Color(name, rgb_value=(199 / 255, 234 / 255, 229 / 255))
        case "AtlasBlue":
            return Color(name, rgb_value=(87 / 255, 144 / 255, 252 / 255))
        case "AtlasYellow":
            return Color(name, rgb_value=(255 / 255, 169 / 255, 14 / 255))
        case "AtlasRed":
            return Color(name, rgb_value=(228 / 255, 37 / 255, 54 / 255))
        case "AtlasGray":
            return Color(name, rgb_value=(148 / 255, 164 / 255, 162 / 255))
        case "AtlasPurple":
            return Color(name, rgb_value=(150 / 255, 74 / 255, 139 / 255))
        case "AtlasBrown":
            return Color(name, rgb_value=(169 / 255, 107 / 255, 89 / 255))
        case "AtlasOrange":
            return Color(name, rgb_value=(255 / 255, 94 / 255, 2 / 255))
        case "AtlasGreen":
            return Color(name, rgb_value=(173 / 255, 173 / 255, 125 / 255))
        case "AtlasCyan":
            return Color(name, rgb_value=(146 / 255, 218 / 255, 221 / 255))
        case "AtlasBlueDarker":
            return Color(name, rgb_value=(63 / 255, 144 / 255, 218 / 255))
        case "AtlasYellowDarker":
            return Color(name, rgb_value=(248 / 255, 156 / 255, 32 / 255))
        case "AtlasRedDarker":
            return Color(name, rgb_value=(189 / 255, 31 / 255, 1 / 255))
        case "AtlasGrayDarker":
            return Color(name, rgb_value=(113 / 255, 117 / 255, 129 / 255))
        case "AtlasPurpleDarker":
            return Color(name, rgb_value=(131 / 255, 45 / 255, 182 / 255))
        case "AtlasBrownDarker":
            return Color(name, rgb_value=(169 / 255, 107 / 255, 89 / 255))
        case "AtlasOrangeDarker":
            return Color(name, rgb_value=(231 / 255, 99 / 255, 0 / 255))
        case "AtlasGreenDarker":
            return Color(name, rgb_value=(185 / 255, 172 / 255, 112 / 255))
        case "AtlasCyanDarker":
            return Color(name, rgb_value=(134 / 255, 200 / 255, 221 / 255))
        case "MicroCvdPurple1":
            return Color(name, rgb_value=(239 / 255, 182 / 255, 214 / 255))
        case "MicroCvdPurple2":
            return Color(name, rgb_value=(231 / 255, 148 / 255, 193 / 255))
        case "MicroCvdPurple3":
            return Color(name, rgb_value=(204 / 255, 121 / 255, 167 / 255))
        case "MicroCvdPurple4":
            return Color(name, rgb_value=(161 / 255, 82 / 255, 127 / 255))
        case "MicroCvdPurple5":
            return Color(name, rgb_value=(125 / 255, 53 / 255, 96 / 255))
        case "MicroCvdTurquoise1":
            return Color(name, rgb_value=(163 / 255, 228 / 255, 215 / 255))
        case "MicroCvdTurquoise2":
            return Color(name, rgb_value=(72 / 255, 201 / 255, 176 / 255))
        case "MicroCvdTurquoise3":
            return Color(name, rgb_value=(67 / 255, 186 / 255, 143 / 255))
        case "MicroCvdTurquoise4":
            return Color(name, rgb_value=(0 / 255, 158 / 255, 115 / 255))
        case "MicroCvdTurquoise5":
            return Color(name, rgb_value=(20 / 255, 143 / 255, 119 / 255))
        case "MicroCvdBlue1":
            return Color(name, rgb_value=(231 / 255, 244 / 255, 255 / 255))
        case "MicroCvdBlue2":
            return Color(name, rgb_value=(188 / 255, 225 / 255, 255 / 255))
        case "MicroCvdBlue3":
            return Color(name, rgb_value=(125 / 255, 204 / 255, 255 / 255))
        case "MicroCvdBlue4":
            return Color(name, rgb_value=(86 / 255, 180 / 255, 233 / 255))
        case "MicroCvdBlue5":
            return Color(name, rgb_value=(9 / 255, 139 / 255, 217 / 255))
        case "MicroCvdOrange1":
            return Color(name, rgb_value=(255 / 255, 213 / 255, 175 / 255))
        case "MicroCvdOrange2":
            return Color(name, rgb_value=(252 / 255, 176 / 255, 118 / 255))
        case "MicroCvdOrange3":
            return Color(name, rgb_value=(240 / 255, 145 / 255, 99 / 255))
        case "MicroCvdOrange4":
            return Color(name, rgb_value=(193 / 255, 119 / 255, 84 / 255))
        case "MicroCvdOrange5":
            return Color(name, rgb_value=(157 / 255, 101 / 255, 76 / 255))
        case "MicroCvdGreen1":
            return Color(name, rgb_value=(221 / 255, 255 / 255, 160 / 255))
        case "MicroCvdGreen2":
            return Color(name, rgb_value=(189 / 255, 236 / 255, 111 / 255))
        case "MicroCvdGreen3":
            return Color(name, rgb_value=(151 / 255, 206 / 255, 47 / 255))
        case _:
            raise ValueError(f"Unknown color: {name}")
