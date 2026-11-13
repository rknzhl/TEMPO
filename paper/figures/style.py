"""Shared article colors and typography."""
from pathlib import Path

WIDTH_MM = 170
COLORS = {
    "TEMPO": "#2F6BBD", "POD": "#777777", "GMM": "#D64F4F",
    "FNO": "#2E9D78", "CNO": "#8B6BB1", "DeepONet": "#D99532",
    "NeuralPOD": "#75A0D8",
}
REGIMES = ["#2F6BBD", "#D64F4F", "#2E9D78", "#8B6BB1", "#D99532"]
SIGNED = "RdBu_r"
POSITIVE = "viridis"
ERROR = "magma"


def apply():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": .5, "lines.linewidth": 1.2,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.labelpad": 3, "axes.titlepad": 5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=list(COLORS.values())),
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "tempo-figures",
    })


def size(height_mm, width_mm=WIDTH_MM):
    return width_mm / 25.4, height_mm / 25.4


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300,
                    facecolor="white")
