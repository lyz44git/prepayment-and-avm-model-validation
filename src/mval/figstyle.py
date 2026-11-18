"""Matplotlib house style (reference data-viz palette, light surface, thin marks, recessive axes)."""
import matplotlib as mpl, matplotlib.pyplot as plt
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})
def note(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.8, color=INK2, wrap=True)
