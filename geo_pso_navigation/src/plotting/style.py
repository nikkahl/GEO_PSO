"""
plotting/style.py
Shared publication style for all figures: serif fonts, visible grid,
300 dpi, English labels. Import `apply_style()` once at module load.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


apply_style()
