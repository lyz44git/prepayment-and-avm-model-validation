"""Shared style and chart panels for every figure in the repository.

A navy, blue and grey palette with orange kept for the difference or threshold being explained, thin grids, direct
labels instead of legends where possible, and rounded marks. Every number is read from the tables in reports/; nothing
is typed in by hand.
"""
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
P1, P2, REM, P3 = "reports/p1/", "reports/p2/", "reports/p1/remediation/", "reports/p3/"

THEME = os.environ.get("FIG_THEME", "light")      # "dark" draws the GitHub dark-mode variants (figures/*-dark.svg)
if THEME == "dark":
    BG, AXEDGE, GOOD, ONPILL = "#0d1117", "#3a424d", "#1f3a2a", "#0d1117"
    INK, INK2, MUTED, GRID = "#e6edf3", "#c9d1d9", "#8b949e", "#262c36"
    NAVY, BLUE, LIGHT, PALE, GREY, ORANGE, OSOFT = "#a5c8f0", "#5a9be0", "#3d6b99", "#1b2430", "#6e7681", "#f0883e", "#4a2e1a"
else:
    BG, AXEDGE, GOOD, ONPILL = "#ffffff", "#b9bec5", "#d9eadf", "#ffffff"
    INK, INK2, MUTED, GRID = "#1d232b", "#4b5563", "#7b8490", "#e9e6df"
    NAVY, BLUE, LIGHT, PALE, GREY, ORANGE, OSOFT = "#1f3b5c", "#3f78b0", "#a9c4de", "#dfe6ee", "#9aa3ad", "#e07b39", "#f6d9c4"
HALO = [patheffects.withStroke(linewidth=3, foreground=BG)]


def style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXEDGE, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 4, "ytick.major.pad": 4, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path", "lines.solid_capstyle": "round", "lines.dash_capstyle": "round"})


def load(rel):
    return json.loads((ROOT / rel).read_text())


def rows(rel):
    with open(ROOT / rel) as f:
        return list(csv.DictReader(f))


def fmt(v, d=0):
    return f"{v:,.{d}f}"


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", color=INK, pad=10)


def clean(ax, x=True, y=False):
    """Hide the axis lines that the chart does not need."""
    ax.spines["bottom"].set_visible(x)
    ax.spines["left"].set_visible(y)


def dot(ax, x, y, color=NAVY, s=46, z=4, marker="o"):  # noqa: D103
    ax.scatter([x], [y], s=s, color=color, edgecolor=BG, linewidth=1.2, zorder=z, marker=marker)


def pill(ax, x0, x1, y, color, lw=9, z=2, alpha=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=z, alpha=alpha)


def hbars(ax, labels, values, colors, fmtv, vmax=None, lw=12, note=None, label_w=None):
    """Rounded horizontal bars with the value written at the end of each bar."""
    n = len(values); vmax = vmax or max(values) * 1.22
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        y = n - 1 - i
        ax.barh(y, vmax, height=0.56, color=PALE, alpha=0.55, zorder=1, lw=0)
        ax.barh(y, v, height=0.56, color=c, zorder=2, lw=0)
        t = ax.annotate(fmtv(v), (v, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, fontweight="bold", color=INK)
        if note and note[i]:
            ax.annotate(note[i], (v, y), xytext=(5, -10), textcoords="offset points", va="center", fontsize=7.5, color=ORANGE, fontweight="bold")
    ax.set_yticks(range(n), labels[::-1]); ax.set_ylim(-0.7, n - 0.3); ax.set_xlim(0, vmax * 1.12)
    ax.grid(False); clean(ax, x=False); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8, labelcolor=INK)


def dotrows(ax, labels, est, lo, hi, colors, fmtv, xlim, band=None, ref=None, ref_label=None, bold=None, pad=0.02):
    """Point estimates with intervals, one row each, value written to the right."""
    n = len(labels)
    if band:
        ax.axvspan(band[0], band[1], color=PALE, alpha=0.6, lw=0, zorder=0)
    for i in range(n):
        y = n - 1 - i
        ax.plot([lo[i], hi[i]], [y, y], color=colors[i], lw=2.2, zorder=3, solid_capstyle="round")
        dot(ax, est[i], y, colors[i], s=52)
        ax.text(hi[i] + (xlim[1] - xlim[0]) * pad, y, fmtv(est[i]), va="center", fontsize=8.5, fontweight="bold", color=INK)
    if ref is not None:
        ax.axvline(ref, color=ORANGE, lw=1.1, ls=(0, (4, 3)), zorder=1)
        if ref_label:
            ax.text(ref, n - 0.45, ref_label, ha="center", va="bottom", fontsize=7.5, color=ORANGE, fontweight="bold")
    ax.set_yticks(range(n), labels[::-1]); ax.set_ylim(-0.6, n - 0.2); ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False); clean(ax, x=False)
    for t, lab in zip(ax.get_yticklabels(), labels[::-1]):
        t.set_color(INK); t.set_fontsize(8)
        if bold and lab in bold:
            t.set_fontweight("bold")


def rbox(ax, x, y, w, h, color, alpha=1.0, r=0.08, ec="none", lw=0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]; b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(u + (v - u) * t):02x}" for u, v in zip(a, b))


def nice_log(ax, axis, ticks, labels=None):
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, labels or [fmt(t, 1 if t < 1 else 0) for t in ticks])
    ax.minorticks_off()


def align_panel_titles(fig):
    """Put the panel titles ("A   ...", "B   ...") of each row of panels on one baseline, the highest in that row."""
    import re
    fig.canvas.draw()
    items = []
    for ax in fig.axes:
        t = ax._left_title
        if t.get_visible() and re.match(r"^[A-Z] {2,}", t.get_text()):
            x, y = t.get_transform().transform(t.get_position())
            items.append((ax.get_position(), t, x, y))
    rows = []
    for it in sorted(items, key=lambda i: -i[0].y1):
        for r in rows:
            if it[0].y0 < r[0][0].y1 and it[0].y1 > r[0][0].y0:
                r.append(it); break
        else:
            rows.append([it])
    inv = fig.transFigure.inverted()
    for r in rows:
        if len(r) < 2:
            continue
        top = max(i[3] for i in r)
        for _, t, x, y in r:
            fx, fy = inv.transform((x, top))
            fig.text(fx, fy, t.get_text(), fontproperties=t.get_fontproperties(), color=t.get_color(), ha="left", va="baseline")
            t.set_visible(False)


def save(fig, rel):
    if fig.get_layout_engine() is not None:      # fix the constrained layout before the titles are aligned
        fig.canvas.draw(); fig.set_layout_engine("none")
    align_panel_titles(fig)
    path = ROOT / rel
    if THEME == "dark":                           # dark-mode variant for <picture> in the README
        fig.savefig(path.with_name(path.stem + "-dark.svg"), bbox_inches="tight", pad_inches=0.18)
    else:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        pad_to_content(path)
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("wrote", rel, THEME)


def pad_to_content(path, pad=54):
    """Crop the saved PNG to its visible content and add the same margin (0.18 in at 300 dpi) on every side."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, BG)).point(lambda v: 255 if v > 30 else 0).getbbox()
    if not box:
        return
    l, t, r, b = box
    out = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), BG)
    out.paste(im.crop(box), (pad, pad))
    out.save(path, dpi=(300, 300))


