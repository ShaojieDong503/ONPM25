#!/usr/bin/env python3
"""
Region x year block heatmaps: our model vs the CanOSSEM final product.

Reads canossem_block_metrics_<family>.csv (both sides computed on the identical
169,648 matched cell-days) and writes a three-panel figure: our model, CanOSSEM,
and the block-by-block difference.

  python make_canossem_block_heatmap.py                  # R2, the default
  python make_canossem_block_heatmap.py --metric rmse
  python make_canossem_block_heatmap.py --metric both

Color follows the job, not habit
--------------------------------
Panels 1-2 encode MAGNITUDE, so they use one hue, light->dark, on a shared scale --
not a diverging map. A diverging ramp on R2 in [0,1] implies a meaningful midpoint
at 0.5, and there isn't one; it makes 0.5 look like a neutral "zero" and splits a
single continuum into two apparent classes. (The earlier thesis figure used
coolwarm here.)

Panel 3 encodes POLARITY -- who wins, and by how much -- so it uses two hues with a
neutral gray midpoint, symmetric about zero, so equal margins read equally in both
directions.

For predictive R2, zero is a real anchor: below it the block mean beats the model.
The sequential ramp bottoms out at 0 and anything below is drawn in the
out-of-range tint, with the value printed in the cell regardless, so the sign never
depends on color alone.

Every cell is annotated, which doubles as the table view: color is never the sole
carrier of a number. Annotation ink flips to white on dark cells by measured
luminance rather than a fixed threshold guess.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "outputs" / "canossem_benchmark" / "canossem_block_metrics_lgbm.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "canossem_benchmark"

REGION_ORDER = ["Central", "East", "North", "Toronto", "West_NH", "West_SW"]
REGION_LABEL = {"West_NH": "West-NH", "West_SW": "West-SW"}
YEAR_ORDER = list(range(2012, 2024))

# --- palette -----------------------------------------------------------------
# Sequential: the reference palette's blue ramp, steps 100 -> 700.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Diverging: blue <-> red poles with a neutral gray midpoint. Gray, not a hue --
# the midpoint has to read as "nothing".
RED = ["#fbd5d5", "#f2a8a8", "#e87b7b", "#e34948", "#c53434", "#9e2727", "#7a1d1d"]
GRAY_MID = "#f0efec"
UNDER_TINT = "#f2d5d5"        # R2 < 0: off the bottom of the scale

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
SEQ.set_under(UNDER_TINT)
SEQ_R = LinearSegmentedColormap.from_list("seq_blue_r", BLUE[::-1])
SEQ_R.set_over(UNDER_TINT)
# Equal step count per arm, and the two poles matched in lightness -- otherwise a
# margin of +0.3 and one of -0.3 do not read as equal, which is the whole point of
# a diverging scale.
DIV_RED = ["#c53434", "#e34948", "#e87b7b", "#f2a8a8", "#fbd5d5"]
DIV_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]
DIV = LinearSegmentedColormap.from_list("div_red_gray_blue",
                                        DIV_RED + [GRAY_MID] + DIV_BLUE)

METRICS = {
    "r2": {
        "ours": "ours_final_r2_predictive", "can": "canossem_r2_predictive",
        "label": "Predictive R²", "fmt": "{:.2f}", "cmap": SEQ,
        "norm": Normalize(vmin=0.0, vmax=1.0), "better": "higher",
        "stem": "canossem_block_r2_heatmap",
    },
    "rmse": {
        "ours": "ours_final_rmse", "can": "canossem_rmse",
        "label": "RMSE (µg/m³)", "fmt": "{:.2f}", "cmap": SEQ_R,
        "norm": None, "better": "lower",
        "stem": "canossem_block_rmse_heatmap",
    },
}


def luminance(rgba) -> float:
    """WCAG relative luminance."""
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgba[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def pick_ink(rgba) -> str:
    """Whichever of white / primary ink has the higher contrast ratio on this cell.

    Computed, not guessed at with a lightness threshold -- the crossover sits near
    L=0.19, far darker than eyeballing suggests, so a guessed cutoff leaves white
    text on mid-tone fills.
    """
    bg = luminance(rgba)
    ink = luminance(mpl.colors.to_rgb(INK))
    c_white = 1.05 / (bg + 0.05)
    c_ink = (max(bg, ink) + 0.05) / (min(bg, ink) + 0.05)
    return "#ffffff" if c_white > c_ink else INK


def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    p = df.pivot_table(index="region", columns="year", values=col, aggfunc="mean")
    return p.reindex(index=REGION_ORDER, columns=YEAR_ORDER)


def draw(ax, grid: pd.DataFrame, cmap, norm, title: str, fmt: str, *,
         note: str = "") -> mpl.image.AxesImage:
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(YEAR_ORDER)))
    ax.set_xticklabels(YEAR_ORDER, fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(len(REGION_ORDER)))
    ax.set_yticklabels([REGION_LABEL.get(r, r) for r in grid.index], fontsize=9,
                       color=INK_SECONDARY)
    # Title sits above the note; the note above the grid. Pad has to clear both.
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=26)
    if note:
        ax.text(0, 1.012, note, transform=ax.transAxes, fontsize=8.5,
                color=INK_MUTED, va="bottom", ha="left")
    # 2px surface gap between cells: minor-tick grid drawn in the surface color.
    ax.set_xticks(np.arange(-0.5, len(YEAR_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REGION_ORDER), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    vals = grid.to_numpy(dtype=float)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=8.2, color=pick_ink(cmap(norm(v))))
    return im


def build(df: pd.DataFrame, spec: dict, family: str, out_dir: Path, dpi: int) -> Path:
    ours, can = pivot(df, spec["ours"]), pivot(df, spec["can"])
    diff = ours - can

    norm, clipped = spec["norm"], 0
    if norm is None:
        # RMSE: shared scale across both panels, but capped at a robust upper bound.
        # Scaling to the raw max lets one 2023 smoke block (11.4 ug/m3, 4x the median)
        # compress the other 140 cells into a single indistinguishable tone. The cap
        # is drawn as over-range and the value is printed regardless, so nothing is
        # hidden -- only the ramp is spent where the data actually lives.
        both = np.concatenate([ours.to_numpy().ravel(), can.to_numpy().ravel()])
        both = both[np.isfinite(both)]
        lo = float(np.floor(both.min() * 2) / 2)
        hi = float(np.ceil(np.percentile(both, 98) * 2) / 2)
        clipped = int((both > hi).sum())
        norm = Normalize(vmin=lo, vmax=hi)

    # Diverging panel: symmetric about zero so equal margins read equally.
    if spec["better"] == "higher":
        dgrid, dlabel = diff, f"{spec['label']}:  ours − CanOSSEM"
        wins = int((diff.to_numpy() > 0).sum())
    else:
        dgrid, dlabel = -diff, f"{spec['label']}:  CanOSSEM − ours"
        wins = int((diff.to_numpy() < 0).sum())
    lim = float(np.nanmax(np.abs(dgrid.to_numpy())))
    dnorm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    n_blocks = int(df.shape[0])
    n_rows = int(df["canossem_n"].sum())
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 13.0), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.34, right=0.88)

    below = (lambda g: int((g.to_numpy() < 0).sum())) if spec["better"] == "higher" else (lambda g: 0)
    neg_ours, neg_can = below(ours), below(can)
    im1 = draw(axes[0], ours, spec["cmap"], norm,
               "Our model — Ontario-only LightGBM, held out", spec["fmt"],
               note="one model per outer group; each block scored by a model that never saw it")
    im2 = draw(axes[1], can, spec["cmap"], norm,
               "CanOSSEM — final-product estimate", spec["fmt"],
               note="final product, not CanOSSEM out-of-fold — see caveat below")
    im3 = draw(axes[2], dgrid, DIV, dnorm,
               f"Difference — blue favours our model  ({wins} of {n_blocks} blocks)",
               "{:+.2f}", note=f"symmetric about zero, ±{lim:.2f}")

    over = "max" if clipped else None
    for ax, im, lab, nneg in ((axes[0], im1, spec["label"], neg_ours),
                              (axes[1], im2, spec["label"], neg_can),
                              (axes[2], im3, dlabel, 0)):
        cax = ax.inset_axes([1.015, 0.0, 0.014, 1.0])
        ext = "min" if nneg else (over if im is not im3 else None)
        cb = fig.colorbar(im, cax=cax, extend=ext or "neither")
        cb.set_label(lab, fontsize=9, color=INK_SECONDARY)
        cb.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_SECONDARY, length=2)
        cb.outline.set_visible(False)
    axes[2].set_xlabel("Year", fontsize=10, color=INK_SECONDARY, labelpad=7)

    sub = (f"72 region-year blocks · both sides scored on the identical {n_rows:,} matched "
           f"cell-days · seed 2026")
    fig.suptitle(f"Block performance, our model vs CanOSSEM — {spec['label']}",
                 fontsize=14.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.977)
    fig.text(0.055, 0.952, sub, fontsize=9.5, color=INK_SECONDARY, ha="left")

    caveat = ("Not like-for-like: our values are held out (out-of-fold); CanOSSEM values are "
              "final-product estimates, not CanOSSEM out-of-fold predictions. "
              "A reference benchmark, not an external validation of either product.")
    if neg_ours or neg_can:
        caveat = ("Pale-red cells have predictive R² < 0 — the block mean beats the model "
                  "there; the printed value carries the sign. " + caveat)
    if clipped:
        caveat = (f"Colour scale is capped at {norm.vmax:g} µg/m³ (98th percentile); "
                  f"{clipped} cell(s) above it are drawn pale-red with the true value "
                  f"printed. " + caveat)
    # Wrap by hand: fig.text(wrap=True) measures against the pre-tight-bbox canvas
    # and spills past the right edge once bbox_inches='tight' crops.
    fig.text(0.055, 0.020, "\n".join(textwrap.wrap(caveat, width=148)),
             fontsize=8.2, color=INK_MUTED, ha="left", va="top")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['stem']}_{family}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--metric", choices=["r2", "rmse", "both"], default="r2")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    family = args.csv.stem.split("_")[-1]
    missing = sorted({c for m in METRICS.values() for c in (m["ours"], m["can"])} - set(df.columns))
    if missing:
        raise SystemExit(f"[fatal] {args.csv} lacks {missing}. Rebuild it with "
                         f"build_canossem_block_metrics.py in oof mode (it needs both sides).")
    if len(df) != 72:
        raise SystemExit(f"[fatal] expected 72 region-year blocks, found {len(df)}")

    print(f"[input] {args.csv}  ({len(df)} blocks, {int(df['canossem_n'].sum()):,} matched rows)")
    for key in (["r2", "rmse"] if args.metric == "both" else [args.metric]):
        spec = METRICS[key]
        d = df[spec["ours"]] - df[spec["can"]]
        wins = int((d > 0).sum()) if spec["better"] == "higher" else int((d < 0).sum())
        print(f"[{key}] ours better in {wins}/72 blocks   "
              f"median margin {abs(d).median():.3f}   max {abs(d).max():.3f}")
        print(f"        -> {build(df, spec, family, args.out_dir, args.dpi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
