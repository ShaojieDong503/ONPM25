#!/usr/bin/env python3
"""
TreeSHAP attribution for one thesis LightGBM fold.

Explains the Stage-1 model on its OWN HELD-OUT rows, so the attributions describe
out-of-fold behaviour rather than memorised training rows.

    python shap_fold_analysis.py --fold 4                  # compute + figures
    python shap_fold_analysis.py --fold 4 --stage compute   # values only (~30 min)
    python shap_fold_analysis.py --fold 4 --stage figures   # re-render from cache

SHAP values come from LightGBM's own `predict(pred_contrib=True)`, which is exact
TreeSHAP and returns values identical to shap.TreeExplainer (verified: max abs
difference 0.0) at the same speed, without depending on the shap package to compute.
The shap package is used only for the beeswarm rendering.

Outputs, under <out-root>/GROUP_0N/shap/:
    shap_values.npy                 (rows x 651) float32 raw attributions
    shap_rows.parquet               row metadata aligned to shap_values
    shap_feature_importance.csv     per-feature mean|SHAP|, with family labels
    shap_family_summary.csv         aggregated by resolution / temporal / source
    fig1_top_features.png           top-25 global importance
    fig2_repaired_share.png         438 repaired vs 213 directly-resolved
    fig3_source_families.png        contribution by data source
    fig4_beeswarm.png               per-row attribution spread, top 20
    fig5_dependence.png             dependence for the top predictors
    shap_summary.json               headline numbers
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- palette
# Values from the data-viz reference palette. Roles, not raw hex, at point of use.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
NEUTRAL = "#f0efec"
DEEMPH = "#c3c2b7"


# ---------------------------------------------------------------- feature taxonomy

ROLL_SUFFIXES = ("_roll3_min", "_roll3_mean", "_roll3_max",
                 "_roll7_min", "_roll7_mean", "_roll7_max")


def temporal_stat(name: str) -> str:
    if name.endswith("_lag1"):
        return "lag1"
    for suf in ROLL_SUFFIXES:
        if name.endswith(suf):
            return "roll3" if "_roll3_" in suf else "roll7"
    return "same-day"


def source_family(name: str) -> str:
    n = name.lower()
    if n.startswith("src_viirs"):
        return "VIIRS active fire"
    if n.startswith("src_hms"):
        return "HMS smoke plume"
    if n.startswith("burned_"):
        return "Burned area"
    if n.startswith("src_merra_aer"):
        return "MERRA-2 aerosol"
    if n.startswith("src_merra_flx"):
        return "MERRA-2 surface flux"
    if n.startswith("src_merra_slv"):
        return "MERRA-2 winds/levels"
    if "aod" in n:
        return "Satellite AOD"
    if any(n.startswith(p) for p in ("dayofyear", "doy", "month", "season", "dow")):
        return "Calendar"
    if any(k in n for k in ("road", "lc_", "landcover", "land_cover", "imperv", "urban",
                            "forest", "crop", "water_frac", "elev")):
        return "Land cover / roads"
    return "Local meteorology"


def build_taxonomy(features: list[str], stored_names: set[str]) -> pd.DataFrame:
    """One row per canonical feature with its family labels.

    `resolution` records whether the canonical name is present verbatim in the shard
    parquet ("direct") or only under the stored `_lag1_` rolling spelling
    ("repaired") -- the 213/438 split the verification work established.
    """
    rows = []
    for f in features:
        stored = f
        if "_roll3_" in f or "_roll7_" in f:
            stored = f.replace("_roll3_", "_lag1_roll3_").replace("_roll7_", "_lag1_roll7_")
        rows.append({
            "feature": f,
            "stored_name": stored,
            "resolution": "direct" if f in stored_names else "repaired",
            "temporal_stat": temporal_stat(f),
            "source_family": source_family(f),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- compute

def compute_shap(fold: int, shard_root: Path, out_root: Path, limit: int | None) -> Path:
    import run_lgbm_thesis_fold as F
    import pyarrow.parquet as pq

    lg = logging.getLogger("shap_load")
    lg.addHandler(logging.StreamHandler(sys.stdout))
    lg.setLevel(logging.INFO)

    gdir = out_root / f"GROUP_{fold:02d}"
    sdir = gdir / "shap"
    sdir.mkdir(parents=True, exist_ok=True)

    bundle = pickle.load((gdir / "stage1_model.pkl").open("rb"))
    model = bundle["model"]
    features = list(bundle["feature_cols"])
    heldout = list(bundle["holdout_blocks"])

    print(f"[shap] fold {fold}: {len(features)} features, {len(heldout)} held-out blocks")
    print(f"[shap] blocks: {', '.join(heldout)}")

    meta, X, _ = F.load_blocks(shard_root, heldout, features, lg)
    if limit:
        meta, X = meta.iloc[:limit].copy(), X[:limit]
    print(f"[shap] explaining {X.shape[0]:,} rows x {X.shape[1]} features")

    t0 = time.perf_counter()
    contrib = model.predict(X, pred_contrib=True)
    print(f"[shap] TreeSHAP done in {time.perf_counter() - t0:.0f}s")

    sv = np.asarray(contrib[:, :-1], dtype=np.float32)
    base_value = float(contrib[0, -1])

    # Additivity: base + sum(shap) must reconstruct the model's own prediction.
    pred = model.predict(X)
    recon = base_value + sv.sum(axis=1)
    max_err = float(np.abs(recon - pred).max())
    print(f"[shap] additivity check: max |base + sum(shap) - pred| = {max_err:.3e}")
    if max_err > 1e-3:
        raise RuntimeError(f"SHAP additivity violated (max error {max_err:.3e}); "
                           f"the attributions do not reconstruct the model")

    np.save(sdir / "shap_values.npy", sv)
    keep = [c for c in ("grid_cell_id", "CanOSSEM_RASTER_CELL", "date", "year",
                        "fold_region", "pm25", "_source_block") if c in meta.columns]
    rows = meta[keep].copy()
    rows["pred_stage1"] = pred
    rows["shap_sum"] = sv.sum(axis=1)
    rows.to_parquet(sdir / "shap_rows.parquet", index=False)

    stored_names = set(pq.read_schema(shard_root / "pair_blocks" / heldout[0] /
                                      "frame.parquet").names)
    tax = build_taxonomy(features, stored_names)
    tax["mean_abs_shap"] = np.abs(sv).mean(axis=0)
    tax["mean_shap"] = sv.mean(axis=0)
    tax["max_abs_shap"] = np.abs(sv).max(axis=0)
    tax["share_pct"] = 100.0 * tax["mean_abs_shap"] / tax["mean_abs_shap"].sum()
    tax = tax.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    tax["rank"] = np.arange(1, len(tax) + 1)
    tax.to_csv(sdir / "shap_feature_importance.csv", index=False)

    summary = {
        "fold": fold, "rows": int(X.shape[0]), "features": int(X.shape[1]),
        "base_value": base_value, "additivity_max_error": max_err,
        "heldout_blocks": heldout,
        "obs_mean": float(pd.to_numeric(meta["pm25"]).mean()),
        "pred_mean": float(pred.mean()),
    }
    (sdir / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[shap] wrote {sdir}")
    return sdir


# ---------------------------------------------------------------- figures

def _style(ax, *, xlabel: str = "", ylabel: str = "", title: str = "", subtitle: str = ""):
    """Recessive chrome: hairline grid, no top/right spines, muted axis ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=16)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9,
                va="bottom", ha="left")


def make_figures(fold: int, out_root: Path, top_n: int = 25) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "axes.grid": False,
    })

    sdir = out_root / f"GROUP_{fold:02d}" / "shap"
    tax = pd.read_csv(sdir / "shap_feature_importance.csv")
    sv = np.load(sdir / "shap_values.npy")
    summary = json.loads((sdir / "shap_summary.json").read_text(encoding="utf-8"))
    total = tax["mean_abs_shap"].sum()

    # ---------------------------------------------------------- fig 1: top features
    # Job: compare magnitude low->high -> horizontal bar, one sequential hue.
    top = tax.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * top_n + 1.8))
    norm = top["mean_abs_shap"] / top["mean_abs_shap"].max()
    colors = [SEQ[min(len(SEQ) - 1, int(2 + v * (len(SEQ) - 3)))] for v in norm]
    ax.barh(range(len(top)), top["mean_abs_shap"], color=colors, height=0.72)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f.replace("src_merra_", "").replace("_wmean", "")
                        for f in top["feature"]], fontsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for i, (v, s) in enumerate(zip(top["mean_abs_shap"], top["share_pct"])):
        ax.text(v + top["mean_abs_shap"].max() * 0.012, i, f"{v:.3f}",
                va="center", fontsize=7.5, color=INK_2)
    _style(ax, xlabel="mean |SHAP|  (µg/m³ contribution to the prediction)",
           title=f"What drives Stage-1 PM2.5 — fold {fold}, top {top_n} of 651 predictors",
           subtitle=f"TreeSHAP over {summary['rows']:,} held-out grid-cell-days · "
                    f"LightGBM · base value {summary['base_value']:.2f} µg/m³")
    fig.tight_layout()
    fig.savefig(sdir / "fig1_top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- fig 2: repaired share
    # Job: part-to-whole across two groups -> stacked bar, 2 categorical series,
    # direct-labelled (so identity never rests on color alone).
    grp = tax.groupby("resolution").agg(
        n=("feature", "size"), shap=("mean_abs_shap", "sum")).reindex(["direct", "repaired"])
    grp["share"] = 100 * grp["shap"] / grp["shap"].sum()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), height_ratios=[1, 1])
    # Title block lives in FIGURE coordinates with reserved headroom, so it cannot
    # collide with the first axes' own title/subtitle.
    fig.subplots_adjust(top=0.72, bottom=0.18, hspace=0.75)
    fig.text(0.02, 0.955, f"Do the 438 repaired rolling features actually matter? — fold {fold}",
             color=INK, fontsize=13, fontweight="bold", va="top", ha="left")
    fig.text(0.02, 0.885,
             "Canonical names the shard stores under a different spelling.",
             color=INK_2, fontsize=9.5, va="top", ha="left")
    fig.text(0.02, 0.845,
             "Before the repair they reached the model as constant 0.0.",
             color=INK_2, fontsize=9.5, va="top", ha="left")
    labels = {"direct": f"Directly resolved  ({int(grp.loc['direct','n'])} features)",
              "repaired": f"Repaired rolling names  ({int(grp.loc['repaired','n'])} features)"}
    for ax, col, unit, ttl in (
        (axes[0], "n", "predictors", "Share of the feature set"),
        (axes[1], "shap", "mean |SHAP|", "Share of total attribution"),
    ):
        vals = grp[col].to_numpy(dtype=float)
        pct = 100 * vals / vals.sum()
        left = 0.0
        for v, p, key, color in zip(vals, pct, grp.index, (BLUE, ORANGE)):
            ax.barh([0], [p], left=left, color=color, height=0.5,
                    edgecolor=SURFACE, linewidth=2)
            if p > 6:
                ax.text(left + p / 2, 0, f"{p:.1f}%", ha="center", va="center",
                        color="#ffffff", fontsize=11, fontweight="bold")
            left += p
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        _style(ax, title="", subtitle="")
        ax.text(0, 1.35, ttl, transform=ax.transAxes, color=INK, fontsize=10.5,
                fontweight="bold", va="bottom")
        ax.spines["left"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    axes[1].legend(handles, [labels["direct"], labels["repaired"]],
                   loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=2,
                   frameon=False, fontsize=9.5, labelcolor=INK_2)
    fig.savefig(sdir / "fig2_repaired_share.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- fig 3: source families
    fam = (tax.groupby("source_family")
           .agg(n=("feature", "size"), shap=("mean_abs_shap", "sum"))
           .sort_values("shap"))
    fam["share"] = 100 * fam["shap"] / fam["shap"].sum()

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(fam) + 2.0))
    norm = (fam["shap"] / fam["shap"].max()).to_numpy()
    colors = [SEQ[min(len(SEQ) - 1, int(2 + v * (len(SEQ) - 3)))] for v in norm]
    ax.barh(range(len(fam)), fam["shap"], color=colors, height=0.66)
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels([f"{i}  ({int(n)})" for i, n in zip(fam.index, fam["n"])], fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for i, (v, p) in enumerate(zip(fam["shap"], fam["share"])):
        ax.text(v + fam["shap"].max() * 0.012, i, f"{p:.1f}%", va="center",
                fontsize=8.5, color=INK_2)
    _style(ax, xlabel="summed mean |SHAP| across the family  (µg/m³)",
           title=f"Where the signal comes from — fold {fold}",
           subtitle="Predictor count in brackets; label shows the family's share of total attribution")
    fig.tight_layout()
    fig.savefig(sdir / "fig3_source_families.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- fig 4: beeswarm
    # Per-row spread. Feature VALUE is encoded on a diverging blue<->red ramp with a
    # neutral midpoint, which is the palette's diverging pair.
    import shap as shap_pkg
    rows = pd.read_parquet(sdir / "shap_rows.parquet")
    feats = list(tax.sort_values("rank")["feature"])
    order = [feats.index(f) for f in tax.head(20)["feature"]]

    import run_lgbm_thesis_fold as F
    lg = logging.getLogger("swarm"); lg.addHandler(logging.NullHandler()); lg.setLevel(logging.ERROR)
    bundle = pickle.load((out_root / f"GROUP_{fold:02d}" / "stage1_model.pkl").open("rb"))
    shard_root = Path(summary.get("shard_root", ROOT / "Data_zerofilled"))
    _, X, _ = F.load_blocks(shard_root, bundle["holdout_blocks"],
                            list(bundle["feature_cols"]), lg)
    X = X[: sv.shape[0]]

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    fig = plt.figure(figsize=(9.5, 7.5))
    shap_pkg.summary_plot(
        sv[:, order], X[:, order],
        feature_names=[f.replace("src_merra_", "").replace("_wmean", "")
                       for f in tax.head(20)["feature"]],
        cmap=cmap, show=False, plot_size=None, color_bar_label="Predictor value")
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK_2)
        lbl.set_fontsize(8)
    ax.set_xlabel("SHAP value  (µg/m³ pushed above or below the base prediction)",
                  color=INK_2, fontsize=9)
    ax.set_title(f"Attribution spread across held-out days — fold {fold}, top 20",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(sdir / "fig4_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- fig 5: dependence
    n_dep = 3
    fig, axes = plt.subplots(1, n_dep, figsize=(4.2 * n_dep, 3.8))
    for ax, feat in zip(np.atleast_1d(axes), tax.head(n_dep)["feature"]):
        j = feats.index(feat)
        x = X[:, j].astype("float64")
        y = sv[:, j].astype("float64")
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.scatter(x, y, s=5, alpha=0.25, color=BLUE, linewidths=0, zorder=2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        _style(ax, xlabel=feat.replace("src_merra_", "").replace("_wmean", ""),
               ylabel="SHAP (µg/m³)")
    fig.suptitle(f"How the top predictors act — fold {fold}", color=INK, fontsize=12,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(sdir / "fig5_dependence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- family summary table
    out = []
    for key in ("resolution", "temporal_stat", "source_family"):
        g = (tax.groupby(key).agg(n_features=("feature", "size"),
                                  mean_abs_shap_sum=("mean_abs_shap", "sum"))
             .reset_index().rename(columns={key: "group"}))
        g.insert(0, "dimension", key)
        g["share_pct"] = 100 * g["mean_abs_shap_sum"] / total
        g["shap_per_feature"] = g["mean_abs_shap_sum"] / g["n_features"]
        out.append(g.sort_values("share_pct", ascending=False))
    pd.concat(out, ignore_index=True).to_csv(sdir / "shap_family_summary.csv", index=False)
    print(f"[shap] figures written to {sdir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="TreeSHAP attribution for one thesis fold.")
    ap.add_argument("--fold", type=int, default=4, choices=range(1, 9), metavar="{1..8}")
    ap.add_argument("--stage", choices=["compute", "figures", "all"], default="all")
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data_zerofilled")
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "outputs" / "lgbm_thesis_zerofilled")
    ap.add_argument("--limit", type=int, default=None,
                    help="explain only the first N held-out rows (for a quick look)")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    if args.stage in ("compute", "all"):
        sdir = compute_shap(args.fold, args.shard_root, args.out_root, args.limit)
        # Record the shard root so the figure stage can reload the design matrix.
        p = sdir / "shap_summary.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s["shard_root"] = str(args.shard_root)
        p.write_text(json.dumps(s, indent=2), encoding="utf-8")

    if args.stage in ("figures", "all"):
        make_figures(args.fold, args.out_root, top_n=args.top_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
