#!/usr/bin/env python3
"""
Compare the three learner families on the identical out-of-fold predictions.

Every metric is recomputed here from `holdout_predictions.parquet`, so all three
families are measured the same way rather than trusting each run's stored metrics.
The three tables are first checked to cover the same 169,882 cell-days in the same
fold assignment -- otherwise the comparison is not like-for-like.

    python compare_models.py
    python compare_models.py --out-dir outputs/model_comparison
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# LightGBM was re-fitted end to end on the raw shards; XGB and RF kept their original
# Stage-1 models (bit-identical either way) and had only their correctors refitted, so
# their corrected held-out tables sit beside the originals as *_refit.parquet.
FAMILIES = {
    "lgbm": ROOT / "outputs" / "lgbm_thesis",
    "xgb": ROOT / "outputs" / "xgb_thesis",
    "rf": ROOT / "outputs" / "rf_thesis",
}
LABEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}


# ---------------------------------------------------------------- metrics

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Predictive R^2 (1 - SSE/SST), not squared correlation."""
    y = np.asarray(y, "float64"); p = np.asarray(p, "float64")
    err = p - y
    sse = float((err ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float("nan") if sst == 0 else 1.0 - sse / sst,
        "bias": float(err.mean()),
        "slope": float(np.polyfit(y, p, 1)[0]) if y.size >= 3 else float("nan"),
        "within3": float((np.abs(err) <= 3.0).mean()),
    }


def load_oof(path: Path) -> pd.DataFrame:
    """Prefer the corrector-refit table, and say which one was used.

    A run whose corrector was refitted keeps its original holdout_predictions.parquet
    on disk, carrying the contaminated pred_corrector/pred_final. Reading that by
    default would silently compare a corrected family against uncorrected ones, so the
    refit file wins and the choice is printed rather than assumed.
    """
    refit = sorted(glob.glob(str(path / "GROUP_*" / "holdout_predictions_refit.parquet")))
    orig = sorted(glob.glob(str(path / "GROUP_*" / "holdout_predictions.parquet")))
    if refit and len(refit) == len(orig):
        files, which = refit, "holdout_predictions_refit.parquet (corrector refitted)"
    elif refit:
        raise FileNotFoundError(
            f"{path.name}: {len(refit)} refit tables but {len(orig)} folds -- a partial "
            f"refit would mix corrected and uncorrected folds. Finish or remove it.")
    else:
        files, which = orig, "holdout_predictions.parquet"
    print(f"  [{path.name}] {which}  ({len(files)} folds)")
    parts = [pd.read_parquet(f) for f in files]
    if not parts:
        raise FileNotFoundError(f"no holdout predictions under {path}")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["key"] = df["grid_cell_id"].astype(str) + "@" + df["date"].dt.strftime("%Y-%m-%d")
    return df


def check_comparable(frames: dict[str, pd.DataFrame]) -> list[str]:
    """The three families must be scored on the same rows in the same folds."""
    notes = []
    ref_name, ref = next(iter(frames.items()))
    for name, df in frames.items():
        if name == ref_name:
            continue
        if len(df) != len(ref):
            notes.append(f"{name}: {len(df):,} rows vs {ref_name} {len(ref):,}")
        if set(df["key"]) != set(ref["key"]):
            notes.append(f"{name}: covers a different set of cell-days than {ref_name}")
        m = df.set_index("key")["outer_fold"]
        r = ref.set_index("key")["outer_fold"]
        common = m.index.intersection(r.index)
        n_diff = int((m.loc[common] != r.loc[common]).sum())
        if n_diff:
            notes.append(f"{name}: {n_diff:,} rows assigned to a different fold than {ref_name}")
        obs_m = df.set_index("key")["obs_pm25"].loc[common].to_numpy("float64")
        obs_r = ref.set_index("key")["obs_pm25"].loc[common].to_numpy("float64")
        if not np.allclose(obs_m, obs_r, rtol=0, atol=1e-4):
            notes.append(f"{name}: observed target differs from {ref_name}")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare learner families on identical OOF rows.")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "model_comparison")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {k: load_oof(p) for k, p in FAMILIES.items() if p.exists()}
    print(f"[compare] families: {', '.join(frames)}")

    notes = check_comparable(frames)
    if notes:
        print("[compare] WARNING - the runs are not strictly like-for-like:")
        for n in notes:
            print("   -", n)
    else:
        print("[compare] all families cover the same cell-days, folds and observations")

    # ------------------------------------------------------------ pooled
    rows = []
    for k, df in frames.items():
        for stage, col in (("stage1", "pred_stage1"), ("final", "pred_final")):
            m = metrics(df["obs_pm25"], df[col])
            rows.append({"family": k, "label": LABEL[k], "stage": stage, **m})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(args.out_dir / "pooled_metrics.csv", index=False)

    # ------------------------------------------------------------ per fold
    rows = []
    for k, df in frames.items():
        for fold, sub in df.groupby("outer_fold"):
            m = metrics(sub["obs_pm25"], sub["pred_final"])
            m1 = metrics(sub["obs_pm25"], sub["pred_stage1"])
            rows.append({"family": k, "fold": int(fold), "n": m["n"],
                         "stage1_rmse": m1["rmse"], "stage1_r2": m1["r2"],
                         "final_rmse": m["rmse"], "final_mae": m["mae"],
                         "final_r2": m["r2"], "final_bias": m["bias"]})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(args.out_dir / "per_fold_metrics.csv", index=False)

    # ------------------------------------------------------------ by year / region / band
    ref = next(iter(frames.values()))
    cuts = [(-np.inf, 12, "< 12"), (12, 25, "12-25"), (25, 50, "25-50"), (50, np.inf, ">= 50")]
    rows = []
    for k, df in frames.items():
        for lo, hi, name in cuts:
            m_ = (df["obs_pm25"] >= lo) & (df["obs_pm25"] < hi)
            if m_.sum() < 30:
                continue
            m = metrics(df.loc[m_, "obs_pm25"], df.loc[m_, "pred_final"])
            rows.append({"family": k, "dimension": "obs_band", "group": name, **m})
        for dim in ("year", "region"):
            for g, sub in df.groupby(dim):
                m = metrics(sub["obs_pm25"], sub["pred_final"])
                rows.append({"family": k, "dimension": dim, "group": str(g), **m})
    strat = pd.DataFrame(rows)
    strat.to_csv(args.out_dir / "stratified_metrics.csv", index=False)

    # ------------------------------------------------------------ ensemble
    keys = ref[["key", "obs_pm25"]].copy()
    for k, df in frames.items():
        keys = keys.merge(df[["key", "pred_final"]].rename(columns={"pred_final": k}), on="key")
    ens = keys[list(frames)].mean(axis=1)
    ens_m = metrics(keys["obs_pm25"], ens)

    # ------------------------------------------------------------ report
    lines = []
    lines.append("# Model comparison — LightGBM vs XGBoost vs Random Forest\n")
    lines.append(f"All three scored on the identical {len(ref):,} out-of-fold grid-cell-days "
                 f"across 72 region-year blocks and the same 8 folds.\n")
    if notes:
        lines.append("**Caveat — the runs are not strictly like-for-like:**\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Pooled out-of-fold\n")
    lines.append("| model | stage | RMSE | MAE | R² | bias | slope | within 3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in pooled.sort_values(["stage", "rmse"]).iterrows():
        lines.append(f"| {r['label']} | {r['stage']} | {r['rmse']:.4f} | {r['mae']:.4f} | "
                     f"{r['r2']:.4f} | {r['bias']:+.4f} | {r['slope']:.4f} | {r['within3']:.3f} |")
    lines.append("")
    lines.append(f"**Mean of the three (equal weight):** RMSE {ens_m['rmse']:.4f}, "
                 f"MAE {ens_m['mae']:.4f}, R² {ens_m['r2']:.4f}\n")

    lines.append("## Corrector effect (final vs stage-1)\n")
    lines.append("| model | stage-1 RMSE | final RMSE | Δ RMSE | stage-1 MAE | final MAE | Δ MAE |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in frames:
        s = pooled[(pooled.family == k) & (pooled.stage == "stage1")].iloc[0]
        f = pooled[(pooled.family == k) & (pooled.stage == "final")].iloc[0]
        lines.append(f"| {LABEL[k]} | {s['rmse']:.4f} | {f['rmse']:.4f} | {f['rmse']-s['rmse']:+.4f} "
                     f"| {s['mae']:.4f} | {f['mae']:.4f} | {f['mae']-s['mae']:+.4f} |")
    lines.append("")

    lines.append("## Per fold (final RMSE)\n")
    piv = per_fold.pivot(index="fold", columns="family", values="final_rmse")
    piv = piv[[c for c in ("lgbm", "xgb", "rf") if c in piv.columns]]
    lines.append("| fold | " + " | ".join(LABEL[c] for c in piv.columns) + " | best |")
    lines.append("|---" * (len(piv.columns) + 2) + "|")
    for fold, r in piv.iterrows():
        best = LABEL[r.idxmin()]
        lines.append(f"| {fold} | " + " | ".join(f"{v:.4f}" for v in r) + f" | {best} |")
    lines.append("")
    wins = piv.idxmin(axis=1).value_counts()
    lines.append("Folds won: " + ", ".join(f"{LABEL[k]} {v}" for k, v in wins.items()) + "\n")

    lines.append("## By observed concentration band (final RMSE)\n")
    band = strat[strat.dimension == "obs_band"].pivot(index="group", columns="family", values="rmse")
    band = band.reindex([c[2] for c in cuts]).dropna(how="all")
    nvals = strat[strat.dimension == "obs_band"].pivot(index="group", columns="family", values="n")
    lines.append("| band (µg/m³) | n | " + " | ".join(LABEL[c] for c in band.columns) + " |")
    lines.append("|---" * (len(band.columns) + 2) + "|")
    for g, r in band.iterrows():
        lines.append(f"| {g} | {int(nvals.loc[g].iloc[0]):,} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    lines.append("")

    lines.append("## By year (final RMSE)\n")
    yr = strat[strat.dimension == "year"].pivot(index="group", columns="family", values="rmse")
    lines.append("| year | " + " | ".join(LABEL[c] for c in yr.columns) + " |")
    lines.append("|---" * (len(yr.columns) + 1) + "|")
    for g, r in yr.iterrows():
        lines.append(f"| {g} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    lines.append("")

    path = args.out_dir / "model_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    # The report uses Unicode (delta, superscript two, micro). A Windows console at
    # cp1252 cannot encode those, so echo through the console's own codec rather than
    # letting the whole run die after the file is already written.
    enc = sys.stdout.encoding or "utf-8"
    for line in lines[:40]:
        print(line.encode(enc, errors="replace").decode(enc))
    print(f"\n[compare] wrote {path}")
    print(f"[compare] wrote pooled_metrics.csv, per_fold_metrics.csv, stratified_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
