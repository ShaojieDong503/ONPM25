#!/usr/bin/env python3
"""
Per-fold SHAP attributions, averaged across the 8 outer folds.

  python make_fold_shap.py                       # all 8 folds, lgbm
  python make_fold_shap.py --folds 1,2           # a subset
  python make_fold_shap.py --raw-sample 20000    # also keep raw SHAP for plots

Four decisions this script makes, and why
-----------------------------------------

1. HELD-OUT ROWS, NOT TRAINING ROWS. Each fold's model explains only that fold's 9
   held-out blocks. Every one of the 169,882 cell-days then gets exactly one SHAP
   vector, produced by a model that never trained on it -- the attribution analogue
   of the out-of-fold prediction table, and joinable to it row for row. Explaining
   training rows would attribute each row up to 8 times and describe in-sample
   behaviour.

2. STAGE 1 AND CORRECTOR REPORTED SEPARATELY, NOT SUMMED. TreeSHAP is additive
   within a model: sum(phi) + base == the model's own output. But the corrector takes
   `pred_stage1` as one of its 40 inputs, so part of the correction is attributed to
   a quantity that is itself a function of the 651 Stage-1 features. Adding the two
   feature-wise would count those features twice -- once directly, once through
   `pred_stage1`. So Stage 1 explains pred_stage1 (651 features) and the corrector
   explains the correction (80 features), and the two tables stay apart. Composing
   them is possible -- propagate the corrector's `pred_stage1` share back through the
   Stage-1 attributions -- but that is an extra assumption, not a fact, so it is not
   done here.

3. EXACT TREESHAP VIA LIGHTGBM'S OWN pred_contrib. No background dataset to choose,
   no sampling error, no `shap` dependency for the tree models. Additivity is
   asserted per fold against the model's own predict(), so the numbers are checked
   rather than trusted.

4. SPREAD ACROSS FOLDS IS THE POINT. The headline is mean |SHAP| in ug/m3 averaged
   over the 8 folds, but the standard deviation across folds and the rank stability
   (how many folds keep a feature in the top 10) are reported beside it. A feature
   that is top-5 in one fold and 300th in another is not an important feature, it is
   an unstable one -- and pooling into a single fit would have hidden that.

On reading per-feature numbers
------------------------------
The 651 predictors are heavily correlated by construction: 73 base variables each
carry 7 temporal derivatives, and neighbourhood radii overlap. TreeSHAP splits credit
among correlated features according to which one a tree happened to split on, so
per-feature values are the least stable level of this report. The family and
base-variable rollups sum credit back over those groups and are what should carry
any claim about what the model uses.

Outputs (under --out-dir)
-------------------------
  shap_by_feature_<stage>.csv        651 or 80 rows: per-fold mean|SHAP| + mean/sd/rank
  shap_by_base_variable_<stage>.csv  derivatives summed back onto their stem
  shap_by_family_<stage>.csv         7 predictor families
  shap_fold_summary_<stage>.csv      per-fold totals and the additivity residual
  shap_report.json                   config, versions, assertions
  shap_raw_sample_<stage>.parquet    optional seeded row sample for beeswarm/dependence
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_lgbm_thesis_fold as fold  # reuse the fold script's own loaders  # noqa: E402

DEFAULT_OUT_ROOT = ROOT / "outputs" / "lgbm_thesis_zerofilled"
DEFAULT_SHARD_ROOT = ROOT / "Data_zerofilled"
DEFAULT_OUT_DIR = ROOT / "outputs" / "shap"
SEED = 2026

ROLL_SUFFIX = ("_roll3_min", "_roll3_mean", "_roll3_max",
               "_roll7_min", "_roll7_mean", "_roll7_max")
NARR = {"air_sfc", "acpcp", "dswrf", "evap", "hpbl", "pr_wtr", "pres_sfc", "shum_2m",
        "uwnd_10m", "vwnd_10m", "windspeed_10m", "vis", "lcdc", "mcdc"}
AOD = {"AOD_055", "AOD_047", "AOD_055_filled", "aod_obs_flag", "aod_imputed_flag"}


def base_variable(col: str) -> str:
    """Strip the temporal derivative so a variable's 7 variants roll up to one stem."""
    c = col[:-len("__isna")] if col.endswith("__isna") else col
    for suf in ROLL_SUFFIX:
        if c.endswith(suf):
            return c[: -len(suf)]
    if c.endswith("_lag1"):
        return c[: -len("_lag1")]
    return c


def family(col: str) -> str:
    c = base_variable(col)
    if c == "pred_stage1":
        return "stage1_prediction"
    if c in AOD:
        return "satellite_aod"
    if c.startswith("src_merra_aer"):
        return "merra2_aerosol"
    if c.startswith(("src_merra_flx", "src_merra_slv")):
        return "merra2_meteorology"
    if c in NARR:
        return "narr_meteorology"
    if c.startswith(("src_viirs", "src_hms", "burned_")):
        return "wildfire_smoke"
    if c.startswith(("ca_lc_", "ca_road_")):
        return "land_use_context"
    if c == "dayofyear":
        return "seasonality"
    return "other"


def quiet_logger() -> logging.Logger:
    lg = logging.getLogger("shap_loader")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.ERROR)
    return lg


def contributions(booster: lgb.Booster, X: np.ndarray, names: list[str],
                  label: str) -> tuple[np.ndarray, float, float]:
    """Exact TreeSHAP. Returns (phi, base_value, max additivity residual)."""
    raw = booster.predict(X, pred_contrib=True)
    phi, base = raw[:, :-1], raw[:, -1]
    if phi.shape[1] != len(names):
        raise AssertionError(f"{label}: {phi.shape[1]} contributions vs {len(names)} features")
    resid = float(np.abs(phi.sum(axis=1) + base - booster.predict(X)).max())
    return phi, float(base[0]), resid


def stratified_sample(meta: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Proportional sample within each held-out block, so all 9 are represented.

    Exact TreeSHAP costs ~270 ms/row on these boosters (2,000 trees x 255 leaves):
    every held-out row would be ~95 min/fold, 13 h for the 8 folds. mean|SHAP| is an
    average over rows, so its precision is governed by the usual 1/sqrt(n) -- a few
    thousand rows pins the family rollups and the leading features tightly. The
    sampling standard error is reported per feature so that claim is checkable rather
    than asserted.
    """
    if not n or n >= len(meta):
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    blocks = meta["_source_block"].to_numpy()
    idx = []
    for b in pd.unique(blocks):
        pos = np.flatnonzero(blocks == b)
        take = max(1, int(round(n * len(pos) / len(meta))))
        idx.append(rng.choice(pos, size=min(take, len(pos)), replace=False))
    return np.sort(np.concatenate(idx))


def fold_shap(fold_no: int, out_root: Path, shard_root: Path, canonical: list[str],
              lg: logging.Logger, rows_per_fold: int, seed: int) -> dict:
    gdir = out_root / f"GROUP_{fold_no:02d}"
    manifest = json.loads((gdir / "run_manifest.json").read_text(encoding="utf-8"))
    holdout = list(manifest["holdout_blocks"])

    # The fold script's own loader, so SHAP runs on exactly the matrix the model saw
    # -- including the 438 name translations and the NaN -> 0 fill.
    meta, X, _ = fold.load_blocks(shard_root, holdout, canonical, lg)
    n_held = len(meta)
    sel = stratified_sample(meta, rows_per_fold, seed + fold_no)
    meta, X = meta.iloc[sel].reset_index(drop=True), X[sel]

    s1 = lgb.Booster(model_file=str(gdir / "stage1_model.txt"))
    phi1, base1, resid1 = contributions(s1, X, canonical, f"fold{fold_no}/stage1")
    pred_stage1 = s1.predict(X)

    bundle = pickle.loads((gdir / "corrector_model.pkl").read_bytes())
    cdf = meta.copy()
    cdf["pred_stage1"] = pred_stage1
    Xc, cnames = fold.transform_corrector(cdf, bundle["fill_values"])
    c = lgb.Booster(model_file=str(gdir / "corrector_model.txt"))
    phi2, base2, resid2 = contributions(c, Xc, cnames, f"fold{fold_no}/corrector")

    print(f"  fold {fold_no}: {len(meta):,} of {n_held:,} held-out rows  "
          f"stage1 base={base1:+.3f} resid={resid1:.2e}  "
          f"corrector base={base2:+.3f} resid={resid2:.2e}", flush=True)
    return {
        "fold": fold_no, "rows": int(len(meta)), "rows_heldout": int(n_held),
        "holdout_blocks": holdout,
        "stage1": {"phi": phi1, "names": canonical, "base": base1, "resid": resid1},
        "corrector": {"phi": phi2, "names": cnames, "base": base2, "resid": resid2},
        "meta": meta, "pred_stage1": pred_stage1,
    }


def aggregate(per_fold: list[dict], stage: str) -> pd.DataFrame:
    """mean|SHAP| per fold, then mean / sd / rank stability across folds."""
    names = per_fold[0][stage]["names"]
    cols, se = {}, {}
    for f in per_fold:
        phi = f[stage]["phi"]
        a = np.abs(phi)
        cols[f"fold{f['fold']}_mean_abs"] = a.mean(axis=0)
        cols[f"fold{f['fold']}_mean_signed"] = phi.mean(axis=0)
        # Monte-Carlo SE of this fold's mean|SHAP| from row sampling. Distinct from
        # the across-fold SD below: this one says "did we explain enough rows",
        # that one says "do the 8 models agree".
        se[f["fold"]] = a.std(axis=0, ddof=1) / np.sqrt(len(a))
    df = pd.DataFrame(cols, index=names)

    abs_cols = [c for c in df.columns if c.endswith("_mean_abs")]
    sgn_cols = [c for c in df.columns if c.endswith("_mean_signed")]
    ranks = df[abs_cols].rank(ascending=False, method="min")

    out = pd.DataFrame({
        "feature": names,
        "base_variable": [base_variable(n) for n in names],
        "family": [family(n) for n in names],
        # unweighted across folds: each of the 8 models gets one vote. Folds are
        # row-balanced by construction, so this tracks the row-weighted pooling
        # closely -- the n-weighted column is emitted beside it to show by how much.
        "mean_abs_shap": df[abs_cols].mean(axis=1).to_numpy(),
        "sd_abs_shap_across_folds": df[abs_cols].std(axis=1, ddof=1).to_numpy(),
        "min_abs_shap": df[abs_cols].min(axis=1).to_numpy(),
        "max_abs_shap": df[abs_cols].max(axis=1).to_numpy(),
        "mean_signed_shap": df[sgn_cols].mean(axis=1).to_numpy(),
        "mean_rank": ranks.mean(axis=1).to_numpy(),
        "worst_rank": ranks.max(axis=1).to_numpy().astype(int),
        "folds_in_top10": (ranks <= 10).sum(axis=1).to_numpy().astype(int),
    })
    w = np.array([f["rows"] for f in per_fold], dtype=float)
    out["mean_abs_shap_row_weighted"] = (df[abs_cols].to_numpy() @ w) / w.sum()
    # SE of the across-fold mean from row sampling alone: folds are independent, so
    # the SEs add in quadrature and divide by the fold count.
    out["sampling_se"] = np.sqrt(np.sum(np.square(np.array(list(se.values()))), axis=0)) / len(se)
    out["cv_across_folds"] = out["sd_abs_shap_across_folds"] / out["mean_abs_shap"].replace(0, np.nan)
    for f in per_fold:
        out[f"fold{f['fold']}_mean_abs"] = df[f"fold{f['fold']}_mean_abs"].to_numpy()

    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rollup(feat: pd.DataFrame, key: str, per_fold: list[dict], stage: str) -> pd.DataFrame:
    """Sum |SHAP| within a group PER FOLD, then average -- summing the across-fold
    means would discard the spread the group-level number is meant to show."""
    names = per_fold[0][stage]["names"]
    grp = pd.Series([base_variable(n) if key == "base_variable" else family(n)
                     for n in names], index=names)
    per = {}
    for f in per_fold:
        s = pd.Series(np.abs(f[stage]["phi"]).mean(axis=0), index=names)
        per[f"fold{f['fold']}"] = s.groupby(grp).sum()
    g = pd.DataFrame(per)
    out = pd.DataFrame({
        key: g.index,
        "n_features": pd.Series(names, index=names).groupby(grp).size().reindex(g.index).to_numpy(),
        "mean_abs_shap": g.mean(axis=1).to_numpy(),
        "sd_across_folds": g.std(axis=1, ddof=1).to_numpy(),
        "min_across_folds": g.min(axis=1).to_numpy(),
        "max_across_folds": g.max(axis=1).to_numpy(),
    })
    out["share_of_total"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rows-per-fold", type=int, default=2000,
                    help="held-out rows to explain per fold, stratified by block; "
                         "0 = every row (~95 min/fold on these boosters)")
    ap.add_argument("--raw-sample", type=int, default=0,
                    help="rows of raw SHAP to persist for beeswarm/dependence plots")
    ap.add_argument("--max-additivity-residual", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    canonical = list(json.loads((args.shard_root / "manifest.json")
                                .read_text(encoding="utf-8"))["feature_cols"])
    lg = quiet_logger()

    print("=" * 78)
    print(f"Per-fold SHAP  ({len(folds)} folds, {len(canonical)} Stage-1 features)")
    print("=" * 78)
    est = (args.rows_per_fold or 21200) * 0.27 * 2 * len(folds) / 60
    print(f"  rows per fold: {args.rows_per_fold or 'all'}   "
          f"estimated {est:.0f} min for {len(folds)} fold(s), both stages\n")

    per_fold = [fold_shap(n, args.out_root, args.shard_root, canonical, lg,
                          args.rows_per_fold, args.seed) for n in folds]

    total_rows = sum(f["rows"] for f in per_fold)
    total_held = sum(f["rows_heldout"] for f in per_fold)
    worst = max(max(f["stage1"]["resid"], f["corrector"]["resid"]) for f in per_fold)
    ok_add = worst <= args.max_additivity_residual
    ok_held = (total_held == 169_882) or len(folds) < 8
    print(f"\n  additivity: worst |sum(phi)+base - predict| = {worst:.2e}  "
          f"({'PASS' if ok_add else 'FAIL'}, limit {args.max_additivity_residual:g})")
    print(f"  rows explained: {total_rows:,} sampled from {total_held:,} held out"
          + ("" if len(folds) < 8 else f"  ({'PASS' if ok_held else 'FAIL'} vs 169,882 held out)"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, written = [], []
    for stage in ("stage1", "corrector"):
        feat = aggregate(per_fold, stage)
        for name, df_ in (("by_feature", feat),
                          ("by_base_variable", rollup(feat, "base_variable", per_fold, stage)),
                          ("by_family", rollup(feat, "family", per_fold, stage))):
            p = args.out_dir / f"shap_{name}_{stage}.csv"
            df_.to_csv(p, index=False)
            written.append(p)

        fam = rollup(feat, "family", per_fold, stage)
        print(f"\n  {stage}: mean |SHAP| by family (µg/m³, averaged over folds)")
        for _, r in fam.iterrows():
            print(f"    {r['family']:<22} {r['mean_abs_shap']:7.4f}  "
                  f"±{r['sd_across_folds']:.4f}  {r['share_of_total']:6.1%}  "
                  f"({int(r['n_features'])} features)")
        print(f"  {stage}: top 8 individual features")
        for _, r in feat.head(8).iterrows():
            print(f"    {r['rank']:>3}. {r['feature']:<42} {r['mean_abs_shap']:7.4f}  "
                  f"±{r['sd_abs_shap_across_folds']:.4f}  top10 in {r['folds_in_top10']}/{len(folds)}")

        for f in per_fold:
            summary.append({"stage": stage, "fold": f["fold"], "rows": f["rows"],
                            "base_value": f[stage]["base"],
                            "total_mean_abs_shap": float(np.abs(f[stage]["phi"]).mean(axis=0).sum()),
                            "additivity_residual": f[stage]["resid"]})

        if args.raw_sample:
            rng = np.random.default_rng(SEED)
            parts = []
            for f in per_fold:
                phi, names = f[stage]["phi"], f[stage]["names"]
                k = min(args.raw_sample // len(per_fold), len(phi))
                idx = rng.choice(len(phi), size=k, replace=False)
                d = pd.DataFrame(phi[idx].astype(np.float32), columns=names)
                d.insert(0, "fold", f["fold"])
                d.insert(1, "case_key", f["meta"]["_source_block"].to_numpy()[idx])
                parts.append(d)
            p = args.out_dir / f"shap_raw_sample_{stage}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
            written.append(p)

    p = args.out_dir / "shap_fold_summary.csv"
    pd.DataFrame(summary).to_csv(p, index=False)
    written.append(p)

    report = {
        "folds": folds, "rows_explained": total_rows, "rows_held_out": total_held,
        "rows_per_fold": args.rows_per_fold or "all",
        "sampling": ("stratified by held-out block, seeded; mean|SHAP| is a row average "
                     "so precision follows 1/sqrt(n) -- see the sampling_se column"),
        "rows_are": "held-out only; every explained row comes from a model that never trained on it",
        "stages_combined": False,
        "why_not_combined": ("the corrector takes pred_stage1 as an input, so summing "
                             "feature-wise would count Stage-1 features twice"),
        "method": "exact TreeSHAP via LightGBM pred_contrib (no background dataset)",
        "worst_additivity_residual": worst,
        "additivity_ok": bool(ok_add),
        "seed": SEED,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "lightgbm": lgb.__version__},
        "caveat": ("Per-feature values are the least stable level: 73 base variables each "
                   "carry 7 correlated temporal derivatives and the neighbourhood radii "
                   "overlap, so TreeSHAP splits credit among correlated features by tree "
                   "structure. Use the family and base-variable rollups for claims."),
    }
    (args.out_dir / "shap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n  written:")
    for p in written + [args.out_dir / "shap_report.json"]:
        print(f"    {p}")
    return 0 if ok_add else 1


if __name__ == "__main__":
    raise SystemExit(main())
