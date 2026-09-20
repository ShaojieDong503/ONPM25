#!/usr/bin/env python3
"""
Build region-year block metrics for the CanOSSEM final product.

Reads the CanOSSEM WFS extracts directly, joins them to the monitored grid-cell-days,
and writes per-block and pooled metrics using the same definitions as
`holdout_block_metrics.csv`, so the two tables can sit side by side.

Quick start
-----------
  python build_canossem_block_metrics.py                       # auto-detect the obs source
  python build_canossem_block_metrics.py --obs-from shards     # observations from the shard frames
  python build_canossem_block_metrics.py --oof-root outputs/xgb_thesis_zerofilled
  python build_canossem_block_metrics.py --no-strict           # warn instead of exiting 1

Observation source
------------------
Two modes, auto-detected by default:

  oof     Read obs_pm25 AND our predictions from <oof-root>/GROUP_*/holdout_predictions.parquet.
          Emits canossem_* and ours_* columns plus per-block margins, so the comparison
          table is complete in one file. Requires a finished 8-fold run.

  shards  Read observations from <shard-root>/pair_blocks/*/frame.parquet. Emits
          canossem_* only. Works before any model has been fitted.

Both modes evaluate CanOSSEM against the identical observations, per spec V.

The deduplication that this script exists for
---------------------------------------------
The WFS extract emits 4 cells near the provincial border TWICE per day for all 12
years -- one row tagged PROVINCE_TERRITORY='ON', an identical twin tagged
'Out-of-Canada'. PM2.5 is bit-identical in all 17,510 affected cell-days, so dropping
one is lossless, but a naive left merge fans 169,882 rows out to 185,310 and
double-weights every West_SW and East block by up to 35%.

The previous comparison table (outputs/gcloud_result_evaluation/
gcloud_vs_canossem_case_metrics.csv) was built that way: its canossem_n sums to
185,076 rather than 169,648. This script deduplicates on (date, grid_cell_id) before
joining and asserts the resulting counts, so the failure cannot recur silently.

Not a like-for-like comparison
------------------------------
Our side is held out (out-of-fold); the CanOSSEM side is a final-product estimate,
not CanOSSEM out-of-fold predictions. This is a reference benchmark, not an external
validation of either product. The caveat is written into the emitted JSON so it
travels with the numbers.

Outputs (under --out-dir, default outputs/canossem_benchmark/)
--------------------------------------------------------------
  canossem_block_metrics.csv        72 rows, one per region-year block
  canossem_pooled_metrics.json      pooled metrics + run configuration + the caveat
  canossem_matched_cell_days.parquet  the row-level matched table (spec Z item 17)
  canossem_build_report.json        every assertion, expected vs observed

Exit code is non-zero if any assertion fails (unless --no-strict).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LAMBDA = ROOT.parent

DEFAULT_CANOSSEM_DIR = LAMBDA / "CanOSSEM_data"
DEFAULT_OOF_ROOT = ROOT / "outputs" / "lgbm_thesis_zerofilled"
DEFAULT_SHARD_ROOT = ROOT / "Data_zerofilled"
DEFAULT_OUT_DIR = ROOT / "outputs" / "canossem_benchmark"

YEAR_MIN, YEAR_MAX = 2012, 2023

# Spec A / Y: the monitored evaluation table.
EXPECT_TARGET_ROWS = 169_882
EXPECT_MATCHED_ROWS = 169_648      # spec V; the 234 unmatched are cell-days absent from the WFS extract
EXPECT_BLOCKS = 72
EXPECT_CELLS = 41

CELL_COL = "CanOSSEM_RASTER_CELL"
CAN_PM_COL = "CanOSSEM_RASTER_PM2.5"
CAN_OBS_COL = "OBSERVED_RASTER_PM2.5"

_RESULTS: list[dict] = []


# ---------------------------------------------------------------- assertions
def check(name: str, ok: bool, expected, observed, note: str = "") -> bool:
    _RESULTS.append({"check": name, "ok": bool(ok), "expected": expected,
                     "observed": observed, "note": note})
    tail = f"   {note}" if note else ""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected {expected}, observed {observed}{tail}",
          flush=True)
    return ok


# ---------------------------------------------------------------- metrics
def block_metrics(obs: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    """Same definitions as holdout_block_metrics.csv.

    R2 is PREDICTIVE (1 - SSE/SST), never squared Pearson, so it may be negative for a
    poorly performing block. Bias is pred - obs, so positive means overprediction.
    """
    obs = np.asarray(obs, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    n = len(obs)
    err = pred - obs
    sse = float((err ** 2).sum())
    sst = float(((obs - obs.mean()) ** 2).sum())
    if n >= 2:
        slope, intercept = np.polyfit(obs, pred, 1)
    else:
        slope = intercept = np.nan
    return {
        f"{prefix}_n": int(n),
        f"{prefix}_rmse": float(np.sqrt(sse / n)) if n else np.nan,
        f"{prefix}_mae": float(np.abs(err).mean()) if n else np.nan,
        f"{prefix}_r2_predictive": float(1.0 - sse / sst) if sst > 0 else np.nan,
        f"{prefix}_bias_pred_minus_obs": float(err.mean()) if n else np.nan,
        f"{prefix}_within_3": float((np.abs(err) <= 3.0).mean()) if n else np.nan,
        f"{prefix}_pred_on_obs_slope": float(slope),
        f"{prefix}_pred_on_obs_intercept": float(intercept),
    }


# ---------------------------------------------------------------- observations
def load_obs_from_oof(oof_root: Path) -> pd.DataFrame:
    files = sorted(oof_root.glob("GROUP_*/holdout_predictions.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No GROUP_*/holdout_predictions.parquet under {oof_root}. "
            f"Run the 8 folds first, or use --obs-from shards.")
    cols = ["grid_cell_id", "date", "year", "region", "naps_id", "station_name",
            "case_key", "outer_fold", "model_family", "obs_pm25",
            "pred_stage1", "pred_corrector", "pred_final"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    print(f"  {len(files)} fold file(s) -> {len(df):,} rows")
    fams = sorted(df["model_family"].unique())
    if len(fams) != 1:
        raise ValueError(f"Expected one model_family in {oof_root}, found {fams}")
    return df


def load_obs_from_shards(shard_root: Path) -> pd.DataFrame:
    blocks = sorted((shard_root / "pair_blocks").glob("PAIR_*"))
    if not blocks:
        raise FileNotFoundError(f"No pair_blocks/PAIR_* under {shard_root}")
    cols = ["date", "CanOSSEM_RASTER_CELL", "pm25", "year", "fold_region",
            "naps_id", "station_name"]
    parts = []
    for b in blocks:
        d = pd.read_parquet(b / "frame.parquet", columns=cols)
        d["case_key"] = b.name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"CanOSSEM_RASTER_CELL": "grid_cell_id",
                            "pm25": "obs_pm25", "fold_region": "region"})
    print(f"  {len(blocks)} block(s) -> {len(df):,} rows")
    return df


# ---------------------------------------------------------------- canossem
def load_canossem(canossem_dir: Path, cells: set[int], years: range) -> tuple[pd.DataFrame, dict]:
    parts, missing_years = [], []
    for y in years:
        f = canossem_dir / f"CanOSSEM_{y}_WFS_Ontario.parquet"
        if not f.exists():
            missing_years.append(y)
            continue
        d = pd.read_parquet(f, columns=["DATE", CELL_COL, CAN_PM_COL, CAN_OBS_COL])
        # double in CanOSSEM, string in the shard frames -- both must become int64 or
        # the merge silently matches nothing.
        d["grid_cell_id"] = d[CELL_COL].astype("int64")
        parts.append(d[d["grid_cell_id"].isin(cells)])
    if missing_years:
        raise FileNotFoundError(f"Missing CanOSSEM year file(s): {missing_years}")
    can = pd.concat(parts, ignore_index=True)
    can["date"] = pd.to_datetime(can["DATE"]).dt.normalize()

    n_raw = len(can)
    dup_mask = can.duplicated(["date", "grid_cell_id"], keep=False)
    dup_rows = int(dup_mask.sum())
    dup_groups = int(can.loc[dup_mask].groupby(["date", "grid_cell_id"]).ngroups)
    dup_cells = int(can.loc[dup_mask, "grid_cell_id"].nunique())
    # Lossless only if the twins agree. Verify rather than assume.
    spread = (can.loc[dup_mask].groupby(["date", "grid_cell_id"])[CAN_PM_COL]
              .agg(lambda s: s.max() - s.min()))
    max_spread = float(spread.max()) if len(spread) else 0.0

    can = can.drop_duplicates(["date", "grid_cell_id"], keep="first")
    stats = {"rows_raw": n_raw, "rows_deduplicated": len(can),
             "duplicate_rows_dropped": n_raw - len(can),
             "duplicate_cell_days": dup_groups, "duplicate_cells": dup_cells,
             "max_within_group_pm25_spread": max_spread}
    print(f"  {n_raw:,} rows at monitored cells -> {len(can):,} after dedup "
          f"({dup_groups:,} duplicated cell-days across {dup_cells} cells)")
    return can[["date", "grid_cell_id", CAN_PM_COL, CAN_OBS_COL]], stats


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-block metrics for the CanOSSEM final product.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--canossem-dir", type=Path, default=DEFAULT_CANOSSEM_DIR,
                    help="directory holding CanOSSEM_<year>_WFS_Ontario.parquet")
    ap.add_argument("--obs-from", choices=["auto", "oof", "shards"], default="auto",
                    help="where observations come from; auto prefers oof when available")
    ap.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT,
                    help="directory holding GROUP_*/holdout_predictions.parquet")
    ap.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT,
                    help="shard root holding pair_blocks/PAIR_*/frame.parquet")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--year-min", type=int, default=YEAR_MIN)
    ap.add_argument("--year-max", type=int, default=YEAR_MAX)
    ap.add_argument("--no-strict", action="store_true",
                    help="report failed assertions but exit 0")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    years = range(args.year_min, args.year_max + 1)

    print("=" * 78)
    print("CanOSSEM block metrics")
    print("=" * 78)

    # ------------------------------------------------------------ observations
    mode = args.obs_from
    if mode == "auto":
        mode = "oof" if sorted(args.oof_root.glob("GROUP_*/holdout_predictions.parquet")) else "shards"
    print(f"\n### observations  (source={mode})")
    tgt = load_obs_from_oof(args.oof_root) if mode == "oof" else load_obs_from_shards(args.shard_root)
    has_ours = "pred_final" in tgt.columns
    family = str(tgt["model_family"].iloc[0]) if has_ours else None

    tgt["date"] = pd.to_datetime(tgt["date"]).dt.normalize()
    tgt["grid_cell_id"] = tgt["grid_cell_id"].astype("int64")
    cells = set(tgt["grid_cell_id"].unique())

    check("target.rows", len(tgt) == EXPECT_TARGET_ROWS, EXPECT_TARGET_ROWS, len(tgt))
    check("target.blocks", tgt["case_key"].nunique() == EXPECT_BLOCKS,
          EXPECT_BLOCKS, tgt["case_key"].nunique())
    check("target.cells", len(cells) == EXPECT_CELLS, EXPECT_CELLS, len(cells))
    n_dup_t = int(tgt.duplicated(["date", "grid_cell_id"]).sum())
    check("target.no_duplicate_cell_days", n_dup_t == 0, 0, n_dup_t,
          "a duplicated observation would be double-counted in every metric")

    # ------------------------------------------------------------ canossem
    print(f"\n### canossem  ({args.canossem_dir})")
    can, dstats = load_canossem(args.canossem_dir, cells, years)
    check("canossem.duplicates_are_lossless", dstats["max_within_group_pm25_spread"] == 0.0,
          0.0, dstats["max_within_group_pm25_spread"],
          "twins must agree on PM2.5 for keep='first' to be information-preserving")
    n_dup_c = int(can.duplicated(["date", "grid_cell_id"]).sum())
    check("canossem.deduplicated", n_dup_c == 0, 0, n_dup_c)

    # ------------------------------------------------------------ join
    print("\n### join")
    m = tgt.merge(can, on=["date", "grid_cell_id"], how="left")
    check("join.no_row_fanout", len(m) == len(tgt), len(tgt), len(m),
          "a left merge onto duplicated keys silently inflates the table")
    matched = m[CAN_PM_COL].notna()
    n_matched = int(matched.sum())
    check("join.matched_rows", n_matched == EXPECT_MATCHED_ROWS,
          EXPECT_MATCHED_ROWS, n_matched,
          f"{len(m) - n_matched} cell-days absent from the WFS extract")
    m = m[matched].copy()

    # CanOSSEM's own daily aggregate is NOT our NAPS daily mean -- different hour
    # handling. Reported as a diagnostic; never used as the target.
    agree = float((m[CAN_OBS_COL] - m["obs_pm25"]).abs().lt(1e-6).mean())
    print(f"  CanOSSEM OBSERVED_RASTER_PM2.5 identical to our NAPS daily mean: {agree:.2%} "
          f"(diagnostic only; obs_pm25 is the target for both sides)")

    # ------------------------------------------------------------ metrics
    print("\n### metrics")
    rows = []
    for case_key, g in m.groupby("case_key", sort=True):
        rec = {"case_key": case_key, "year": int(g["year"].iloc[0]),
               "region": str(g["region"].iloc[0]), "rows_in_block": int(len(g))}
        if mode == "oof":
            rec["outer_fold"] = int(g["outer_fold"].iloc[0])
        rec.update(block_metrics(g["obs_pm25"], g[CAN_PM_COL], "canossem"))
        if has_ours:
            rec.update(block_metrics(g["obs_pm25"], g["pred_stage1"], "ours_stage1"))
            rec.update(block_metrics(g["obs_pm25"], g["pred_final"], "ours_final"))
            rec["rmse_margin_canossem_minus_ours"] = rec["canossem_rmse"] - rec["ours_final_rmse"]
            rec["r2_margin_ours_minus_canossem"] = (rec["ours_final_r2_predictive"]
                                                    - rec["canossem_r2_predictive"])
            rec["winner"] = "ours" if rec["ours_final_rmse"] < rec["canossem_rmse"] else "canossem"
        rows.append(rec)
    blocks = pd.DataFrame(rows).sort_values(["year", "region"]).reset_index(drop=True)

    check("metrics.blocks", len(blocks) == EXPECT_BLOCKS, EXPECT_BLOCKS, len(blocks))
    check("metrics.n_sums_to_matched", int(blocks["canossem_n"].sum()) == n_matched,
          n_matched, int(blocks["canossem_n"].sum()),
          "per-block n must reconcile with the matched row count")

    pooled = {"canossem": block_metrics(m["obs_pm25"], m[CAN_PM_COL], "canossem")}
    if has_ours:
        pooled["ours_stage1"] = block_metrics(m["obs_pm25"], m["pred_stage1"], "ours_stage1")
        pooled["ours_final"] = block_metrics(m["obs_pm25"], m["pred_final"], "ours_final")

    # ------------------------------------------------------------ write
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{family}" if family else ""
    p_blocks = args.out_dir / f"canossem_block_metrics{suffix}.csv"
    p_pooled = args.out_dir / f"canossem_pooled_metrics{suffix}.json"
    p_rows = args.out_dir / f"canossem_matched_cell_days{suffix}.parquet"
    p_report = args.out_dir / f"canossem_build_report{suffix}.json"

    blocks.to_csv(p_blocks, index=False)
    keep = ["grid_cell_id", "date", "year", "region", "case_key", "naps_id",
            "station_name", "obs_pm25", CAN_PM_COL, CAN_OBS_COL]
    if mode == "oof":
        keep = keep[:5] + ["outer_fold", "model_family"] + keep[5:] + \
               ["pred_stage1", "pred_corrector", "pred_final"]
    m[[c for c in keep if c in m.columns]].to_parquet(p_rows, index=False)

    config = {
        "canossem_dir": str(args.canossem_dir), "obs_source": mode,
        "oof_root": str(args.oof_root) if mode == "oof" else None,
        "shard_root": str(args.shard_root) if mode == "shards" else None,
        "years": [args.year_min, args.year_max], "model_family": family,
        "deduplication": dstats,
        "canossem_observed_agrees_with_naps_daily_mean_frac": agree,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__},
        "caveat": ("Not like-for-like: our predictions are held out (out-of-fold), "
                   "CanOSSEM values are final-product estimates rather than CanOSSEM "
                   "out-of-fold predictions. This is a reference benchmark, not an "
                   "external validation of either product."),
    }
    p_pooled.write_text(json.dumps({"pooled": pooled, "config": config}, indent=2,
                                   default=str), encoding="utf-8")
    failed = [r for r in _RESULTS if not r["ok"]]
    p_report.write_text(json.dumps({"checks": _RESULTS, "n_checks": len(_RESULTS),
                                    "n_failed": len(failed), "config": config},
                                   indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------ summary
    print(f"\n### pooled  (n={pooled['canossem']['canossem_n']:,})")
    c = pooled["canossem"]
    print(f"  canossem     R2={c['canossem_r2_predictive']:.4f}  RMSE={c['canossem_rmse']:.4f}  "
          f"MAE={c['canossem_mae']:.4f}  bias={c['canossem_bias_pred_minus_obs']:+.4f}  "
          f"within3={c['canossem_within_3']:.4f}")
    if has_ours:
        o = pooled["ours_final"]
        print(f"  ours ({family:<4}) R2={o['ours_final_r2_predictive']:.4f}  "
              f"RMSE={o['ours_final_rmse']:.4f}  MAE={o['ours_final_mae']:.4f}  "
              f"bias={o['ours_final_bias_pred_minus_obs']:+.4f}  "
              f"within3={o['ours_final_within_3']:.4f}")
        w = int((blocks["winner"] == "ours").sum())
        print(f"  ours has lower RMSE in {w} of {len(blocks)} blocks")
    neg = blocks[blocks["canossem_r2_predictive"] < 0]
    if len(neg):
        print(f"\n  blocks with negative CanOSSEM predictive R2: {len(neg)}")
        for _, r in neg.iterrows():
            print(f"    {r['case_key']:<20} R2={r['canossem_r2_predictive']:+.4f}  "
                  f"bias={r['canossem_bias_pred_minus_obs']:+.3f}")

    print(f"\n  {p_blocks}")
    print(f"  {p_pooled}")
    print(f"  {p_rows}")
    print(f"  {p_report}")
    print(f"\n[done] {len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    for r in failed:
        print(f"  FAILED {r['check']}: expected {r['expected']}, observed {r['observed']}")
    if failed and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
