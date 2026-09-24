#!/usr/bin/env python3
"""
Predict daily PM2.5 for every Ontario raster cell with the final two-stage model.

Input is the grid feature build under `ontario_surface_build/`:
    temporal_tables/temporal_YYYY_MM.parquet   140 base predictors per cell-day
    static_features.parquet                    land cover + roads, per cell
Those supply the 140 BASE features; the 511 lag/rolling derivatives are built here
with the same rule the shard builder uses.

Chunked by CELL, never by month
-------------------------------
roll7 needs seven days of lookback, so a cell's whole 2012-2023 series has to be in
memory together -- month-wise chunking would corrupt every month boundary. One cell
is ~11 MB at 651 float32, so cells are processed in batches. Nothing materialises the
full 46.7M x 651 matrix (~138 GB); only predictions are kept.

    python predict_raster_cells.py --dry-run        # plan and feature check only
    python predict_raster_cells.py --years 2012 2013
    python predict_raster_cells.py                  # all years, ~46.7M cell-days

Output: grid_predictions_lgbm.parquet (cell, date, year, pred_stage1, pred_final)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import thesis_core as TC  # noqa: E402

DEFAULT_GRID = Path(r"D:\lambda\Ontario_RealTarget_GPD\outputs\ontario_surface_build")
CELL, DATE = "CanOSSEM_RASTER_CELL", "date"


def stored_name(c: str) -> str:
    return (c.replace("_roll3_", "_lag1_roll3_").replace("_roll7_", "_lag1_roll7_")
            if ("_roll3_" in c or "_roll7_" in c) else c)


def temporal_bases(shard_root: Path) -> list[str]:
    return [l.strip() for l in (shard_root / "temporal_feature_selection.txt")
            .read_text(encoding="utf-8").splitlines() if l.strip()]


def build_derivatives(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """The shard builder's rule, verbatim (build_support_family_temporal_x_shards.py).

    shift(1) -> float32 -> rolling(w, min_periods=1) over the LAGGED series, grouped
    by cell, sorted by date. The current day is never inside its own window.
    """
    df = df.sort_values([CELL, DATE]).reset_index(drop=True)
    cell = df[CELL]
    new = {}
    for b in bases:
        lag1 = df.groupby(CELL, sort=False)[b].shift(1).astype("float32")
        new[stored_name(f"{b}_lag1")] = lag1
        g = lag1.groupby(cell, sort=False)
        for w in (3, 7):
            r = g.rolling(w, min_periods=1)
            for stat in ("min", "mean", "max"):
                s = getattr(r, stat)().reset_index(level=0, drop=True).astype("float32")
                new[stored_name(f"{b}_roll{w}_{stat}")] = s
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def month_files(grid_dir: Path, years: tuple[int, int] | None) -> list[Path]:
    fs = sorted(Path(p) for p in glob.glob(str(grid_dir / "temporal_tables" / "temporal_*.parquet")))
    if years:
        lo, hi = years
        fs = [f for f in fs if lo <= int(f.stem.split("_")[1]) <= hi]
    return fs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID)
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "outputs" / "final_ontario_model")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "raster_prediction")
    ap.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"))
    ap.add_argument("--cells-per-batch", type=int, default=600,
                    help="cells held in memory at once (~11 MB each at 651 float32)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check inputs and feature coverage, predict nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TC.configure_logger(args.out_dir / "raster_prediction.log", args.verbose)

    feats = TC.canonical_features(args.shard_root)
    bases = temporal_bases(args.shard_root)
    files = month_files(args.grid_dir, tuple(args.years) if args.years else None)
    if not files:
        raise SystemExit(f"[error] no temporal_tables under {args.grid_dir}")

    static_path = args.grid_dir / "static_features.parquet"
    if not static_path.exists():
        raise SystemExit(f"[error] missing {static_path}")
    static = pd.read_parquet(static_path)
    if CELL not in static.columns:
        raise SystemExit(f"[error] {static_path.name} has no {CELL} column; "
                         f"cols={list(static.columns)[:8]}")

    # Feature coverage: base cols must come from temporal+static, derivatives are built.
    tcols = set(pq.read_schema(files[0]).names)
    scols = set(static.columns)
    have = tcols | scols
    derived = {stored_name(f"{b}_{s}") for b in bases
               for s in ("lag1", "roll3_min", "roll3_mean", "roll3_max",
                         "roll7_min", "roll7_mean", "roll7_max")}
    base_needed = [c for c in feats if stored_name(c) not in derived]
    absent = [c for c in base_needed if c not in have and stored_name(c) not in have]

    print(f"[plan] months={len(files)}  cells(static)={len(static):,}")
    print(f"[plan] base features needed={len(base_needed)}  "
          f"derivatives to build={len(derived)}  total={len(base_needed) + len(derived)}")
    print(f"[plan] absent base features: {len(absent)}")
    if absent:
        print(f"        {absent[:10]}")
        raise SystemExit("[error] cannot assemble 651 features; see absent list above")
    if len(base_needed) + len(derived) != len(feats):
        print(f"[warn] {len(base_needed)}+{len(derived)} != {len(feats)}")

    if args.dry_run:
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"[dry-run] would predict {rows:,} cell-days in batches of "
              f"{args.cells_per_batch} cells")
        return 0

    if not (args.model_dir / "stage1_model_bundle.pkl").exists():
        raise SystemExit(f"[error] no final model at {args.model_dir}. "
                         f"Run train_final_ontario_model.py first.")
    model = TC.load_model(args.model_dir)

    t0 = time.time()
    keep = [CELL, DATE] + [c for c in base_needed if c in tcols]

    # Do NOT read all 144 months at once. The full base panel is 46.7M x 140; pandas
    # widens to float64 and pd.concat copies, which reached 130 GB and was OOM-killed
    # on a 125 GB box. Instead take the cell list from the cheapest possible read,
    # then pull only the rows for each cell batch out of each monthly file.
    logger.info("Scanning %d monthly tables for the cell list...", len(files))
    cells = set()
    for f in files:
        cells.update(pd.read_parquet(f, columns=[CELL])[CELL].unique().tolist())
    cells = np.array(sorted(cells))
    logger.info("cells=%d across %d months", len(cells), len(files))
    print(f"[load] {len(cells):,} cells across {len(files)} monthly tables", flush=True)

    file_cols = {f: [c for c in keep if c in set(pq.read_schema(f).names)] for f in files}

    # static_features is keyed by (cell, YEAR) -- land cover is time-varying, 12 rows
    # per cell. Joining on cell alone is a cartesian expansion that silently multiplies
    # every cell-day by 12 (and was the real cause of the earlier OOM). Join on both.
    static_key = [CELL, "year"] if "year" in static.columns else [CELL]
    dup = int(static.duplicated(subset=static_key).sum())
    if dup:
        raise SystemExit(f"[error] static_features has {dup} duplicate rows on "
                         f"{static_key}; the join would multiply cell-days")
    logger.info("static join key: %s (%d rows, %d cells)", static_key, len(static),
                static[CELL].nunique())
    static_idx = static.set_index(static_key)
    stored_feats = [stored_name(c) for c in feats]
    out_parts = []

    for bi in range(0, len(cells), args.cells_per_batch):
        batch = cells[bi:bi + args.cells_per_batch]
        want = set(batch.tolist())
        # One cell's full 2012-2023 series must stay together (roll7 looks back 7
        # days), so gather this batch's rows from every month before deriving.
        sub = pd.concat(
            [d for d in (pd.read_parquet(f, columns=file_cols[f],
                                         filters=[(CELL, "in", want)])
                         for f in files) if len(d)],
            ignore_index=True)
        if sub.empty:
            continue
        sub[DATE] = pd.to_datetime(sub[DATE])
        # A cell's full series must stay together: roll7 looks back seven days.
        sub = build_derivatives(sub, bases)
        n_before = len(sub)
        if "year" not in sub.columns:
            sub["year"] = sub[DATE].dt.year
        sub = sub.join(static_idx, on=static_key, rsuffix="_static")
        if len(sub) != n_before:
            raise SystemExit(f"[error] static join changed row count "
                             f"{n_before} -> {len(sub)}; key {static_key} is not unique")

        missing = [c for c in stored_feats if c not in sub.columns]
        if missing:
            raise SystemExit(f"[error] batch missing {len(missing)} features: {missing[:5]}")
        X = sub[stored_feats].to_numpy("float32")
        np.nan_to_num(X, copy=False, nan=0.0)          # spec-H fill, as in training

        p1 = model.stage1.predict(X).astype("float64")
        d = sub.copy()
        d["pred_stage1"] = p1
        Xc, cols = TC.F.transform_corrector(d, model.fill_values)
        if cols != model.corrector_cols:
            raise AssertionError("corrector feature order differs from training")
        pf = p1 + model.corrector.predict(Xc).astype("float64")

        out_parts.append(pd.DataFrame({
            "CanOSSEM_RASTER_CELL": sub[CELL].astype(str),
            "date": sub[DATE], "year": sub[DATE].dt.year.astype("int16"),
            "pred_stage1": p1.astype("float32"),
            "pred_final": pf.astype("float32"),
        }))
        done = min(bi + args.cells_per_batch, len(cells))
        print(f"  cells {done:>6,}/{len(cells):,}  rows so far "
              f"{sum(len(p) for p in out_parts):>10,}  {time.time()-t0:.0f}s", flush=True)

    out = pd.concat(out_parts, ignore_index=True)
    del out_parts
    path = args.out_dir / "grid_predictions_lgbm.parquet"
    out.to_parquet(path, index=False)

    annual = (out.groupby(["year", "CanOSSEM_RASTER_CELL"])["pred_final"]
              .mean().reset_index())
    annual.to_parquet(args.out_dir / "grid_surface_annual_lgbm.parquet", index=False)

    TC.save_json({
        "rows": int(len(out)), "cells": int(out["CanOSSEM_RASTER_CELL"].nunique()),
        "years": [int(out["year"].min()), int(out["year"].max())],
        "model_dir": str(args.model_dir), "grid_dir": str(args.grid_dir),
        "pred_final": {"mean": float(out.pred_final.mean()),
                       "min": float(out.pred_final.min()),
                       "max": float(out.pred_final.max())},
        "clipping": "none (CV convention); clip at 0 only for a released product",
        "elapsed_seconds": round(time.time() - t0, 1),
    }, args.out_dir / "raster_prediction_summary.json")

    print(f"\n[done] {len(out):,} cell-days, {out['CanOSSEM_RASTER_CELL'].nunique():,} cells "
          f"in {(time.time()-t0)/60:.1f} min -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
