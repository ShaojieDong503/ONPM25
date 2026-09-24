#!/usr/bin/env python3
"""
External test: the final Ontario model applied to monitored QUEBEC cells.

Quebec never appears in training, so this measures transfer to a province the model
has never seen -- a stronger test than the region-year block CV, which always has
the same grid cells in training under a different year.

Data provenance
---------------
The QC rows already exist with all 651 features, built by the production shard
builder, in the SUPPORT_SHARED blocks of
`materialized_support_family_shards_pruned_temporal_x`. Nothing is reconstructed
here. Province is identified by NAPS id prefix (05 = QC), which reproduces the
counts in the earlier external-province validation exactly.

    python quebec_external_test.py --clip          # clip QC into Data_external/
    python quebec_external_test.py                 # score it
    python quebec_external_test.py --province MB   # any support province

Scoring uses `thesis_core`, so the loaders, corrector transform and metrics are the
ones from `run_lgbm_thesis_fold.py`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import thesis_core as TC  # noqa: E402

DEFAULT_SUPPORT_ROOT = Path(
    r"D:\lambda\Ontario_RealTarget_GPD\outputs"
    r"\materialized_support_family_shards_pruned_temporal_x")

#: NAPS identifiers encode the province in their first two digits.
NAPS_PREFIX = {"QC": "05", "MB": "07", "SK": "08", "AB": "09", "BC": "10", "ON": "06"}


def clip_province(support_root: Path, province: str, out_root: Path,
                  shard_root: Path, rows_per_block: int = 50_000) -> Path:
    """Copy one province's rows out of the SUPPORT_SHARED blocks into a shard root.

    Written in the same layout the loaders expect (`pair_blocks/<name>/frame.parquet`
    + meta.json + manifest.json), so `load_blocks` reads it unchanged.
    """
    prefix = NAPS_PREFIX[province]
    feats = TC.canonical_features(shard_root)

    src = sorted((support_root / "pair_blocks").glob("SUPPORT_SHARED_*/frame.parquet"))
    if not src:
        raise SystemExit(f"[error] no SUPPORT_SHARED blocks under {support_root}")
    print(f"[clip] scanning {len(src)} support blocks for province {province} "
          f"(naps prefix {prefix})")

    parts = []
    for f in src:
        df = pd.read_parquet(f)
        keep = df["naps_id"].astype(str).str.startswith(prefix)
        if keep.any():
            parts.append(df.loc[keep])
    if not parts:
        raise SystemExit(f"[error] no rows with naps prefix {prefix}")

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["CanOSSEM_RASTER_CELL", "date"]).reset_index(drop=True)
    print(f"[clip] {len(df):,} rows, {df['CanOSSEM_RASTER_CELL'].nunique()} cells, "
          f"{df['naps_id'].nunique()} stations, "
          f"{df['date'].dt.year.min()}-{df['date'].dt.year.max()}")

    # Every canonical feature must resolve, or scoring would silently zero-fill.
    stored = {c: (c.replace("_roll3_", "_lag1_roll3_").replace("_roll7_", "_lag1_roll7_")
                  if ("_roll3_" in c or "_roll7_" in c) else c) for c in feats}
    have = set(df.columns)
    absent = [c for c, s in stored.items() if c not in have and s not in have]
    if absent:
        raise SystemExit(f"[error] {len(absent)} canonical feature(s) absent from the "
                         f"support blocks: {absent[:5]}")
    print(f"[clip] feature check: {sum(c in have for c in feats)} direct, "
          f"{sum(c not in have and s in have for c, s in stored.items())} translated, "
          f"0 absent")

    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "pair_blocks").mkdir(parents=True)

    names = []
    for i in range(0, len(df), rows_per_block):
        chunk = df.iloc[i:i + rows_per_block].reset_index(drop=True)
        name = f"{province}_BLOCK_{i // rows_per_block + 1:03d}"
        bdir = out_root / "pair_blocks" / name
        bdir.mkdir(parents=True)
        chunk.to_parquet(bdir / "frame.parquet", index=False)
        (bdir / "meta.json").write_text(json.dumps({
            "block_name": name, "rows": int(len(chunk)),
            "feature_count": len(feats), "feature_cols": feats,
            "province": province,
        }, indent=2), encoding="utf-8")
        names.append(name)

    shutil.copy2(shard_root / "manifest.json", out_root / "manifest.json")
    (out_root / "blocks.json").write_text(json.dumps({
        "province": province, "blocks": names, "rows": int(len(df)),
        "cells": int(df["CanOSSEM_RASTER_CELL"].nunique()),
        "stations": int(df["naps_id"].nunique()),
        "source": str(support_root),
        "note": "clipped from SUPPORT_SHARED blocks; features built by the production "
                "shard builder, not reconstructed",
    }, indent=2), encoding="utf-8")

    print(f"[clip] wrote {len(names)} block(s) -> {out_root}")
    return out_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--province", default="QC", choices=sorted(NAPS_PREFIX))
    ap.add_argument("--clip", action="store_true",
                    help="build the province shard root from SUPPORT_SHARED first")
    ap.add_argument("--clip-only", action="store_true",
                    help="clip and stop; do not score (no model needed)")
    ap.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT_ROOT)
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--external-root", type=Path, default=None,
                    help="where the clipped province lives (default Data_external/<prov>)")
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "outputs" / "final_ontario_model")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prov = args.province
    ext_root = args.external_root or (ROOT / "Data_external" / prov.lower())
    out_dir = args.out_dir or (ROOT / "outputs" / f"external_{prov.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = TC.configure_logger(out_dir / "external_test.log", args.verbose)

    if args.clip or args.clip_only:
        clip_province(args.support_root, prov, ext_root, args.shard_root)
    if args.clip_only:
        print(f"[clip-only] done -> {ext_root}")
        return 0

    if not (ext_root / "manifest.json").exists():
        raise SystemExit(f"[error] no clipped data at {ext_root}. Run with --clip first.")
    if not (args.model_dir / "stage1_model_bundle.pkl").exists():
        raise SystemExit(f"[error] no final model at {args.model_dir}. "
                         f"Run train_final_ontario_model.py first.")

    t0 = time.time()
    model = TC.load_model(args.model_dir)
    blocks = json.loads((ext_root / "blocks.json").read_text(encoding="utf-8"))["blocks"]
    logger.info("EXTERNAL TEST | province=%s | blocks=%d | model=%s",
                prov, len(blocks), args.model_dir)

    df, X, _ = TC.load_blocks(ext_root, blocks, model.feature_cols, logger)
    y = pd.to_numeric(df["pm25"], errors="raise").to_numpy("float64")
    if not np.isfinite(y).all():
        raise ValueError("external PM2.5 contains non-finite values")

    p1, pc, pf = TC.score(model, X, df, logger)
    if not np.allclose(pf, p1 + pc, atol=1e-12, rtol=0):
        raise AssertionError("pred_final != pred_stage1 + pred_corrector")

    preds = pd.DataFrame({
        "grid_cell_id": df["CanOSSEM_RASTER_CELL"].astype(str),
        "date": pd.to_datetime(df["date"]),
        "year": pd.to_numeric(df["year"], errors="raise").astype(int),
        "naps_id": df["naps_id"].astype(str),
        "province": prov, "model_family": "lgbm",
        "obs_pm25": y, "pred_stage1": p1,
        "pred_corrector": pc, "pred_final": pf,
    })
    dup = int(preds.duplicated(["grid_cell_id", "date"]).sum())
    if dup:
        raise ValueError(f"{dup} duplicate cell-days in the external predictions")
    preds.to_parquet(out_dir / f"{prov.lower()}_predictions.parquet", index=False)

    # Overall, per-year and per-cell, plus the spatial/temporal decomposition.
    def spatial_temporal(d: pd.DataFrame) -> dict:
        cm = d.groupby("grid_cell_id")[["obs_pm25", "pred_final"]].mean()
        oa = d["obs_pm25"] - d.groupby("grid_cell_id")["obs_pm25"].transform("mean")
        pa = d["pred_final"] - d.groupby("grid_cell_id")["pred_final"].transform("mean")
        r2 = lambda a, b: float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        return {"spatial_r2": r2(cm["obs_pm25"], cm["pred_final"]),
                "temporal_r2": r2(oa, pa)}

    summary = {
        "province": prov, "rows": int(len(preds)),
        "cells": int(preds["grid_cell_id"].nunique()),
        "stations": int(preds["naps_id"].nunique()),
        "years": [int(preds["year"].min()), int(preds["year"].max())],
        "model_dir": str(args.model_dir),
        "stage1": TC.metrics(y, p1), "final": TC.metrics(y, pf),
        **spatial_temporal(preds),
        "by_year": {int(yr): TC.metrics(g["obs_pm25"].to_numpy(), g["pred_final"].to_numpy())
                    for yr, g in preds.groupby("year")},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    TC.save_json(summary, out_dir / f"{prov.lower()}_metrics.json")

    per_cell = (preds.groupby("grid_cell_id")
                .apply(lambda g: pd.Series(TC.metrics(g["obs_pm25"].to_numpy(),
                                                      g["pred_final"].to_numpy())),
                       include_groups=False)
                .reset_index())
    per_cell.to_csv(out_dir / f"{prov.lower()}_per_cell_metrics.csv", index=False)

    logger.info("FINAL | %s", json.dumps(summary["final"]))
    print(f"\n[{prov}] {len(preds):,} rows, {summary['cells']} cells, "
          f"{summary['years'][0]}-{summary['years'][1]}")
    # TC.metrics names it r2_predictive (1 - SSE/SST), deliberately not squared
    # correlation -- on an external province those two diverge a lot, because a
    # biased-but-correlated prediction scores well on one and badly on the other.
    print(f"  stage1 : rmse={summary['stage1']['rmse']:.3f} "
          f"r2={summary['stage1']['r2_predictive']:.4f}")
    print(f"  final  : rmse={summary['final']['rmse']:.3f} "
          f"r2={summary['final']['r2_predictive']:.4f} "
          f"mae={summary['final']['mae']:.3f}")
    print(f"  spatial_r2={summary['spatial_r2']:.4f}  temporal_r2={summary['temporal_r2']:.4f}")
    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
