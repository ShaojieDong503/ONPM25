#!/usr/bin/env python3
"""
Train the FINAL two-stage LightGBM on all available Ontario data.

No holdout: every one of the 72 region-year blocks goes into training. This is the
model used for raster-cell prediction and for external-province testing, not for
reporting cross-validated skill -- it has no out-of-sample rows by construction.

Identical to one thesis fold except for the data: same `run_lgbm_thesis_fold.py`
loaders, hyperparameters, corrector pool rule, fill values and seed 2026.

    python train_final_ontario_model.py
    python train_final_ontario_model.py --holdout-check   # also report in-sample fit

Outputs to <out-dir>:
    stage1_model_bundle.pkl   corrector_model.pkl
    run_manifest.json         training.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import thesis_core as TC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "final_ontario_model")
    ap.add_argument("--holdout-check", action="store_true",
                    help="also score the training rows, to confirm the model fitted "
                         "(in-sample only -- never report this as skill)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TC.configure_logger(args.out_dir / "training.log", args.verbose)
    t0 = time.time()

    logger.info("=" * 100)
    logger.info("FINAL ONTARIO MODEL (all blocks, no holdout)")
    logger.info("shard_root=%s", args.shard_root)
    logger.info("=" * 100)

    feats = TC.canonical_features(args.shard_root)
    blocks = sorted(p.name for p in (args.shard_root / "pair_blocks").iterdir()
                    if p.is_dir() and (p / "frame.parquet").exists())
    logger.info("PLAN | all %d blocks -> training, 0 held out", len(blocks))
    if len(blocks) != 72:
        logger.warning("expected 72 Ontario blocks, found %d", len(blocks))

    model = TC.fit_and_score(shard_root=args.shard_root, train_blocks=blocks,
                             test_blocks=None, feature_cols=feats, logger=logger)

    in_sample = {}
    if args.holdout_check:
        logger.info("Scoring the training rows (IN-SAMPLE, not skill)...")
        train_df, X, _ = TC.load_blocks(args.shard_root, blocks, feats, logger)
        y = pd.to_numeric(train_df["pm25"], errors="raise").to_numpy("float64")
        p1, pc, pf = TC.score(model, X, train_df, logger)
        in_sample = {"stage1": TC.metrics(y, p1), "final": TC.metrics(y, pf)}
        logger.info("IN-SAMPLE (optimistic) | %s", json.dumps(in_sample["final"]))

    TC.save_model(model, args.out_dir, extra={
        "scope": "final_ontario_all_blocks", "n_blocks": len(blocks),
        "shard_root": str(args.shard_root), "seed": TC.STAGE1_PARAMS["random_state"],
    })

    elapsed = time.time() - t0
    TC.save_json({
        "status": "complete",
        "scope": "final_ontario_all_blocks",
        "shard_root": str(args.shard_root),
        "n_blocks": len(blocks), "blocks": blocks,
        "train_rows": model.train_rows,
        "stage1_feature_count": len(feats),
        "corrector_raw_feature_count": len(TC.CORRECTOR_RAW_COLS),
        "corrector_final_feature_count": len(model.corrector_cols),
        "corrector_pool": model.pool_diag,
        "stage1_params": TC.STAGE1_PARAMS,
        "corrector_params": TC.CORRECTOR_PARAMS,
        "in_sample_metrics": in_sample,
        "note": "No holdout. Do not report in-sample metrics as model skill.",
        "elapsed_seconds": round(elapsed, 1),
    }, args.out_dir / "run_manifest.json")

    logger.info("DONE in %.1f min -> %s", elapsed / 60, args.out_dir)
    print(f"[final-model] {model.train_rows:,} training rows, "
          f"{len(feats)} features -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
