#!/usr/bin/env python3
"""
Run ONE outer fold with the corrector trained on OUT-OF-FOLD Stage-1 residuals.

Everything except the corrector's training target is identical to
run_lgbm_thesis_fold.py, whose functions this script imports rather than copies:
same loaders, same 651-predictor resolution, same NaN policy, same pool rule, same
40 -> 80 corrector encoding, same hyperparameters, same seed.

Why
---
The thesis corrector's target is `obs - stage1.predict(X_train)` where that Stage 1
was trained on X_train. Those residuals are optimistically small -- measured on fold
1, mean |residual| is 0.60 in-sample against 1.44 on held-out rows. The corrector is
therefore calibrated to a problem roughly half as hard as the one it meets at
prediction time. This script asks what happens when it is shown honest residuals.

Design
------
Two levels of holding out. The OUTER test blocks are predicted and never fitted on --
not by the corrector, not by any inner model. The honest residuals come from an INNER
split of the training blocks only:

    72 blocks
    |-- 9   OUTER TEST      touched only at scoring time
    +-- 63  training material
        |-- inner group h (h != g) is fold h's own held-out set, so the inner
        |   partition already exists in case_plans/ and needs no new file
        |-- inner model h: fit on 63 - 9 = 54 blocks, predict h's 9
        |   after 7 fits every training row has a Stage-1 prediction from a
        |   model that did not see it
        |-- CORRECTOR fits here, on the 63 blocks, target = obs - oof prediction
        +-- OUTER Stage 1 fits on all 63, and predicts the 9 test blocks

    pred_final = outer_stage1(test) + corrector(test)

Cost: 7 inner + 1 outer = 8 Stage-1 fits per fold, against 1 for the thesis design.

    python run_lgbm_oof_corrector_fold.py --fold 1 --shard-root Data \
        --case-plans-dir case_plans --out-root outputs/robustness/oof_corrector

Outputs mirror run_lgbm_thesis_fold.py, plus oof_stage1_train.parquet (the honest
training-row predictions) so a later corrector question can be answered by
refit_correctors.py without repeating the 7 inner fits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor
import sklearn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("F", HERE / "run_lgbm_thesis_fold.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)          # the production fold script; single source of truth

N_GROUPS = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="One outer fold, corrector trained on out-of-fold Stage-1 residuals.")
    ap.add_argument("--fold", type=int, required=True, choices=range(1, N_GROUPS + 1),
                    metavar="{1..8}")
    ap.add_argument("--shard-root", type=Path, default=HERE / "Data")
    ap.add_argument("--case-plans-dir", type=Path, default=None,
                    help="defaults to <shard-root>/case_plans")
    ap.add_argument("--out-root", type=Path,
                    default=HERE / "outputs" / "robustness" / "oof_corrector")
    ap.add_argument("--features-file", type=Path, default=None,
                    help="optional Stage-1 predictor subset, as in the thesis script")
    ap.add_argument("--threads", type=int, default=None,
                    help="estimator thread count. Set this when several folds run "
                         "concurrently: the default n_jobs=-1 makes every process "
                         "claim all cores, and LightGBM's n_jobs overrides "
                         "OMP_NUM_THREADS, so exporting that is not enough.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Both param dicts are reused as-is for the 7 inner fits and the outer fit.
    if args.threads:
        F.STAGE1_PARAMS["n_jobs"] = int(args.threads)
        F.CORRECTOR_PARAMS["n_jobs"] = int(args.threads)
    return args


def resolve_features(shard_root: Path, features_file: Path | None, logger) -> list[str]:
    """The manifest's 651, or a declared subset of them. Mirrors the thesis script."""
    manifest = F.read_json(shard_root / "manifest.json")
    feats = list(manifest.get("feature_cols", []))
    if len(feats) != 651 or len(set(feats)) != 651:
        raise ValueError(f"manifest has {len(feats)} Stage-1 features; expected 651 unique")
    if features_file is None:
        return feats
    raw = features_file.read_text(encoding="utf-8").strip()
    req = json.loads(raw) if raw.startswith("[") else [l.strip() for l in raw.splitlines() if l.strip()]
    unknown = [c for c in req if c not in set(feats)]
    if unknown:
        raise ValueError(f"{features_file}: {len(unknown)} name(s) not in the manifest: {unknown[:10]}")
    sub = [c for c in feats if c in set(req)]
    logger.info("FEATURE SUBSET | %s | %d of %d", features_file.name, len(sub), len(feats))
    return sub


def inner_partition(case_plans_dir: Path, fold_no: int, train_blocks: list[str]) -> dict[int, list[str]]:
    """The inner split already exists: fold g's 63 training blocks are exactly the
    union of the other seven folds' held-out sets, so each inner group is one of those.

    Asserted rather than assumed -- if the plans ever stop partitioning cleanly this
    fails here instead of silently leaving training rows without an OOF prediction.
    """
    train = set(train_blocks)
    groups: dict[int, list[str]] = {}
    covered: set[str] = set()
    for h in range(1, N_GROUPS + 1):
        if h == fold_no:
            continue
        plan, _ = F.load_fold_plan(case_plans_dir, h)
        blocks = [b for b in plan["heldout_case_keys"] if b in train]
        if not blocks:
            raise ValueError(f"inner group {h} contributes no training block")
        if covered & set(blocks):
            raise ValueError(f"inner group {h} overlaps an earlier inner group")
        groups[h] = blocks
        covered |= set(blocks)
    if covered != train:
        missing = sorted(train - covered)
        raise ValueError(f"inner partition misses {len(missing)} training block(s): {missing[:5]}")
    return groups


def main() -> int:
    args = parse_args()
    case_plans_dir = args.case_plans_dir or (args.shard_root / "case_plans")
    fold_out = args.out_root / f"GROUP_{args.fold:02d}"
    fold_out.mkdir(parents=True, exist_ok=True)
    logger = F.configure_logger(fold_out / "training.log", args.verbose)
    start = time.time()

    try:
        logger.info("=" * 100)
        logger.info("OOF-CORRECTOR OUTER FOLD START | fold=%d", args.fold)
        logger.info("shard_root=%s", args.shard_root)
        logger.info("=" * 100)

        # ---- plan: the outer split -------------------------------------------
        plan, plan_path = F.load_fold_plan(case_plans_dir, args.fold)
        train_blocks = list(plan["train_pair_blocks"])
        test_blocks = list(plan["heldout_case_keys"])
        logger.info("PLAN | %s | train=%d test=%d", plan_path.name,
                    len(train_blocks), len(test_blocks))

        feats = resolve_features(args.shard_root, args.features_file, logger)

        # ---- the inner split, derived from the same plans ---------------------
        inner = inner_partition(case_plans_dir, args.fold, train_blocks)
        logger.info("INNER PARTITION | %d groups | sizes=%s",
                    len(inner), [len(v) for v in inner.values()])

        # ---- load once --------------------------------------------------------
        logger.info("Loading %d training blocks...", len(train_blocks))
        train_df, X_train, _ = F.load_blocks(args.shard_root, train_blocks, feats, logger)
        logger.info("Loading %d held-out blocks...", len(test_blocks))
        test_df, X_test, _ = F.load_blocks(args.shard_root, test_blocks, feats, logger)

        y_train = pd.to_numeric(train_df["pm25"], errors="raise").to_numpy("float64")
        y_test = pd.to_numeric(test_df["pm25"], errors="raise").to_numpy("float64")
        block_of = train_df["_source_block"].astype(str).to_numpy()

        # ---- 1. inner fits -> out-of-fold Stage-1 predictions -----------------
        oof = np.full(len(train_df), np.nan, dtype="float64")
        for h, blocks in sorted(inner.items()):
            m = block_of == blocks[0] if len(blocks) == 1 else np.isin(block_of, blocks)
            t0 = time.time()
            inner_model = LGBMRegressor(**F.STAGE1_PARAMS).fit(X_train[~m], y_train[~m])
            oof[m] = inner_model.predict(X_train[m]).astype("float64")
            logger.info("  inner %d | fit on %d rows, scored %d | %.0fs",
                        h, int((~m).sum()), int(m.sum()), time.time() - t0)

        if np.isnan(oof).any():
            raise AssertionError(
                f"{int(np.isnan(oof).sum())} training rows received no out-of-fold "
                f"prediction; the corrector target would be NaN")
        resid_oof = y_train - oof
        logger.info("OOF RESIDUALS | mean|r|=%.4f sd=%.4f  (in-sample would be smaller)",
                    float(np.abs(resid_oof).mean()), float(resid_oof.std()))

        # ---- 2. outer Stage 1: the model that actually predicts the test set ---
        logger.info("Fitting OUTER Stage-1 on all %d training blocks...", len(train_blocks))
        stage1 = LGBMRegressor(**F.STAGE1_PARAMS).fit(X_train, y_train, feature_name=list(feats))
        pred_stage1_test = stage1.predict(X_test).astype("float64")
        stage1_metrics = F.metrics(y_test, pred_stage1_test)
        logger.info("STAGE1 HOLDOUT | %s", json.dumps(stage1_metrics))

        # ---- 3. corrector on the honest residuals -----------------------------
        # Pool rule, fill values and encoding are the thesis script's; only the
        # residual definition differs.
        train_df = train_df.copy()
        train_df["pred_stage1"] = oof
        train_df["resid_stage1"] = resid_oof
        test_df = test_df.copy()
        test_df["pred_stage1"] = pred_stage1_test

        F.validate_corrector_columns(train_df)
        F.validate_corrector_columns(test_df)
        pool_df, _mask, pool_diag = F.build_corrector_pool(train_df, logger)
        fill_values = F.fit_corrector_fill_values(pool_df)
        X_corr_train, corr_cols = F.transform_corrector(pool_df, fill_values)
        X_corr_test, corr_cols_test = F.transform_corrector(test_df, fill_values)
        if corr_cols != corr_cols_test:
            raise AssertionError("corrector train/test feature order differs")

        y_corr = pd.to_numeric(pool_df["resid_stage1"], errors="raise").to_numpy("float64")
        logger.info("Fitting corrector on %d rows x %d inputs (OOF residual target)...",
                    len(pool_df), X_corr_train.shape[1])
        corrector = LGBMRegressor(**F.CORRECTOR_PARAMS).fit(
            X_corr_train, y_corr, feature_name=corr_cols)

        pred_corrector_test = corrector.predict(X_corr_test).astype("float64")
        pred_final_test = pred_stage1_test + pred_corrector_test
        final_metrics = F.metrics(y_test, pred_final_test)
        logger.info("FINAL HOLDOUT | %s", json.dumps(final_metrics))
        logger.info("CORRECTOR DELTA | r2 %+.4f | rmse %+.4f",
                    final_metrics["r2_predictive"] - stage1_metrics["r2_predictive"],
                    final_metrics["rmse"] - stage1_metrics["rmse"])

        # ---- outputs ----------------------------------------------------------
        predictions = pd.DataFrame({
            "grid_cell_id": test_df["CanOSSEM_RASTER_CELL"].astype(str),
            "date": pd.to_datetime(test_df["date"]),
            "year": pd.to_numeric(test_df["year"], errors="raise").astype(int),
            "region": test_df["fold_region"].astype(str),
            "naps_id": test_df["naps_id"].astype(str),
            "station_name": test_df["station_name"].astype(str),
            "case_key": test_df["_source_block"].astype(str),
            "outer_fold": int(args.fold),
            "fold_label": f"GROUP_{args.fold:02d}",
            "model_family": "lgbm",
            "obs_pm25": y_test,
            "pred_stage1": pred_stage1_test,
            "pred_corrector": pred_corrector_test,
            "pred_final": pred_final_test,
            "resid_stage1": y_test - pred_stage1_test,
            "resid_final": y_test - pred_final_test,
        })
        if int(predictions.duplicated(["grid_cell_id", "date"]).sum()):
            raise ValueError("duplicate cell-days in the holdout predictions")
        if set(predictions["case_key"]) != set(test_blocks):
            raise AssertionError("prediction table does not cover exactly the planned blocks")
        predictions.to_parquet(fold_out / "holdout_predictions.parquet", index=False)
        predictions.to_csv(fold_out / "holdout_predictions.csv", index=False)

        # the honest training-row predictions, so a corrector-only question later
        # does not have to repeat the 7 inner fits
        pd.DataFrame({
            "grid_cell_id": train_df["CanOSSEM_RASTER_CELL"].astype(str),
            "date": pd.to_datetime(train_df["date"]),
            "outer_fold": int(args.fold),
            "pred_stage1": oof,
        }).to_parquet(fold_out / "oof_stage1_train.parquet", index=False)

        rows = []
        for block, g in predictions.groupby("case_key", sort=True):
            m1 = F.metrics(g["obs_pm25"].to_numpy(), g["pred_stage1"].to_numpy())
            mf = F.metrics(g["obs_pm25"].to_numpy(), g["pred_final"].to_numpy())
            rows.append({"case_key": block,
                         **{f"stage1_{k}": v for k, v in m1.items()},
                         **{f"final_{k}": v for k, v in mf.items()}})
        pd.DataFrame(rows).to_csv(fold_out / "holdout_block_metrics.csv", index=False)

        F.save_pickle({"model": stage1, "stage": "stage1_outer", "feature_cols": feats,
                       "fold_no": int(args.fold), "train_blocks": train_blocks,
                       "holdout_blocks": test_blocks, "params": F.STAGE1_PARAMS},
                      fold_out / "stage1_model.pkl")
        F.save_pickle({"model": corrector, "stage": "corrector_oof",
                       "raw_feature_cols": F.CORRECTOR_RAW_COLS,
                       "corrector_feature_cols": corr_cols, "fill_values": fill_values,
                       "pool_diagnostics": pool_diag, "fold_no": int(args.fold),
                       "params": F.CORRECTOR_PARAMS},
                      fold_out / "corrector_model.pkl")
        stage1.booster_.save_model(str(fold_out / "stage1_model.txt"))
        corrector.booster_.save_model(str(fold_out / "corrector_model.txt"))

        F.save_json({
            "fold_no": int(args.fold), "fold_label": f"GROUP_{args.fold:02d}",
            "experiment": "oof_corrector",
            "corrector_target": "observed_pm25 - OUT-OF-FOLD stage1 prediction",
            "inner_partition": {str(h): b for h, b in sorted(inner.items())},
            "inner_fits": len(inner), "outer_fits": 1,
            "train_rows": int(len(train_df)), "holdout_rows": int(len(test_df)),
            "corrector_train_rows": int(len(pool_df)),
            "oof_residual_mean_abs": float(np.abs(resid_oof).mean()),
            "oof_residual_sd": float(resid_oof.std()),
            "stage1_holdout": stage1_metrics, "final_holdout": final_metrics,
            "stage1_feature_count": len(feats),
            "stage1_feature_source": str(args.features_file or (args.shard_root / "manifest.json")),
            "cv_prediction_clipping": "none", "seed": F.SEED,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                         "scikit_learn": sklearn.__version__},
            "stage1_params": F.STAGE1_PARAMS, "corrector_params": F.CORRECTOR_PARAMS,
            "elapsed_seconds": float(time.time() - start),
        }, fold_out / "run_manifest.json")
        F.save_json({"stage1_holdout": stage1_metrics, "final_holdout": final_metrics,
                     "corrector_pool": pool_diag}, fold_out / "metrics.json")

        logger.info("=" * 100)
        logger.info("FOLD %02d COMPLETE | %d inner + 1 outer fit | elapsed=%.1fs",
                    args.fold, len(inner), time.time() - start)
        logger.info("=" * 100)
        return 0

    except Exception:
        logger.exception("OOF-CORRECTOR FOLD %02d FAILED", args.fold)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
