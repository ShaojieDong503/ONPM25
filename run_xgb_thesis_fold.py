#!/usr/bin/env python3
"""
Run ONE thesis XGBoost outer fold for the Ontario PM2.5 model.

Expected layout
---------------
<shard_root>/
  manifest.json
  pair_blocks/
    PAIR_<year>_<region>/
      frame.parquet
  case_plans/
    GROUP_01.json
    ...
    GROUP_08.json

Each GROUP_0N.json is the thesis Table A5 mapping:
- 63 training region-year blocks
- 9 held-out region-year blocks
- support_blocks = []

Outputs for one fold
--------------------
<out_root>/GROUP_0N/
  stage1_model.pkl
  corrector_model.pkl
  stage1_model.json
  corrector_model.json
  holdout_predictions.parquet
  holdout_predictions.csv
  metrics.json
  holdout_block_metrics.csv
  corrector_pool_diagnostics.json
  run_manifest.json
  training.log

Model behavior
--------------
Stage 1:
- XGBoost regression
- 651 canonical predictors from manifest.json
- repaired rolling-name resolution
- NaN -> 0 before fitting/scoring
- fixed thesis hyperparameters
- seed 2026
- no early stopping
- no validation-set tuning

Corrector:
- target = observed PM2.5 - in-sample Stage-1 prediction on TRAINING rows
- training pool keeps rows satisfying:
      smoke/fire signal
      OR observed PM2.5 >= 15
      OR abs(Stage-1 residual) >= 3
- 40 raw corrector inputs
- each raw input gets a missingness flag -> 80 corrector features
- same XGBoost hyperparameters as Stage 1
- applied to every held-out row

Final prediction:
    pred_final = pred_stage1 + pred_corrector

CV predictions are not clipped/floored at zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Configuration
# =============================================================================

SEED = 2026

DEFAULT_SHARD_ROOT = Path(
    r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle"
    r"\Ontario_RealTarget_GPD\outputs"
    r"\materialized_support_family_shards_pruned_temporal_x_repaired"
)

DEFAULT_OUT_ROOT = Path(
    r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle"
    r"\Ontario_RealTarget_GPD\outputs"
    r"\thesis_xgb_grouped_runs"
)

STAGE1_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

CORRECTOR_PARAMS = dict(STAGE1_PARAMS)

KEY_COLUMNS = [
    "date",
    "year",
    "fold_region",
    "CanOSSEM_RASTER_CELL",
    "pm25",
    "_year_region_pair",
    "naps_id",
    "station_name",
]

RADII4 = [100, 250, 500, 1000]
RADII2 = [500, 1000]


def build_corrector_raw_cols() -> List[str]:
    cols = ["pred_stage1"]

    # VIIRS same-day: count, nearest distance, FRP sum x 4 radii = 12
    for var in ["count", "dist_nearest", "frp_mw_sum"]:
        cols.extend([f"src_viirs_{var}_{r}km" for r in RADII4])

    # VIIRS lag-1: count, nearest distance x 4 radii = 8
    for var in ["count", "dist_nearest"]:
        cols.extend([f"src_viirs_{var}_{r}km_lag1" for r in RADII4])

    # HMS: any + density weighted mean x 4 radii = 8
    cols.extend([f"src_hms_any_{r}km" for r in RADII4])
    cols.extend([f"src_hms_density_weight_wmean_{r}km" for r in RADII4])

    # Burned area: fraction, weighted fraction, nearest distance x 2 radii = 6
    for var in ["frac", "weighted_frac", "nearest_km"]:
        cols.extend([f"burned_{var}_{r}km" for r in RADII2])

    # Satellite AOD = 5
    cols.extend([
        "AOD_055",
        "AOD_047",
        "AOD_055_filled",
        "aod_obs_flag",
        "aod_imputed_flag",
    ])

    assert len(cols) == 40, f"Corrector raw feature count={len(cols)}, expected 40"
    return cols


CORRECTOR_RAW_COLS = build_corrector_raw_cols()


# =============================================================================
# Logging / IO
# =============================================================================

def configure_logger(path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("thesis_xgb_fold")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# Fold plan validation
# =============================================================================

def load_fold_plan(case_plans_dir: Path, fold_no: int) -> Tuple[dict, Path]:
    path = case_plans_dir / f"GROUP_{fold_no:02d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold plan not found: {path}")

    plan = read_json(path)

    train_blocks = list(plan.get("train_pair_blocks", []))
    test_blocks = list(plan.get("heldout_case_keys", []))

    if len(train_blocks) != 63:
        raise ValueError(
            f"{path.name}: expected 63 training blocks, found {len(train_blocks)}"
        )
    if len(test_blocks) != 9:
        raise ValueError(
            f"{path.name}: expected 9 held-out blocks, found {len(test_blocks)}"
        )
    if len(set(train_blocks)) != 63:
        raise ValueError(f"{path.name}: duplicate training block(s)")
    if len(set(test_blocks)) != 9:
        raise ValueError(f"{path.name}: duplicate held-out block(s)")
    if set(train_blocks) & set(test_blocks):
        raise ValueError(f"{path.name}: train/holdout block overlap")
    if len(set(train_blocks) | set(test_blocks)) != 72:
        raise ValueError(
            f"{path.name}: train+holdout do not cover exactly 72 unique blocks"
        )

    support = plan.get("support_blocks", [])
    if support:
        raise ValueError(
            f"{path.name}: support_blocks must be empty for Ontario-only thesis run"
        )

    if int(plan.get("fold_no", fold_no)) != fold_no:
        raise ValueError(
            f"{path.name}: fold_no inside JSON does not match requested fold"
        )

    return plan, path


# =============================================================================
# Stage-1 feature loading
# =============================================================================

def shard_frame_column(canonical: str) -> str:
    """
    Repaired-shard storage-name mapping.

    Example:
      canonical AOD_055_filled_roll3_mean
      stored    AOD_055_filled_lag1_roll3_mean
    """
    if "_roll3_" in canonical or "_roll7_" in canonical:
        return canonical.replace("_roll3_", "_lag1_roll3_").replace(
            "_roll7_", "_lag1_roll7_"
        )
    return canonical


def resolve_stage1_columns(
    df_columns: Sequence[str],
    canonical_features: Sequence[str],
) -> Tuple[List[str], Dict[str, str], dict]:
    available = set(df_columns)

    stored_cols: List[str] = []
    rename_map: Dict[str, str] = {}
    direct = 0
    translated = 0
    absent: List[str] = []

    for canonical in canonical_features:
        if canonical in available:
            stored_cols.append(canonical)
            direct += 1
            continue

        stored = shard_frame_column(canonical)
        if stored in available:
            stored_cols.append(stored)
            rename_map[stored] = canonical
            translated += 1
            continue

        absent.append(canonical)

    if absent:
        raise KeyError(
            f"{len(absent)} canonical features are absent. "
            f"Examples: {absent[:20]}"
        )

    return stored_cols, rename_map, {
        "direct": direct,
        "translated": translated,
        "absent": len(absent),
    }


def pair_block_path(shard_root: Path, block: str) -> Path:
    return shard_root / "pair_blocks" / block / "frame.parquet"


def load_one_block(
    shard_root: Path,
    block: str,
    canonical_features: Sequence[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    path = pair_block_path(shard_root, block)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise KeyError(f"{block}: missing required columns {missing_keys}")

    stored_cols, rename_map, resolution = resolve_stage1_columns(
        df.columns, canonical_features
    )

    xdf = df[stored_cols].rename(columns=rename_map)
    xdf = xdf.reindex(columns=canonical_features)

    if list(xdf.columns) != list(canonical_features):
        raise AssertionError(f"{block}: Stage-1 feature order mismatch")

    X = xdf.to_numpy(dtype=np.float32, copy=True)

    inf_count = int(np.isinf(X).sum())
    if inf_count:
        raise ValueError(f"{block}: {inf_count} infinite Stage-1 values")

    nan_count = int(np.isnan(X).sum())
    if nan_count:
        # Thesis Stage-1 rule: residual missing predictor values -> 0.
        X = np.nan_to_num(X, nan=0.0)

    if not np.isfinite(X).all():
        raise ValueError(f"{block}: non-finite Stage-1 values remain after zero fill")

    logger.info(
        "LOAD | %s | rows=%d | X=%s | direct=%d | translated=%d | "
        "absent=%d | nan_zero_filled=%d",
        block,
        len(df),
        tuple(X.shape),
        resolution["direct"],
        resolution["translated"],
        resolution["absent"],
        nan_count,
    )

    info = {
        "block": block,
        "rows": int(len(df)),
        "direct": int(resolution["direct"]),
        "translated": int(resolution["translated"]),
        "absent": int(resolution["absent"]),
        "nan_zero_filled": nan_count,
    }
    return df, X, info


def load_blocks(
    shard_root: Path,
    blocks: Sequence[str],
    canonical_features: Sequence[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, List[dict]]:
    frames = []
    matrices = []
    infos = []

    for i, block in enumerate(blocks, 1):
        logger.info("Reading %d/%d: %s", i, len(blocks), block)
        df, X, info = load_one_block(
            shard_root, block, canonical_features, logger
        )
        df = df.copy()
        df["_source_block"] = block
        frames.append(df)
        matrices.append(X)
        infos.append(info)

    meta = pd.concat(frames, ignore_index=True)
    Xall = np.vstack(matrices).astype(np.float32, copy=False)

    if len(meta) != Xall.shape[0]:
        raise AssertionError("Metadata / matrix row alignment failure")
    if Xall.shape[1] != len(canonical_features):
        raise AssertionError("Stage-1 matrix width mismatch")

    return meta, Xall, infos


# =============================================================================
# Metrics
# =============================================================================

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    if len(y) != len(pred):
        raise ValueError("Metric arrays have different lengths")

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))  # 1 - SSE/SST
    bias = float(np.mean(pred - y))
    within3 = float(np.mean(np.abs(pred - y) <= 3.0))

    if np.std(y) > 0:
        slope, intercept = np.polyfit(y, pred, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "n": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "r2_predictive": r2,
        "bias_pred_minus_obs": bias,
        "within_3": within3,
        "pred_on_obs_slope": slope,
        "pred_on_obs_intercept": intercept,
    }


# =============================================================================
# Corrector
# =============================================================================

def validate_corrector_columns(df: pd.DataFrame) -> None:
    needed = [c for c in CORRECTOR_RAW_COLS if c != "pred_stage1"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} corrector raw feature(s): {missing[:20]}"
        )


def smoke_signal_columns(df: pd.DataFrame) -> List[str]:
    """
    Reproduce the broad smoke/fire signal rule documented for the historical run:
    any raw same-day/lag fire-smoke-burned feature whose name is in the relevant
    families and whose value is > 0.

    Rolling aggregates are intentionally excluded from the corrector pool signal.

    NOTE:
    This broad historical implementation includes positive nearest-distance
    variables because they match the fire/burned families. Therefore the pool can
    legitimately include nearly all training rows. The script records this.
    """
    candidates = []
    for c in df.columns:
        lc = c.lower()

        family_match = (
            c.startswith("src_hms_")
            or c.startswith("src_viirs_")
            or c.startswith("burned_")
            or "smoke" in lc
            or "fire" in lc
            or "frp" in lc
        )
        if not family_match:
            continue

        # Corrector pool signal should not be based on roll3/roll7 derivatives.
        if "_roll3_" in c or "_roll7_" in c or "_lag1_roll" in c:
            continue

        candidates.append(c)

    return sorted(set(candidates))


def build_corrector_pool(
    train_df: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    signal_cols = smoke_signal_columns(train_df)
    if not signal_cols:
        raise ValueError("No smoke/fire signal columns found")

    signal_values = (
        train_df[signal_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    smoke_mask = (signal_values > 0).any(axis=1)

    y = pd.to_numeric(train_df["pm25"], errors="raise").to_numpy(dtype=np.float64)
    resid = pd.to_numeric(
        train_df["resid_stage1"], errors="raise"
    ).to_numpy(dtype=np.float64)

    high_pm_mask = y >= 15.0
    large_resid_mask = np.abs(resid) >= 3.0

    keep = smoke_mask | high_pm_mask | large_resid_mask

    if not keep.any():
        raise ValueError("Corrector training pool is empty")

    diag = {
        "training_rows": int(len(train_df)),
        "pool_rows": int(keep.sum()),
        "pool_fraction": float(keep.mean()),
        "smoke_signal_rows": int(smoke_mask.sum()),
        "high_pm_ge15_rows": int(high_pm_mask.sum()),
        "abs_stage1_residual_ge3_rows": int(large_resid_mask.sum()),
        "smoke_signal_column_count": int(len(signal_cols)),
        "smoke_signal_columns": signal_cols,
        "pool_rule": (
            "smoke/fire signal OR observed_pm25 >= 15 "
            "OR abs(stage1_training_residual) >= 3"
        ),
    }

    logger.info(
        "CORRECTOR POOL | rows=%d/%d (%.4f) | smoke=%d | high_pm=%d | large_resid=%d",
        diag["pool_rows"],
        diag["training_rows"],
        diag["pool_fraction"],
        diag["smoke_signal_rows"],
        diag["high_pm_ge15_rows"],
        diag["abs_stage1_residual_ge3_rows"],
    )

    return train_df.loc[keep].copy(), keep, diag


def fit_corrector_fill_values(pool_df: pd.DataFrame) -> Dict[str, float]:
    """
    Training-only corrector fill values.

    Distance-like inputs:
      max(finite training-pool value) + 10
      9999 if no finite value exists

    Other inputs:
      training-pool median
      0 if no finite value exists
    """
    fills = {}

    for c in CORRECTOR_RAW_COLS:
        x = pd.to_numeric(pool_df[c], errors="coerce").to_numpy(dtype=np.float64)
        finite = x[np.isfinite(x)]

        if "dist_nearest" in c or "nearest_km" in c:
            fills[c] = float(finite.max() + 10.0) if len(finite) else 9999.0
        else:
            fills[c] = float(np.median(finite)) if len(finite) else 0.0

    return fills


def transform_corrector(
    df: pd.DataFrame,
    fills: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    arrays = []
    names = []

    for c in CORRECTOR_RAW_COLS:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        missing = ~np.isfinite(x)

        filled = x.copy()
        filled[missing] = fills[c]

        arrays.append(filled.astype(np.float32))
        arrays.append(missing.astype(np.float32))
        names.extend([c, f"{c}__isna"])

    X = np.column_stack(arrays).astype(np.float32, copy=False)

    if X.shape[1] != 80:
        raise AssertionError(f"Corrector matrix width={X.shape[1]}, expected 80")
    if not np.isfinite(X).all():
        raise ValueError("Corrector matrix contains non-finite values after fill")

    return X, names


# =============================================================================
# Main run
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one thesis Table-A5 XGBoost fold (Stage 1 + corrector)."
    )
    ap.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 9),
        metavar="{1..8}",
        help="Thesis outer fold number.",
    )
    ap.add_argument(
        "--shard-root",
        type=Path,
        default=DEFAULT_SHARD_ROOT,
        help="Root containing manifest.json, pair_blocks/, case_plans/.",
    )
    ap.add_argument(
        "--case-plans-dir",
        type=Path,
        default=None,
        help="Defaults to <shard-root>/case_plans.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Fold output root.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    case_plans_dir = (
        args.case_plans_dir
        if args.case_plans_dir is not None
        else args.shard_root / "case_plans"
    )
    fold_out = args.out_root / f"GROUP_{args.fold:02d}"
    fold_out.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(fold_out / "training.log", args.verbose)
    start = time.time()

    try:
        logger.info("=" * 100)
        logger.info("THESIS LIGHTGBM OUTER FOLD START")
        logger.info("fold=%d", args.fold)
        logger.info("shard_root=%s", args.shard_root)
        logger.info("case_plans_dir=%s", case_plans_dir)
        logger.info("output=%s", fold_out)
        logger.info("=" * 100)

        # -----------------------------------------------------------------
        # Plan
        # -----------------------------------------------------------------
        plan, plan_path = load_fold_plan(case_plans_dir, args.fold)
        train_blocks = list(plan["train_pair_blocks"])
        test_blocks = list(plan["heldout_case_keys"])

        logger.info(
            "PLAN | %s | train_blocks=%d | holdout_blocks=%d",
            plan_path.name,
            len(train_blocks),
            len(test_blocks),
        )
        logger.info("HOLDOUT BLOCKS: %s", ", ".join(test_blocks))

        # -----------------------------------------------------------------
        # Manifest / canonical Stage-1 predictors
        # -----------------------------------------------------------------
        manifest_path = args.shard_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        manifest = read_json(manifest_path)
        canonical_features = list(manifest.get("feature_cols", []))

        if len(canonical_features) != 651:
            raise ValueError(
                f"Current manifest has {len(canonical_features)} Stage-1 features; expected 651"
            )
        if len(set(canonical_features)) != 651:
            raise ValueError("Manifest feature names are not unique")

        # Fail before model fit if any referenced parquet is missing.
        missing_files = [
            str(pair_block_path(args.shard_root, b))
            for b in train_blocks + test_blocks
            if not pair_block_path(args.shard_root, b).exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} referenced parquet(s) missing. "
                f"Examples: {missing_files[:10]}"
            )

        # -----------------------------------------------------------------
        # Load
        # -----------------------------------------------------------------
        logger.info("Loading 63 training blocks...")
        train_df, X_train, train_load_info = load_blocks(
            args.shard_root,
            train_blocks,
            canonical_features,
            logger,
        )

        logger.info("Loading 9 held-out blocks...")
        test_df, X_test, test_load_info = load_blocks(
            args.shard_root,
            test_blocks,
            canonical_features,
            logger,
        )

        y_train = pd.to_numeric(
            train_df["pm25"], errors="raise"
        ).to_numpy(dtype=np.float64)
        y_test = pd.to_numeric(
            test_df["pm25"], errors="raise"
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(y_train).all():
            raise ValueError("Training PM2.5 contains non-finite values")
        if not np.isfinite(y_test).all():
            raise ValueError("Held-out PM2.5 contains non-finite values")

        logger.info("TRAIN | rows=%d | X=%s", len(train_df), X_train.shape)
        logger.info("HOLDOUT | rows=%d | X=%s", len(test_df), X_test.shape)

        # -----------------------------------------------------------------
        # Stage 1
        # -----------------------------------------------------------------
        logger.info("Fitting Stage-1 XGBoost...")
        stage1 = XGBRegressor(**STAGE1_PARAMS)
        stage1.fit(
            X_train,
            y_train,
        )

        pred_stage1_train = stage1.predict(X_train).astype(np.float64)
        pred_stage1_test = stage1.predict(X_test).astype(np.float64)

        stage1_holdout_metrics = metrics(y_test, pred_stage1_test)
        logger.info(
            "STAGE1 HOLDOUT METRICS | %s",
            json.dumps(stage1_holdout_metrics),
        )

        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df["pred_stage1"] = pred_stage1_train
        train_df["resid_stage1"] = y_train - pred_stage1_train
        test_df["pred_stage1"] = pred_stage1_test

        # -----------------------------------------------------------------
        # Corrector pool
        # -----------------------------------------------------------------
        validate_corrector_columns(train_df)
        validate_corrector_columns(test_df)

        pool_df, pool_mask, pool_diag = build_corrector_pool(
            train_df,
            logger,
        )

        # Corrector target is training residual.
        y_corr = pd.to_numeric(
            pool_df["resid_stage1"], errors="raise"
        ).to_numpy(dtype=np.float64)

        # Fit preprocessing only from corrector training pool.
        fill_values = fit_corrector_fill_values(pool_df)

        X_corr_train, corrector_feature_names = transform_corrector(
            pool_df,
            fill_values,
        )
        X_corr_test, corrector_feature_names_test = transform_corrector(
            test_df,
            fill_values,
        )

        if corrector_feature_names != corrector_feature_names_test:
            raise AssertionError("Corrector train/test feature order differs")

        logger.info(
            "CORRECTOR | train_rows=%d | X_train=%s | X_holdout=%s",
            len(pool_df),
            X_corr_train.shape,
            X_corr_test.shape,
        )

        # -----------------------------------------------------------------
        # Fit corrector
        # -----------------------------------------------------------------
        logger.info("Fitting residual-corrector XGBoost...")
        corrector = XGBRegressor(**CORRECTOR_PARAMS)
        corrector.fit(
            X_corr_train,
            y_corr,
        )

        pred_corrector_test = corrector.predict(
            X_corr_test
        ).astype(np.float64)

        pred_final_test = pred_stage1_test + pred_corrector_test

        if not np.allclose(
            pred_final_test,
            pred_stage1_test + pred_corrector_test,
            atol=1e-12,
            rtol=0,
        ):
            raise AssertionError("pred_final != pred_stage1 + pred_corrector")

        final_holdout_metrics = metrics(y_test, pred_final_test)
        logger.info(
            "FINAL HOLDOUT METRICS | %s",
            json.dumps(final_holdout_metrics),
        )

        # -----------------------------------------------------------------
        # Holdout prediction table
        # -----------------------------------------------------------------
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
            "model_family": "xgb",
            "obs_pm25": y_test,
            "pred_stage1": pred_stage1_test,
            "pred_corrector": pred_corrector_test,
            "pred_final": pred_final_test,
            "resid_stage1": y_test - pred_stage1_test,
            "resid_final": y_test - pred_final_test,
        })

        if len(predictions) != len(test_df):
            raise AssertionError("Prediction row count mismatch")

        duplicate_rows = int(
            predictions.duplicated(["grid_cell_id", "date"]).sum()
        )
        if duplicate_rows:
            raise ValueError(
                f"Holdout predictions contain {duplicate_rows} duplicate cell-days"
            )

        if predictions[
            ["obs_pm25", "pred_stage1", "pred_corrector", "pred_final"]
        ].isna().any().any():
            raise ValueError("NaN in holdout prediction output")

        observed_cases = set(predictions["case_key"].unique())
        if observed_cases != set(test_blocks):
            raise AssertionError(
                "Prediction table does not contain exactly the 9 planned holdout blocks"
            )

        # -----------------------------------------------------------------
        # Per-block metrics
        # -----------------------------------------------------------------
        block_metrics = []

        for block, g in predictions.groupby("case_key", sort=True):
            m1 = metrics(
                g["obs_pm25"].to_numpy(),
                g["pred_stage1"].to_numpy(),
            )
            mf = metrics(
                g["obs_pm25"].to_numpy(),
                g["pred_final"].to_numpy(),
            )

            row = {"case_key": block}
            row.update({f"stage1_{k}": v for k, v in m1.items()})
            row.update({f"final_{k}": v for k, v in mf.items()})
            block_metrics.append(row)

        block_metrics_df = pd.DataFrame(block_metrics)

        # -----------------------------------------------------------------
        # Save models
        # -----------------------------------------------------------------
        stage1_bundle = {
            "model": stage1,
            "model_family": "xgb",
            "stage": "stage1",
            "seed": SEED,
            "params": STAGE1_PARAMS,
            "feature_cols": canonical_features,
            "feature_count": len(canonical_features),
            "fold_no": int(args.fold),
            "train_blocks": train_blocks,
            "holdout_blocks": test_blocks,
            "plan_json": str(plan_path),
            "shard_root": str(args.shard_root),
        }

        corrector_bundle = {
            "model": corrector,
            "model_family": "xgb",
            "stage": "corrector",
            "seed": SEED,
            "params": CORRECTOR_PARAMS,
            "raw_feature_cols": CORRECTOR_RAW_COLS,
            "corrector_feature_cols": corrector_feature_names,
            "fill_values": fill_values,
            "pool_diagnostics": pool_diag,
            "fold_no": int(args.fold),
            "train_blocks": train_blocks,
            "holdout_blocks": test_blocks,
        }

        save_pickle(stage1_bundle, fold_out / "stage1_model.pkl")
        save_pickle(corrector_bundle, fold_out / "corrector_model.pkl")

        # Native XGBoost model representations as additional robust artifacts.
        stage1.save_model(str(fold_out / "stage1_model.json"))
        corrector.save_model(str(fold_out / "corrector_model.json"))

        # -----------------------------------------------------------------
        # Save predictions and diagnostics
        # -----------------------------------------------------------------
        predictions.to_parquet(
            fold_out / "holdout_predictions.parquet",
            index=False,
        )
        predictions.to_csv(
            fold_out / "holdout_predictions.csv",
            index=False,
        )
        block_metrics_df.to_csv(
            fold_out / "holdout_block_metrics.csv",
            index=False,
        )

        save_json(
            pool_diag,
            fold_out / "corrector_pool_diagnostics.json",
        )

        metric_payload = {
            "fold_no": int(args.fold),
            "fold_label": f"GROUP_{args.fold:02d}",
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(test_df)),
            "corrector_train_rows": int(len(pool_df)),
            "stage1_holdout": stage1_holdout_metrics,
            "final_holdout": final_holdout_metrics,
            "corrector_holdout_output": {
                "mean": float(np.mean(pred_corrector_test)),
                "std": float(np.std(pred_corrector_test)),
                "min": float(np.min(pred_corrector_test)),
                "median": float(np.median(pred_corrector_test)),
                "max": float(np.max(pred_corrector_test)),
                "mean_abs": float(np.mean(np.abs(pred_corrector_test))),
                "p99_abs": float(np.quantile(np.abs(pred_corrector_test), 0.99)),
                "max_abs": float(np.max(np.abs(pred_corrector_test))),
            },
        }
        save_json(metric_payload, fold_out / "metrics.json")

        manifest_out = {
            "status": "complete",
            "fold_no": int(args.fold),
            "fold_label": f"GROUP_{args.fold:02d}",
            "assignment_source": plan.get(
                "assignment_source",
                "Thesis Appendix A, Table A5",
            ),
            "plan_json": str(plan_path),
            "shard_root": str(args.shard_root),
            "train_block_count": len(train_blocks),
            "holdout_block_count": len(test_blocks),
            "train_blocks": train_blocks,
            "holdout_blocks": test_blocks,
            "stage1_feature_count": len(canonical_features),
            "stage1_feature_source": str(manifest_path),
            "stage1_nan_handling": "NaN -> 0",
            "corrector_raw_feature_count": len(CORRECTOR_RAW_COLS),
            "corrector_final_feature_count": len(corrector_feature_names),
            "corrector_target": "observed_pm25 - in_sample_stage1_prediction",
            "corrector_pool_rule": (
                "smoke/fire signal OR observed_pm25 >= 15 "
                "OR abs(in_sample_stage1_residual) >= 3"
            ),
            "cv_prediction_clipping": "none",
            "seed": SEED,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xgboost": xgboost.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "stage1_params": STAGE1_PARAMS,
            "corrector_params": CORRECTOR_PARAMS,
            "train_loader_summary": train_load_info,
            "holdout_loader_summary": test_load_info,
            "elapsed_seconds": float(time.time() - start),
        }
        save_json(manifest_out, fold_out / "run_manifest.json")

        logger.info("=" * 100)
        logger.info("FOLD %02d COMPLETE", args.fold)
        logger.info("stage1_model.pkl")
        logger.info("corrector_model.pkl")
        logger.info("holdout_predictions.parquet")
        logger.info("holdout_predictions.csv")
        logger.info("metrics.json")
        logger.info("holdout_block_metrics.csv")
        logger.info("corrector_pool_diagnostics.json")
        logger.info("run_manifest.json")
        logger.info("elapsed_seconds=%.1f", time.time() - start)
        logger.info("=" * 100)

        return 0

    except Exception:
        logger.exception("FOLD %02d FAILED", args.fold)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
