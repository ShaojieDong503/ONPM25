#!/usr/bin/env python3
# GENERATED FILE -- do not edit by hand.
# Derived from run_lgbm_thesis_fold.py by build_thesis_fold_scripts.py.
# Edit the LightGBM script or the substitution table, then regenerate.
"""
Run ONE thesis Random Forest outer fold for the Ontario PM2.5 model.

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
  stage1_estimator.joblib
  corrector_estimator.joblib
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
- Random Forest regression
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
- same Random Forest hyperparameters as Stage 1
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
import joblib
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Configuration
# =============================================================================

SEED = 2026

HERE = Path(__file__).resolve().parent

# The two paths below are the Windows workstation's, and they are dangerous
# anywhere else. On Linux a backslash is an ordinary filename character, so
# Path(r"D:\lambda\...") does not raise and does not resolve -- it is simply a
# RELATIVE name that happens to contain backslashes. A run on the VM therefore
# created one directory called `D:\lambda\...\thesis_rf_grouped_runs` under the
# cwd and wrote 104 GB of Random Forest into it, where compare_models.py does
# not look. Silence, not an error. So: use them when they are really there, and
# otherwise fall back beside this file.
FAMILY_OUT_DIRNAME = "rf_thesis"

DEFAULT_SHARD_ROOT = Path(
    r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle"
    r"\Ontario_RealTarget_GPD\outputs"
    r"\materialized_support_family_shards_pruned_temporal_x_repaired"
)

DEFAULT_OUT_ROOT = Path(
    r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle"
    r"\Ontario_RealTarget_GPD\outputs"
    r"\thesis_rf_grouped_runs"
)

# .parent for the out root: the leaf is created by the run, so its absence says
# nothing, but its parent existing is what tells us we are on the workstation.
if not DEFAULT_SHARD_ROOT.exists():
    DEFAULT_SHARD_ROOT = HERE / "Data"
if not DEFAULT_OUT_ROOT.parent.exists():
    DEFAULT_OUT_ROOT = HERE / "outputs" / FAMILY_OUT_DIRNAME

STAGE1_PARAMS = {
    "n_estimators": 900,
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": 0.6,
    "bootstrap": True,
    "random_state": SEED,
    "n_jobs": -1,
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

# Stage-1 missing-value fills.
#
#   zero    (default, the thesis rule) every residual NaN -> 0.
#   tiered  distance-like predictors -> DISTANCE_FILL, everything else -> 0.
#
# The distinction is what 0 MEANS. For a count, a fraction or a presence indicator,
# absence genuinely is zero and the fill is correct. For a distance-to-nearest, 0
# means "a fire directly overhead" -- the maximum-signal end -- so filling absence
# with 0 inverts the variable on the 49% of rows where no fire is in range.
# DISTANCE_FILL sits beyond every radius (max observed 999.7 km), so it reads as
# "farther than anything seen", which is what absence actually means.
#
# NOTE the corrector uses max(observed)+10 per column (110/260/510/1010) rather than
# one flat constant. Both encode "far"; they are simply different conventions.
DISTANCE_FILL = 9999.0
FILL_POLICIES = ("zero", "tiered")


def distance_feature_mask(canonical_features: Sequence[str]) -> np.ndarray:
    """Columns where 0 is the strongest signal rather than the absence of one."""
    return np.array([("dist_nearest" in c or "nearest_km" in c)
                     for c in canonical_features], dtype=bool)


def build_corrector_raw_cols() -> List[str]:
    """Construct the corrector's fixed 40-input design, in a fixed order.

    The order matters: it becomes the column order of the 80-wide corrector matrix,
    and the fitted model is only reusable against that same order. Built
    programmatically from the radius lists rather than written out by hand so the
    count cannot silently drift; the assert at the end is the guard.

        pred_stage1          1
        VIIRS same-day      12   count / dist_nearest / frp_mw_sum  x 100,250,500,1000 km
        VIIRS lag-1          8   count / dist_nearest               x 4 radii
        HMS smoke            8   any / density_weight_wmean         x 4 radii
        burned area          6   frac / weighted_frac / nearest_km  x 500,1000 km
        satellite AOD        5
        ----------------------
                            40

    Multi-day rolling aggregates are deliberately absent -- those belong to Stage 1.
    """
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
    #
    # Naming, measured rather than assumed (all 169,882 cell-days):
    #   aod_obs_flag      == "AOD_055 was retrieved".  It is NOT "either band
    #                        observed": it disagrees with that reading on 2 rows,
    #                        because AOD_047 is almost a strict subset of AOD_055.
    #                        Describe it as an AOD_055 observation flag.
    #   aod_imputed_flag  == 1 - aod_obs_flag, and both are identical to the derived
    #                        AOD_055__isna. Three of the corrector's 80 inputs
    #                        therefore carry the same bit; TreeSHAP splits credit
    #                        among perfectly collinear features arbitrarily, so only
    #                        the AOD family rollup is interpretable, never the split
    #                        between these three.
    #   AOD_055_filled    == AOD_055 exactly where observed (max diff 0.0), and a
    #                        genuine per-row estimate where imputed (79,737 distinct
    #                        values, not one constant).
    #
    # The definition is left as it is: redefining aod_obs_flag as the union would
    # change 2 rows in 169,882 and invalidate every fitted corrector to buy a better
    # name. The name is corrected in the text instead.
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
    """Log to both the fold's training.log (always DEBUG) and stdout.

    The file handler keeps full detail regardless of --verbose, so a run that looks
    fine on screen can still be audited afterwards. handlers.clear() makes repeated
    calls within one process idempotent instead of duplicating every line.
    """
    logger = logging.getLogger("thesis_rf_fold")
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
    """Read a UTF-8 JSON file (fold plan, manifest)."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: dict, path: Path) -> None:
    """Write JSON, indented for reading. default=str lets Path/NumPy scalars through."""
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def save_pickle(obj, path: Path) -> None:
    """Pickle a fitted-model bundle. Paired with a native .txt export, since pickle
    is sensitive to the library version that wrote it."""
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# Fold plan validation
# =============================================================================

def load_fold_plan(case_plans_dir: Path, fold_no: int) -> Tuple[dict, Path]:
    """Read GROUP_0N.json and refuse anything that is not the thesis Table A5 design.

    Every check here is a precondition of the CV being valid at all, so each one
    raises rather than warns:
      - 63 train + 9 holdout blocks, no duplicates within either list
      - no block appearing in both (train/test contamination)
      - the union covering exactly the 72 region-year blocks
      - support_blocks empty -- this is the Ontario-only run; any out-of-province
        block would silently change what the model is
      - fold_no inside the file agreeing with the fold requested on the command line,
        which catches a renamed or copied plan
    """
    path = case_plans_dir / f"GROUP_{fold_no:02d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold plan not found: {path}")

    plan = read_json(path)

    train_blocks = list(plan.get("train_pair_blocks", []))
    test_blocks = list(plan.get("heldout_case_keys", []))

    # The thesis design is 63/9/72 and that stays the default, so a thesis plan is
    # validated exactly as before. A plan may DECLARE different counts, which is how
    # leave-cells-out runs through this same code path: its "blocks" are cells, so the
    # shape is 35/6/41, not 63/9/72. Declaring the numbers keeps the check strict --
    # a truncated or mis-generated plan still fails -- while letting the design vary.
    n_train = int(plan.get("expected_train_count", 63))
    n_test = int(plan.get("expected_heldout_count", 9))
    n_total = int(plan.get("expected_total_count", 72))

    if len(train_blocks) != n_train:
        raise ValueError(
            f"{path.name}: expected {n_train} training blocks, found {len(train_blocks)}"
        )
    if len(test_blocks) != n_test:
        raise ValueError(
            f"{path.name}: expected {n_test} held-out blocks, found {len(test_blocks)}"
        )
    if len(set(train_blocks)) != n_train:
        raise ValueError(f"{path.name}: duplicate training block(s)")
    if len(set(test_blocks)) != n_test:
        raise ValueError(f"{path.name}: duplicate held-out block(s)")
    if set(train_blocks) & set(test_blocks):
        raise ValueError(f"{path.name}: train/holdout block overlap")
    if len(set(train_blocks) | set(test_blocks)) != n_total:
        raise ValueError(
            f"{path.name}: train+holdout do not cover exactly {n_total} unique blocks"
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
    """Map the manifest's 651 canonical names onto the names the parquet stores.

    Two sources of truth exist -- manifest.json names the features, the parquet
    stores them -- and they disagree on the rolling columns. Each canonical name is
    tried verbatim first, then through shard_frame_column().

    Returns (stored_cols, rename_map, counts). Expected counts for the repaired
    shards: direct=213, translated=438, absent=0.

    An unresolved name raises instead of being backfilled. That is the whole point:
    the historical failure was silently substituting 0.0 for 438 real columns, which
    trained a 213-feature model wearing a 651-feature label and produced no error.
    """
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
    """<shard_root>/pair_blocks/PAIR_<year>_<region>/frame.parquet"""
    return shard_root / "pair_blocks" / block / "frame.parquet"


def load_one_block(
    shard_root: Path,
    block: str,
    canonical_features: Sequence[str],
    logger: logging.Logger,
    fill_policy: str = "zero",
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Load one region-year block and return (full frame, Stage-1 matrix, load info).

    The frame is returned whole -- metadata columns included -- because the corrector
    and the prediction table need them later. Only the 651 predictors go into X.

    Order of operations matters:
      1. reindex to canonical_features, so every block yields identically ordered
         columns and the assert below can catch any drift
      2. inf is a hard error, never silently converted
      3. NaN -> 0 (the thesis Stage-1 rule), counted first so the log records how
         much was filled rather than hiding it
      4. re-assert finiteness, so a fill that failed cannot reach the model
    """
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
    dist_filled = 0
    if nan_count:
        if fill_policy == "tiered":
            # Fill the distance-like columns with the far-sentinel FIRST, then let the
            # zero rule take the rest. Order matters: nan_to_num would otherwise have
            # already turned them into 0 and the sentinel would have nothing to do.
            dmask = distance_feature_mask(canonical_features)
            if dmask.any():
                sub = X[:, dmask]
                dist_filled = int(np.isnan(sub).sum())
                X[:, dmask] = np.nan_to_num(sub, nan=DISTANCE_FILL)
        # Thesis Stage-1 rule: remaining missing predictor values -> 0.
        X = np.nan_to_num(X, nan=0.0)

    if not np.isfinite(X).all():
        raise ValueError(f"{block}: non-finite Stage-1 values remain after zero fill")

    logger.info(
        "LOAD | %s | rows=%d | X=%s | direct=%d | translated=%d | "
        "absent=%d | nan_filled=%d (distance_sentinel=%d, zero=%d)",
        block,
        len(df),
        tuple(X.shape),
        resolution["direct"],
        resolution["translated"],
        resolution["absent"],
        nan_count,
        dist_filled,
        nan_count - dist_filled,
    )

    info = {
        "block": block,
        "rows": int(len(df)),
        "direct": int(resolution["direct"]),
        "translated": int(resolution["translated"]),
        "absent": int(resolution["absent"]),
        "nan_zero_filled": nan_count - dist_filled,
        "nan_distance_filled": dist_filled,
        "fill_policy": fill_policy,
    }
    return df, X, info


def rebuild_matrix(df: pd.DataFrame, canonical_features: Sequence[str],
                   fill_policy: str, label: str) -> np.ndarray:
    """Rebuild the Stage-1 matrix from a frame whose columns have been modified.

    load_blocks returns the frame and the matrix together, so anything that edits the
    frame afterwards -- the fold-local AOD refill, for instance -- leaves the matrix
    stale. This redoes exactly what load_one_block does to get from frame to matrix:
    resolve the stored names, reindex to canonical order, reject inf, apply the fill.
    """
    stored_cols, rename_map, resolution = resolve_stage1_columns(df.columns, canonical_features)
    xdf = df[stored_cols].rename(columns=rename_map).reindex(columns=list(canonical_features))
    if list(xdf.columns) != list(canonical_features):
        raise AssertionError(f"{label}: Stage-1 feature order mismatch after rebuild")
    X = xdf.to_numpy(dtype=np.float32, copy=True)
    if int(np.isinf(X).sum()):
        raise ValueError(f"{label}: infinite Stage-1 values after rebuild")
    if int(np.isnan(X).sum()):
        if fill_policy == "tiered":
            dmask = distance_feature_mask(canonical_features)
            if dmask.any():
                X[:, dmask] = np.nan_to_num(X[:, dmask], nan=DISTANCE_FILL)
        X = np.nan_to_num(X, nan=0.0)
    if not np.isfinite(X).all():
        raise ValueError(f"{label}: non-finite values remain after rebuild")
    return X


def load_blocks(
    shard_root: Path,
    blocks: Sequence[str],
    canonical_features: Sequence[str],
    logger: logging.Logger,
    fill_policy: str = "zero",
) -> Tuple[pd.DataFrame, np.ndarray, List[dict]]:
    """Concatenate many blocks into one training or holdout set.

    `_source_block` is stamped on every row before concatenation -- it is what lets
    per-block metrics and the `case_key` column be recovered afterwards, since the
    blocks become indistinguishable once stacked.

    The two asserts guard the invariant everything downstream assumes: the metadata
    frame and the feature matrix are row-aligned, position by position.
    """
    frames = []
    matrices = []
    infos = []

    for i, block in enumerate(blocks, 1):
        logger.info("Reading %d/%d: %s", i, len(blocks), block)
        df, X, info = load_one_block(
            shard_root, block, canonical_features, logger, fill_policy
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
    """Held-out performance for one set of observation/prediction pairs.

    Deliberate choices, because the alternatives are common and wrong here:
      r2_predictive        1 - SSE/SST, NOT squared Pearson. It can go negative for a
                           badly predicted block, and that is informative rather than
                           a bug -- squared correlation would hide it by ignoring
                           bias and scale.
      bias_pred_minus_obs  mean(pred - obs), so positive means OVERprediction.
      within_3             fraction within +/-3 ug/m3, the interpretable accuracy
                           statistic that RMSE alone does not convey.
      slope/intercept      regression of prediction on observation; slope < 1 is the
                           usual signature of shrinkage toward the mean. Guarded on
                           std(y) > 0 because a constant block makes the fit singular.
    """
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
    """Fail before fitting if any of the 39 source columns is absent.

    pred_stage1 is excluded because it does not exist in the parquet -- it is
    computed at runtime and attached to the frame just before this is called.
    Checked on both train and holdout, since a column present in one and missing in
    the other would otherwise surface only as a shape error much later.
    """
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
    """Rows the corrector trains on.

    IN PRACTICE THIS SELECTS EVERYTHING. The rule reads as a difficulty filter --
    smoke/fire signal OR pm25 >= 15 OR |residual| >= 3 -- but `smoke_mask` tests
    whether any of 34 fire/smoke columns is positive, and two of them
    (`src_hms_density_weight_wmean_500km` and `_1000km`) are positive on every row:
    over a 500-1000 km neighbourhood there is always some HMS smoke contribution.
    Measured on every fold, `pool_fraction == 1.0` and `pool_rows == training_rows`.

    So the corrector is a plain second-stage model fitted to the Stage-1 TRAINING
    residual on ALL training rows. It is not trained only on smoke days, not trained
    only on high-error days, and its target is in-sample -- see `run_oof_corrector`
    in robustness_runner.py for the genuinely out-of-fold variant. Describe it that
    way in the text; the diagnostics JSON records the fraction so the claim stays
    checkable per fold.

    The rule is left exactly as it is: it is what produced the reported models, and
    narrowing it now would be a silent change to the method, not a documentation fix.
    """
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
      0 if no finite value exists

    Other inputs:
      training-pool median
      0 if no finite value exists
    """
    fills = {}
    n_missing = 0

    for c in CORRECTOR_RAW_COLS:
        x = pd.to_numeric(pool_df[c], errors="coerce").to_numpy(dtype=np.float64)
        finite = x[np.isfinite(x)]
        n_missing += int(len(x) - len(finite))

        if "dist_nearest" in c or "nearest_km" in c:
            fills[c] = float(finite.max() + 10.0) if len(finite) else 0.0
        else:
            fills[c] = float(np.median(finite)) if len(finite) else 0.0

    # A pool with zero missing values across all 40 corrector inputs is not a clean
    # dataset, it is a pre-filled one: some upstream step already replaced NaN with 0.
    # That silently zeroes every `__isna` flag and drags every median toward 0, and it
    # is invisible downstream because the model still trains. Refuse to continue.
    # (Stage 1 is unaffected -- load_one_block zero-fills X either way -- so the two
    # shard roots differ only here, which is exactly why this went unnoticed.)
    if n_missing == 0:
        raise SystemExit(
            "[error] corrector pool has 0 missing values across all "
            f"{len(CORRECTOR_RAW_COLS)} raw inputs ({len(pool_df):,} rows). The shard "
            "root looks pre-zero-filled; point --shard-root at the raw shards (Data/), "
            "not Data_zerofilled/. Missingness flags must be derived before filling."
        )

    return fills


def transform_corrector(
    df: pd.DataFrame,
    fills: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """Encode the 40 raw inputs as 80 columns: each value followed by its missing flag.

    Interleaved as [value, value__isna, ...] rather than grouped, so the pairing is
    positional and cannot drift.

    `fills` must come from fit_corrector_fill_values() on the TRAINING pool and be
    passed unchanged when transforming holdout rows -- recomputing them from the
    holdout would leak test-set distribution into the features.

    Filling and flagging together is strictly more informative than letting the
    learner infer missingness natively: the model can condition on "this was absent"
    directly, and the flag preserves that information even where the fill value is
    itself a plausible observation.
    """
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
    """Run one outer fold end to end, in nine steps.

        1  read and validate the fold plan          63 train / 9 holdout blocks
        2  read the manifest, pre-flight the files  651 canonical predictors
        3  load both sides through one loader       identical encoding
        4  fit Stage 1, predict in-sample and out   in-sample feeds step 5
        5  build corrector inputs                   40 raw -> 80 with missing flags
        6  fit the corrector, form pred_final       pred_stage1 + pred_corrector
        7  assemble the out-of-fold table           one row per held-out cell-day
        8  per-block metrics                        Stage 1 vs final, block by block
        9  persist models, predictions, manifest

    Returns 0 on success, 1 on any failure -- the whole body is wrapped so a
    traceback lands in training.log rather than only on the terminal, and the exit
    code is usable by a shell loop driving all 8 folds.
    """
    ap = argparse.ArgumentParser(
        description="Run one thesis Table-A5 Random Forest fold (Stage 1 + corrector)."
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
    ap.add_argument(
        "--features-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON list or newline-delimited file naming the Stage-1 "
            "predictors to use. Must be a SUBSET of manifest.json's feature_cols. "
            "This is how a feature ablation runs through the production flow "
            "unchanged -- only the input changes, not the code path."
        ),
    )
    ap.add_argument(
        "--fill-policy",
        choices=FILL_POLICIES,
        default="zero",
        help=("zero (default, the thesis rule) fills every residual NaN with 0. "
              "tiered sends distance-like predictors to DISTANCE_FILL instead, "
              "because 0 there means 'a fire directly overhead', not absence."),
    )
    ap.add_argument(
        "--aod-fill",
        choices=("global", "fold"),
        default="global",
        help=(
            "global (default) keeps the shards' AOD_055_filled, imputed once over the "
            "whole panel -- including rows that become this fold's held-out set. "
            "fold refits the AOD gap-filler on THIS fold's training rows only and "
            "re-imputes both frames, recomputing the 7 temporal derivatives. See "
            "aod_fold_filler.py; spec G calls this a revision, so it is opt-in."
        ),
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "estimator thread count. MUST be set when several folds run concurrently: "
            "n_jobs=-1 (the default) makes every process claim all cores, so 4 jobs on "
            "a 32-vCPU box request 128 threads and the OpenMP barriers spin instead of "
            "working. LightGBM's n_jobs overrides OMP_NUM_THREADS, so exporting that is "
            "not enough -- the estimator's own count has to be set."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.threads:
        STAGE1_PARAMS["n_jobs"] = int(args.threads)
        CORRECTOR_PARAMS["n_jobs"] = int(args.threads)

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
        logger.info("THESIS RANDOM FOREST OUTER FOLD START")
        logger.info("fold=%d", args.fold)
        logger.info("shard_root=%s", args.shard_root)
        logger.info("case_plans_dir=%s", case_plans_dir)
        logger.info("output=%s", fold_out)
        logger.info("=" * 100)

        # -----------------------------------------------------------------
        # STEP 1 of 9 -- Plan
        # Which 63 blocks train and which 9 are held out. Read first, before any
        # data is touched, so an invalid design costs nothing.
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
        # STEP 2 of 9 -- Manifest and pre-flight
        # The manifest defines the 651 canonical predictors and their order. Both
        # the count and the uniqueness are asserted, then every referenced parquet
        # is confirmed to exist -- so a missing block fails in seconds rather than
        # after the training blocks have already been read.
        # -----------------------------------------------------------------
        manifest_path = args.shard_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        manifest = read_json(manifest_path)
        manifest_features = list(manifest.get("feature_cols", []))

        if len(manifest_features) != 651:
            raise ValueError(
                f"Current manifest has {len(manifest_features)} Stage-1 features; expected 651"
            )
        if len(set(manifest_features)) != 651:
            raise ValueError("Manifest feature names are not unique")

        # A feature ablation is a change of INPUT, not of method: the fold plan,
        # hyperparameters, NaN policy, corrector design and every post-condition stay
        # exactly as they are. Restricting the subset to names already in the manifest
        # means a typo cannot silently invent a different feature set -- it fails here.
        if args.features_file is None:
            canonical_features = manifest_features
            feature_source = str(manifest_path)
        else:
            raw = args.features_file.read_text(encoding="utf-8").strip()
            requested = (json.loads(raw) if raw.startswith("[")
                         else [ln.strip() for ln in raw.splitlines() if ln.strip()])
            requested = list(dict.fromkeys(str(c) for c in requested))
            if not requested:
                raise ValueError(f"{args.features_file}: no feature names found")
            unknown = [c for c in requested if c not in set(manifest_features)]
            if unknown:
                raise ValueError(
                    f"{args.features_file}: {len(unknown)} name(s) are not in the "
                    f"manifest, so they are not a subset of the 651. "
                    f"Examples: {unknown[:10]}"
                )
            # keep manifest order, so column order is identical to the full run
            canonical_features = [c for c in manifest_features if c in set(requested)]
            feature_source = str(args.features_file)
            logger.info(
                "FEATURE SUBSET | %s | using %d of %d manifest predictors",
                args.features_file.name, len(canonical_features), len(manifest_features),
            )

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
        # STEP 3 of 9 -- Load
        # Train and holdout are read through the same loader, so name resolution,
        # zero-fill and column ordering are identical on both sides. A difference
        # here would be train/predict skew: the model fitted on one encoding and
        # scored on another.
        # -----------------------------------------------------------------
        logger.info("Loading 63 training blocks...")
        train_df, X_train, train_load_info = load_blocks(
            args.shard_root,
            train_blocks,
            canonical_features,
            logger,
            fill_policy=args.fill_policy,
        )

        logger.info("Loading 9 held-out blocks...")
        test_df, X_test, test_load_info = load_blocks(
            args.shard_root,
            test_blocks,
            canonical_features,
            logger,
            fill_policy=args.fill_policy,
        )

        # Fold-local AOD imputation, if asked for. Must happen BEFORE the target and
        # the matrices are used: it rewrites AOD_055_filled and its 7 derivatives, so
        # the matrices load_blocks already built are stale and are rebuilt here.
        aod_diag = {"mode": args.aod_fill}
        if args.aod_fill == "fold":
            import aod_fold_filler as AF
            train_df, test_df, aod_diag_fold = AF.refill_fold(train_df, test_df, logger)
            aod_diag.update(aod_diag_fold)
            X_train = rebuild_matrix(train_df, canonical_features,
                                     args.fill_policy, "train/aod-refill")
            X_test = rebuild_matrix(test_df, canonical_features,
                                    args.fill_policy, "holdout/aod-refill")
            logger.info("AOD REFILL | matrices rebuilt | train=%s holdout=%s",
                        X_train.shape, X_test.shape)

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
        # STEP 4 of 9 -- Stage 1
        # Fit on the 63 training blocks only. No eval_set, no early stopping and no
        # inner search: the hyperparameters are fixed, so the held-out blocks play
        # no part in fitting and the holdout estimate stays honest.
        #
        # pred_stage1_train is IN-SAMPLE by design -- it becomes the corrector's
        # target below. That is the thesis definition, and it means the corrector
        # learns to correct residuals the Stage-1 model has already seen.
        # -----------------------------------------------------------------
        logger.info("Fitting Stage-1 Random Forest...")
        stage1 = RandomForestRegressor(**STAGE1_PARAMS)
        # Neither XGBoost's sklearn API nor RandomForestRegressor accepts
        # feature_name in fit(); the names are preserved in the saved bundle instead.
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
        # STEP 5 of 9 -- Corrector inputs
        # Select the training rows, then derive the fill values from those rows
        # ONLY and reuse them unchanged on the holdout. pm25 and resid_stage1
        # choose rows; neither becomes a feature, so nothing the corrector needs
        # at prediction time depends on knowing the answer.
        #
        # See build_corrector_pool's docstring: this rule selects every row in
        # practice, so the corrector is effectively a plain second stage.
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
        # STEP 6 of 9 -- Fit the corrector and form the final prediction
        # Same learner and same hyperparameters as Stage 1, on 80 columns instead
        # of 651. Applied to EVERY held-out row, not only the difficult ones.
        #
        #     pred_final = pred_stage1 + pred_corrector
        #
        # No clipping at zero: slightly negative predictions are kept so the CV
        # evaluation stays unbiased. Flooring is reserved for the released grid.
        # -----------------------------------------------------------------
        logger.info("Fitting residual-corrector Random Forest...")
        corrector = RandomForestRegressor(**CORRECTOR_PARAMS)
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
        # STEP 7 of 9 -- Out-of-fold prediction table
        # One row per held-out cell-day, carrying enough metadata to join back to
        # observations and to be concatenated across all 8 folds.
        #
        # Four post-conditions, each catching a failure that would otherwise look
        # like a successful run: row count preserved, no duplicate cell-day, no NaN
        # in any prediction column, and exactly the 9 planned blocks present.
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
            "model_family": "rf",
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
        # STEP 8 of 9 -- Per-block metrics
        # Stage-1 and final scored separately for each of the 9 held-out
        # region-year blocks, so the corrector's contribution is visible block by
        # block rather than only in the pooled number, where it can average away.
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
        # STEP 9 of 9 -- Persist everything needed to audit or reuse the fold
        # Models are saved twice: pickle (round-trips the sklearn wrapper, but is
        # version-sensitive) and LightGBM's native .txt (portable, survives a
        # library upgrade). The bundles carry feature order, fill values and the
        # block lists, because a model is not reusable without them.
        #
        # run_manifest.json records the plan, the NaN policy, the pool rule, seed
        # and library versions -- so a later reader can tell what was run without
        # re-reading this file.
        # -----------------------------------------------------------------
        stage1_bundle = {
            "model": stage1,
            "model_family": "rf",
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
            "model_family": "rf",
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

        # scikit-learn has no portable native format, so joblib is the only option.
        # It is version-sensitive in the same way the pickle is -- the run_manifest
        # records scikit_learn.__version__ so a later reader knows what wrote it.
        joblib.dump(stage1, fold_out / "stage1_estimator.joblib", compress=3)
        joblib.dump(corrector, fold_out / "corrector_estimator.joblib", compress=3)

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
            "stage1_feature_source": feature_source,
            "stage1_feature_subset": args.features_file is not None,
            "stage1_nan_handling": args.fill_policy,
            "aod_fill": aod_diag,
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
