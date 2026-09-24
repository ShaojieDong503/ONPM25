#!/usr/bin/env python3
"""
Shared two-stage fit/predict, lifted straight out of `run_lgbm_thesis_fold.py`.

Nothing here re-implements the pipeline. Every step -- block loading, name
resolution, Stage-1 fit, corrector pool rule, fill values, corrector transform,
metrics -- is the function the fold script itself calls. This module only supplies
the *data* and collects the results, so a robustness run, a final all-Ontario fit,
an external-province test and a raster prediction all share one code path.

If `run_lgbm_thesis_fold.py` changes, everything built on this changes with it.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_lgbm_thesis_fold as F  # noqa: E402  the production fold script

# Re-export so callers never define their own copy.
STAGE1_PARAMS = F.STAGE1_PARAMS
CORRECTOR_PARAMS = F.CORRECTOR_PARAMS
CORRECTOR_RAW_COLS = F.CORRECTOR_RAW_COLS
metrics = F.metrics
load_blocks = F.load_blocks
load_fold_plan = F.load_fold_plan
save_json = F.save_json
save_pickle = F.save_pickle
configure_logger = F.configure_logger
read_json = F.read_json
pair_block_path = F.pair_block_path


def null_logger(name: str = "thesis_core") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.ERROR)
    return lg


def canonical_features(shard_root: Path | str, *, expect: int | None = 651) -> list[str]:
    shard_root = Path(shard_root)          # accept a string path from a caller
    feats = list(read_json(shard_root / "manifest.json").get("feature_cols", []))
    if expect is not None and len(feats) != expect:
        raise ValueError(f"{shard_root/'manifest.json'} has {len(feats)} features, expected {expect}")
    if len(set(feats)) != len(feats):
        raise ValueError("manifest feature names are not unique")
    return feats


def check_blocks_exist(shard_root: Path | str, blocks: Sequence[str]) -> None:
    """Fail before a model fit rather than after, as the fold script does."""
    shard_root = Path(shard_root)
    missing = [str(pair_block_path(shard_root, b)) for b in blocks
               if not pair_block_path(shard_root, b).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced parquet(s) missing. "
                                f"Examples: {missing[:10]}")


# --------------------------------------------------------------------------- result

@dataclass
class TwoStage:
    """A fitted two-stage model plus whatever it was scored on."""
    stage1: object
    corrector: object
    feature_cols: list[str]
    corrector_cols: list[str]
    fill_values: dict
    pool_diag: dict
    train_rows: int
    # Stage-1 predictions on the TRAINING rows, keyed so they can be rejoined.
    # Retained because the corrector's target is the training residual: with these
    # cached, a corrector can be refitted without refitting Stage 1 at all, which is
    # the difference between minutes and hours when only corrector handling changes.
    train_keys: pd.DataFrame | None = None
    pred_stage1_train: np.ndarray | None = None
    # Populated only when a test set was supplied.
    test_df: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    pred_stage1: np.ndarray | None = None
    pred_corrector: np.ndarray | None = None
    pred_final: np.ndarray | None = None
    stage1_metrics: dict = field(default_factory=dict)
    final_metrics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- fit

def fit_two_stage(
    *,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    logger: logging.Logger,
    stage1_params: dict | None = None,
    corrector_params: dict | None = None,
) -> TwoStage:
    """Stage 1 then the residual corrector, exactly as the fold script does it."""
    from lightgbm import LGBMRegressor

    feature_cols = list(feature_cols)
    y_train = pd.to_numeric(train_df["pm25"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_train).all():
        raise ValueError("Training PM2.5 contains non-finite values")

    logger.info("TRAIN | rows=%d | X=%s", len(train_df), X_train.shape)
    logger.info("Fitting Stage-1 LightGBM...")
    stage1 = LGBMRegressor(**(stage1_params or STAGE1_PARAMS))
    stage1.fit(X_train, y_train, feature_name=feature_cols)

    pred_train = stage1.predict(X_train).astype(np.float64)

    train_df = train_df.copy()
    train_df["pred_stage1"] = pred_train
    train_df["resid_stage1"] = y_train - pred_train

    # Corrector pool rule, fill values and transform all come from the fold script.
    F.validate_corrector_columns(train_df)
    pool_df, _pool_mask, pool_diag = F.build_corrector_pool(train_df, logger)
    y_corr = pd.to_numeric(pool_df["resid_stage1"], errors="raise").to_numpy(dtype=np.float64)

    fill_values = F.fit_corrector_fill_values(pool_df)
    X_corr, corrector_cols = F.transform_corrector(pool_df, fill_values)

    logger.info("CORRECTOR | train_rows=%d | X_train=%s", len(pool_df), X_corr.shape)
    logger.info("Fitting residual-corrector LightGBM...")
    corrector = LGBMRegressor(**(corrector_params or CORRECTOR_PARAMS))
    corrector.fit(X_corr, y_corr, feature_name=corrector_cols)

    keys = [c for c in ("CanOSSEM_RASTER_CELL", "date") if c in train_df.columns]
    return TwoStage(stage1=stage1, corrector=corrector, feature_cols=feature_cols,
                    corrector_cols=corrector_cols, fill_values=fill_values,
                    pool_diag=pool_diag, train_rows=int(len(train_df)),
                    train_keys=train_df[keys].copy() if keys else None,
                    pred_stage1_train=pred_train)


def fit_stage1(X: np.ndarray, y: np.ndarray, feature_cols: Sequence[str],
               params: dict | None = None):
    """Stage 1 alone, exactly as the fold script fits it.

    Used by designs that need many Stage-1 models without a corrector each time
    (e.g. building out-of-fold residuals).
    """
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(**(params or STAGE1_PARAMS))
    m.fit(X, y, feature_name=list(feature_cols))
    return m


def score(model: TwoStage, X: np.ndarray, df: pd.DataFrame,
          logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fitted two-stage model. Returns (stage1, corrector, final).

    The corrector transform reuses the TRAINING fill values -- never recomputed from
    the data being scored.
    """
    pred_stage1 = model.stage1.predict(X).astype(np.float64)
    d = df.copy()
    d["pred_stage1"] = pred_stage1
    F.validate_corrector_columns(d)
    X_corr, cols = F.transform_corrector(d, model.fill_values)
    if cols != model.corrector_cols:
        raise AssertionError("corrector feature order differs between fit and score")
    pred_corr = model.corrector.predict(X_corr).astype(np.float64)
    return pred_stage1, pred_corr, pred_stage1 + pred_corr


def fit_and_score(
    *,
    shard_root: Path,
    train_blocks: Sequence[str],
    test_blocks: Sequence[str] | None,
    feature_cols: Sequence[str],
    logger: logging.Logger,
    test_shard_root: Path | None = None,
) -> TwoStage:
    """Load, fit, and (optionally) score -- the whole fold in one call."""
    feature_cols = list(feature_cols)
    check_blocks_exist(shard_root, train_blocks)

    logger.info("Loading %d training blocks...", len(train_blocks))
    train_df, X_train, _ = load_blocks(shard_root, list(train_blocks), feature_cols, logger)
    model = fit_two_stage(X_train=X_train, train_df=train_df,
                          feature_cols=feature_cols, logger=logger)

    if not test_blocks:
        return model

    troot = test_shard_root or shard_root
    check_blocks_exist(troot, test_blocks)
    logger.info("Loading %d held-out blocks...", len(test_blocks))
    test_df, X_test, _ = load_blocks(troot, list(test_blocks), feature_cols, logger)
    y_test = pd.to_numeric(test_df["pm25"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_test).all():
        raise ValueError("Held-out PM2.5 contains non-finite values")

    p1, pc, pf = score(model, X_test, test_df, logger)

    # Same assertion the fold script makes.
    if not np.allclose(pf, p1 + pc, atol=1e-12, rtol=0):
        raise AssertionError("pred_final != pred_stage1 + pred_corrector")

    model.test_df = test_df
    model.y_test = y_test
    model.pred_stage1, model.pred_corrector, model.pred_final = p1, pc, pf
    model.stage1_metrics = metrics(y_test, p1)
    model.final_metrics = metrics(y_test, pf)
    logger.info("STAGE1 METRICS | %s", json.dumps(model.stage1_metrics))
    logger.info("FINAL  METRICS | %s", json.dumps(model.final_metrics))
    return model


# --------------------------------------------------------------------------- output

def stage1_train_table(model: TwoStage, *, fold: int | None = None) -> pd.DataFrame:
    """Stage-1 predictions on this fold's TRAINING rows, keyed by (cell, date).

    This is the whole input a corrector refit needs beyond the raw shards: the target
    is `pm25 - pred_stage1` on the training rows, and everything else the corrector
    sees comes from the frame. Caching it turns "the corrector handling changed" from
    a full re-fit of every Stage-1 model into a few minutes of work.
    """
    if model.pred_stage1_train is None or model.train_keys is None:
        raise ValueError("model carries no cached training predictions")
    out = model.train_keys.copy()
    out["grid_cell_id"] = out.pop("CanOSSEM_RASTER_CELL").astype(str)
    out["date"] = pd.to_datetime(out["date"])
    out["outer_fold"] = -1 if fold is None else int(fold)
    out["pred_stage1"] = model.pred_stage1_train.astype("float64")
    return out[["grid_cell_id", "date", "outer_fold", "pred_stage1"]]


def prediction_table(model: TwoStage, *, fold: int | None = None,
                     fold_label: str | None = None,
                     model_family: str = "lgbm") -> pd.DataFrame:
    """The same holdout table the fold script writes, with the same guards."""
    t = model.test_df
    if t is None:
        raise ValueError("model was not scored on a test set")
    out = pd.DataFrame({
        "grid_cell_id": t["CanOSSEM_RASTER_CELL"].astype(str),
        "date": pd.to_datetime(t["date"]),
        "year": pd.to_numeric(t["year"], errors="raise").astype(int),
        "region": t["fold_region"].astype(str) if "fold_region" in t else "",
        "naps_id": t["naps_id"].astype(str) if "naps_id" in t else "",
        "station_name": t["station_name"].astype(str) if "station_name" in t else "",
        "case_key": t["_source_block"].astype(str),
        "outer_fold": -1 if fold is None else int(fold),
        "fold_label": fold_label or "",
        "model_family": model_family,
        "obs_pm25": model.y_test,
        "pred_stage1": model.pred_stage1,
        "pred_corrector": model.pred_corrector,
        "pred_final": model.pred_final,
        "resid_stage1": model.y_test - model.pred_stage1,
        "resid_final": model.y_test - model.pred_final,
    })
    dup = int(out.duplicated(["grid_cell_id", "date"]).sum())
    if dup:
        raise ValueError(f"prediction table contains {dup} duplicate cell-days")
    if out[["obs_pm25", "pred_stage1", "pred_corrector", "pred_final"]].isna().any().any():
        raise ValueError("NaN in prediction output")
    return out


def save_model(model: TwoStage, out_dir: Path, *, extra: dict | None = None) -> None:
    """Write the two bundles in the same shape the fold script uses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle({"model": model.stage1, "feature_cols": model.feature_cols,
                 "params": STAGE1_PARAMS, "stage": "stage1",
                 "train_rows": model.train_rows, **(extra or {})},
                out_dir / "stage1_model_bundle.pkl")
    save_pickle({"model": model.corrector, "corrector_cols": model.corrector_cols,
                 "fill_values": model.fill_values, "params": CORRECTOR_PARAMS,
                 "stage": "corrector", "pool_diagnostics": model.pool_diag,
                 **(extra or {})},
                out_dir / "corrector_model.pkl")


def load_model(model_dir: Path) -> TwoStage:
    """Reload a saved two-stage model for scoring."""
    import pickle
    with (model_dir / "stage1_model_bundle.pkl").open("rb") as fh:
        sb = pickle.load(fh)
    with (model_dir / "corrector_model.pkl").open("rb") as fh:
        cb = pickle.load(fh)
    return TwoStage(stage1=sb["model"], corrector=cb["model"],
                    feature_cols=list(sb["feature_cols"]),
                    corrector_cols=list(cb["corrector_cols"]),
                    fill_values=dict(cb["fill_values"]),
                    pool_diag=cb.get("pool_diagnostics", {}),
                    train_rows=int(sb.get("train_rows", -1)))
