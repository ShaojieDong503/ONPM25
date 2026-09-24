#!/usr/bin/env python3
"""Fold-local LightGBM AOD gap-filler.

Why this exists
---------------
`AOD_055_filled` in the shards was produced by a model trained on the WHOLE panel,
including the rows that later become a fold's held-out set. Every fold therefore
inherits an imputation that has seen its test data. The effect is indirect -- the
filler predicts AOD, not PM2.5 -- but it is a real dependence, and spec G names it:

    "Do not silently redesign the AOD imputation procedure to make it fold-specific
     unless explicitly instructed in a later revision."

This is that revision, and it is opt-in: `--aod-fill fold` on the fold script. The
default stays `global`, so the thesis numbers remain reproducible.

What it does, per fold
----------------------
1. fit a LightGBM regressor on TRAINING rows where AOD_055 was actually retrieved
2. predict AOD for every row missing it, in the training AND held-out frames
3. rebuild AOD_055_filled = observed where observed, predicted where not
4. recompute the 7 temporal derivatives from the new series, since they would
   otherwise still carry the global fill and the leak would persist through them

The held-out frame is only ever PREDICTED by this model, never fitted on -- same
discipline as Stage 1.

Two honest limitations
----------------------
* The original filler's strongest predictors, the spatial neighbour-AOD aggregates
  (raw_aod055_nbr_count / mean / max), are NOT in these shards. A fold-local filler
  cannot use them, so it is not a like-for-like reimplementation of the original --
  it is a different, weaker filler trained on less. MERRA-2's aerosol optical-depth
  fields partly compensate, being physical analogues of the quantity being imputed.
* AOD_047 is 85.8% missing and is not refilled here. It stays as it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

CELL = "CanOSSEM_RASTER_CELL"
DATE = "date"
RAW = "AOD_055"
FILLED = "AOD_055_filled"
OBS_FLAG = "aod_obs_flag"
IMP_FLAG = "aod_imputed_flag"

# The 7 derivatives of FILLED, in their STORED spelling. The builder rolls the lagged
# series, so the stored name carries an explicit _lag1_ infix.
DERIVED = {
    "lag1": f"{FILLED}_lag1",
    ("roll3", "min"): f"{FILLED}_lag1_roll3_min",
    ("roll3", "mean"): f"{FILLED}_lag1_roll3_mean",
    ("roll3", "max"): f"{FILLED}_lag1_roll3_max",
    ("roll7", "min"): f"{FILLED}_lag1_roll7_min",
    ("roll7", "mean"): f"{FILLED}_lag1_roll7_mean",
    ("roll7", "max"): f"{FILLED}_lag1_roll7_max",
}

# NARR meteorology that survives into the shards, plus the calendar term. The original
# filler also used albedo/hcdc/doy_sin/doy_cos and the neighbour-AOD aggregates; none
# of those are present here.
NARR_PREDICTORS = ["acpcp", "air_sfc", "dswrf", "evap", "hpbl", "lcdc", "mcdc",
                   "pr_wtr", "pres_sfc", "shum_2m", "vis", "uwnd_10m", "vwnd_10m",
                   "windspeed_10m", "dayofyear"]

FILLER_PARAMS = dict(
    objective="regression", n_estimators=600, learning_rate=0.05, num_leaves=127,
    min_child_samples=40, subsample=0.9, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, max_bin=127, force_col_wise=True,
    random_state=2026, n_jobs=-1, verbosity=-1,
)


def filler_predictors(df: pd.DataFrame, max_merra: int | None = None) -> list[str]:
    """NARR meteorology + calendar + MERRA-2 aerosol/meteorology present in the frame.

    MERRA's optical-depth fields (TOTEXTTAU, BCEXTTAU, SUEXTTAU) are reanalysis
    estimates of the same physical quantity MAIAC retrieves, which is why they are
    included: they are the closest available stand-in for the spatial neighbour-AOD
    features the original filler had and these shards do not.

    Excludes every AOD column, so the filler cannot see the target or its derivatives.
    """
    cols = [c for c in NARR_PREDICTORS if c in df.columns]
    merra = [c for c in df.columns
             if c.startswith("src_merra") and "aod" not in c.lower()]
    if max_merra is not None:
        merra = merra[:max_merra]
    banned = {RAW, FILLED, OBS_FLAG, IMP_FLAG, "AOD_047", *DERIVED.values()}
    return [c for c in cols + merra if c not in banned]


def fit_filler(train_df: pd.DataFrame, predictors: list[str], logger=None):
    """Fit on TRAINING rows where AOD_055 was genuinely retrieved."""
    y = pd.to_numeric(train_df[RAW], errors="coerce")
    obs = y.notna().to_numpy()
    if obs.sum() < 1000:
        raise ValueError(f"only {int(obs.sum())} observed AOD rows in training; "
                         f"too few to fit a filler")
    X = np.nan_to_num(train_df.loc[obs, predictors].to_numpy("float32"), nan=0.0)
    model = LGBMRegressor(**FILLER_PARAMS).fit(X, y[obs].to_numpy("float64"))
    if logger:
        logger.info("AOD FILLER | fit on %d observed of %d training rows | %d predictors",
                    int(obs.sum()), len(train_df), len(predictors))
    return model


def _recompute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the 7 temporal features from the refilled series.

    Reproduces the shard builder's convention exactly: sort by (cell, date), take the
    within-cell lag-1, then roll THAT lagged series with min_periods=1. Rolling the
    unlagged series would leak the current day into its own predictor.
    """
    order = df.index.to_numpy()
    d = df.sort_values([CELL, DATE], kind="mergesort")
    lag1 = d.groupby(CELL, sort=False)[FILLED].shift(1)
    out = {DERIVED["lag1"]: lag1}
    g = lag1.groupby(d[CELL], sort=False)
    for w, wname in ((3, "roll3"), (7, "roll7")):
        r = g.rolling(w, min_periods=1)
        for stat in ("min", "mean", "max"):
            out[DERIVED[(wname, stat)]] = getattr(r, stat)().reset_index(level=0, drop=True)
    for name, series in out.items():
        if name in df.columns:
            d[name] = series.astype("float32")
    return d.loc[order]


def apply_filler(df: pd.DataFrame, model, predictors: list[str],
                 logger=None, label: str = "") -> tuple[pd.DataFrame, dict]:
    """Replace FILLED with observed-where-observed, fold-model-predicted elsewhere.

    Returns (frame, diagnostics). The frame is a copy; the caller's is untouched.
    """
    out = df.copy()
    raw = pd.to_numeric(out[RAW], errors="coerce")
    miss = raw.isna().to_numpy()
    filled = raw.to_numpy("float64").copy()

    if miss.any():
        X = np.nan_to_num(out.loc[miss, predictors].to_numpy("float32"), nan=0.0)
        pred = np.asarray(model.predict(X), dtype="float64")
        # MAIAC AOD is non-negative and the retrieval saturates around 5; keep the
        # imputation inside the range the target can actually take.
        pred = np.clip(pred, 0.0, 5.0)
        filled[miss] = pred

    out[FILLED] = filled.astype("float32")
    # The flags describe the RAW retrieval, so they are recomputed from it rather
    # than carried over -- they must agree with the new fill, not the old one.
    if OBS_FLAG in out.columns:
        out[OBS_FLAG] = (~miss).astype("float32")
    if IMP_FLAG in out.columns:
        out[IMP_FLAG] = miss.astype("float32")
    out = _recompute_derivatives(out)

    # observed values must survive untouched: the fill replaces absence, nothing else
    kept = raw.notna().to_numpy()
    if kept.any():
        worst = float(np.abs(out.loc[kept, FILLED].to_numpy("float64")
                             - raw[kept].to_numpy("float64")).max())
        if worst > 1e-5:
            raise AssertionError(f"{label}: fill altered {int(kept.sum())} observed "
                                 f"AOD values (max |diff| {worst:.3e})")
    diag = {"rows": int(len(out)), "observed": int(kept.sum()), "imputed": int(miss.sum()),
            "imputed_fraction": float(miss.mean()),
            "imputed_mean": float(filled[miss].mean()) if miss.any() else float("nan"),
            "observed_mean": float(raw[kept].mean()) if kept.any() else float("nan")}
    if logger:
        logger.info("AOD FILLER | %s | imputed %d of %d rows (%.1f%%) | "
                    "imputed mean %.4f vs observed %.4f", label, diag["imputed"],
                    diag["rows"], 100 * diag["imputed_fraction"],
                    diag["imputed_mean"], diag["observed_mean"])
    return out, diag


def refill_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, logger=None
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The whole fold-local procedure. Fit on training only, apply to both."""
    predictors = filler_predictors(train_df)
    model = fit_filler(train_df, predictors, logger)
    tr, dtr = apply_filler(train_df, model, predictors, logger, "train")
    te, dte = apply_filler(test_df, model, predictors, logger, "holdout")
    return tr, te, {"predictor_count": len(predictors), "params": FILLER_PARAMS,
                    "train": dtr, "holdout": dte,
                    "note": "fitted on training rows with an observed AOD_055 only; "
                            "the held-out frame is predicted, never fitted on"}


# =============================================================================
# Merged single-column AOD design
# =============================================================================
# The shards carry five AOD base columns (AOD_055, AOD_047, AOD_055_filled,
# aod_obs_flag, aod_imputed_flag) plus seven derivatives of AOD_055_filled. Three of
# those are redundant or harmful:
#
#   AOD_055_filled    imputed once over the WHOLE panel, so every fold inherits a
#                     fill that saw its own held-out rows
#   aod_imputed_flag  identical to 1 - aod_obs_flag, verified to 1.000000
#   AOD_047           85.8% missing, r=0.9995 with AOD_055, no filled counterpart
#
# The merged design keeps one retrieval column and one indicator:
#
#   aod           = AOD_055 where retrieved, else AOD_047, then FOLD-LOCAL filled
#   aod_obs_flag  = 1 where either band was retrieved
#
# Measured: the coalesce gains 2 rows of coverage out of 169,882 (89,891 -> 89,893),
# so this is a structural fix, not a coverage one. It does make the flag's name
# finally accurate -- the shipped aod_obs_flag means "AOD_055 retrieved", which
# differs from "either band retrieved" on exactly those 2 rows.
#
# 12 AOD features -> 9, so Stage 1 goes 651 -> 648.

MERGED = "aod"
MERGED_DERIVED = {
    "lag1": f"{MERGED}_lag1",
    ("roll3", "min"): f"{MERGED}_roll3_min",
    ("roll3", "mean"): f"{MERGED}_roll3_mean",
    ("roll3", "max"): f"{MERGED}_roll3_max",
    ("roll7", "min"): f"{MERGED}_roll7_min",
    ("roll7", "mean"): f"{MERGED}_roll7_mean",
    ("roll7", "max"): f"{MERGED}_roll7_max",
}
# Dropped from the Stage-1 feature list under the merged design.
#
# DERIVED holds the STORED spellings (with the _lag1_ infix); the manifest lists the
# CANONICAL ones without it. Filtering the canonical list with stored names silently
# matches nothing -- the same confusion that once replaced 438 real columns with
# zeros -- so both spellings are named here and the count is asserted below.
CANONICAL_DERIVED = [f"{FILLED}_lag1"] + [
    f"{FILLED}_{w}_{s}" for w in ("roll3", "roll7") for s in ("min", "mean", "max")
]
MERGED_DROP = [RAW, "AOD_047", FILLED, IMP_FLAG,
               *CANONICAL_DERIVED, *DERIVED.values()]


def merged_feature_cols(canonical: list[str]) -> list[str]:
    """The Stage-1 predictor list under the merged design: 651 -> 648.

    Order is preserved except that the AOD block collapses in place, so the matrix
    stays comparable to the thesis one column-for-column outside AOD.
    """
    drop = set(MERGED_DROP)
    out, inserted = [], False
    for c in canonical:
        if c == OBS_FLAG:                      # anchor the new block at the flag
            if not inserted:
                out.extend([MERGED, OBS_FLAG, *_merged_derived_order()])
                inserted = True
            continue
        if c in drop:
            continue
        out.append(c)
    if not inserted:
        raise ValueError(f"{OBS_FLAG} not found in the canonical feature list")
    # 12 AOD features in, 9 out. Asserted rather than assumed: a spelling that fails
    # to match would silently leave the old columns in and change nothing else.
    expected = len(canonical) - 12 + 9
    if len(out) != expected:
        stale = [c for c in out if c.startswith(FILLED) or c in (RAW, "AOD_047", IMP_FLAG)]
        raise AssertionError(
            f"merged feature list is {len(out)}, expected {expected}; "
            f"{len(stale)} old AOD column(s) survived: {stale[:6]}")
    return out


def _merged_derived_order() -> list[str]:
    return [MERGED_DERIVED["lag1"],
            *(MERGED_DERIVED[(w, s)] for w in ("roll3", "roll7")
              for s in ("min", "mean", "max"))]


def build_merged_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce the two bands into `aod` and set the indicator from the result."""
    out = df.copy()
    a55 = pd.to_numeric(out[RAW], errors="coerce")
    a47 = pd.to_numeric(out["AOD_047"], errors="coerce") if "AOD_047" in out.columns else None
    merged = a55 if a47 is None else a55.fillna(a47)
    out[MERGED] = merged.astype("float32")
    out[OBS_FLAG] = merged.notna().astype("float32")
    return out


def _recompute_merged_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Same convention as the shard builder: within-cell lag-1, then roll the LAGGED
    series. The derived names carry no _lag1_ infix here because these columns are
    created by this module, not read from the parquet."""
    order = df.index.to_numpy()
    d = df.sort_values([CELL, DATE], kind="mergesort")
    lag1 = d.groupby(CELL, sort=False)[MERGED].shift(1)
    d[MERGED_DERIVED["lag1"]] = lag1.astype("float32")
    g = lag1.groupby(d[CELL], sort=False)
    for w, wname in ((3, "roll3"), (7, "roll7")):
        r = g.rolling(w, min_periods=1)
        for stat in ("min", "mean", "max"):
            d[MERGED_DERIVED[(wname, stat)]] = (
                getattr(r, stat)().reset_index(level=0, drop=True).astype("float32"))
    return d.loc[order]


def refill_fold_merged(train_df: pd.DataFrame, test_df: pd.DataFrame, logger=None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Merged-column design with a fold-local fill. Fit on training only."""
    tr = build_merged_column(train_df)
    te = build_merged_column(test_df)

    predictors = [c for c in filler_predictors(tr) if c not in {MERGED, *MERGED_DROP}]
    y = pd.to_numeric(tr[MERGED], errors="coerce")
    obs = y.notna().to_numpy()
    if obs.sum() < 1000:
        raise ValueError(f"only {int(obs.sum())} observed AOD rows in training")
    X = np.nan_to_num(tr.loc[obs, predictors].to_numpy("float32"), nan=0.0)
    model = LGBMRegressor(**FILLER_PARAMS).fit(X, y[obs].to_numpy("float64"))
    if logger:
        logger.info("AOD MERGED FILLER | fit on %d observed of %d training rows | "
                    "%d predictors", int(obs.sum()), len(tr), len(predictors))

    diags = {}
    for name, frame in (("train", tr), ("holdout", te)):
        v = pd.to_numeric(frame[MERGED], errors="coerce")
        miss = v.isna().to_numpy()
        filled = v.to_numpy("float64").copy()
        if miss.any():
            Xm = np.nan_to_num(frame.loc[miss, predictors].to_numpy("float32"), nan=0.0)
            filled[miss] = np.clip(model.predict(Xm), 0.0, 5.0)
        frame[MERGED] = filled.astype("float32")
        kept = v.notna().to_numpy()
        if kept.any():
            worst = float(np.abs(filled[kept] - v[kept].to_numpy("float64")).max())
            if worst > 1e-5:
                raise AssertionError(f"{name}: fill altered observed AOD "
                                     f"(max |diff| {worst:.3e})")
        diags[name] = {"rows": int(len(frame)), "observed": int(kept.sum()),
                       "imputed": int(miss.sum()),
                       "imputed_fraction": float(miss.mean())}
        if logger:
            logger.info("AOD MERGED | %s | imputed %d of %d (%.1f%%)", name,
                        diags[name]["imputed"], diags[name]["rows"],
                        100 * diags[name]["imputed_fraction"])

    tr = _recompute_merged_derivatives(tr)
    te = _recompute_merged_derivatives(te)
    return tr, te, {"mode": "merged", "predictor_count": len(predictors),
                    "params": FILLER_PARAMS, **diags,
                    "note": "aod = AOD_055 else AOD_047, filled by a model fitted on "
                            "this fold's training rows only; AOD_055_filled and "
                            "aod_imputed_flag are discarded"}
