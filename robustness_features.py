#!/usr/bin/env python3
"""
Feature-group definitions and fold-assignment variants for the robustness suite.

Ported deliberately from `Ontario_RealTarget_GPD/campaign_runner.py` so the ablation
groups and alternative fold assignments are defined EXACTLY as they were for the
thesis table. Re-deriving them would make the new numbers incomparable to the old
ones, which is the whole point of re-running.

The only intended difference between the thesis campaign and this suite is the data:
the campaign read `materialized_support_family_shards_pruned_temporal_x` (before the
438-column rolling-name repair), this suite reads `Data` (raw shards: the corrector must see real NaN).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- groups

def feature_tokens(name: str) -> set[str]:
    """Which ablation groups a predictor belongs to. Verbatim from campaign_runner."""
    n = name.lower()
    toks: set[str] = set()
    if "aod" in n:
        toks.add("aod")
    if n.startswith("src_merra"):
        toks.add("merra")
    if n.startswith("src_viirs") or n.startswith("src_hms"):
        toks.add("fire")
    if "burned" in n:
        toks.add("burned")
        toks.add("fire")          # burned area counts as a fire input
    return toks


def select_features(all_feats: list[str], drop=None, keep_mode: str | None = None) -> list[str]:
    """Feature subset after dropping token groups, or keeping a named subset.

    `met_only` keeps NARR meteorology + calendar + static land/road, i.e. drops every
    satellite and reanalysis input.
    """
    drop = set(drop or [])
    if keep_mode == "met_only":
        drop = drop | {"aod", "merra", "fire", "burned"}
    return [f for f in all_feats if not (feature_tokens(f) & drop)]


def group_sizes(all_feats: list[str]) -> dict[str, int]:
    """How many predictors each ablation removes -- reported so a 'no effect' result
    can be distinguished from 'nothing was actually dropped'."""
    out = {}
    for tok in ("aod", "merra", "fire", "burned"):
        out[tok] = sum(1 for f in all_feats if tok in feature_tokens(f))
    out["met_only_kept"] = len(select_features(all_feats, keep_mode="met_only"))
    out["total"] = len(all_feats)
    return out


# --------------------------------------------------------------------------- folds

def balanced_assignment(cases: list[dict], n_groups: int, seed: int) -> dict[str, int]:
    """An alternative valid block -> group assignment.

    Shuffle, then greedily fill the least-loaded group by row count. Whole blocks
    only, so the leave-one-block-out property is preserved; only which blocks share
    a fold changes. Verbatim from campaign_runner.
    """
    rng = np.random.default_rng(seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    order.sort(key=lambda i: -cases[i]["rows"])
    loads = [0] * n_groups
    assign: dict[str, int] = {}
    for i in order:
        g = int(np.argmin(loads))
        assign[cases[i]["case_key"]] = g + 1
        loads[g] += cases[i]["rows"]
    return assign


def spatial_fold_assignment(cell_region: dict[str, str], k: int = 7,
                            seed: int = 2026) -> dict[str, int]:
    """Leave-cells-out: split grid cells into k folds, round-robin within region.

    A held-out cell is entirely absent from its fold's training set, which is the
    property the block design does NOT have. Verbatim from campaign_runner.run_spatial_cv.
    """
    by_region: dict[str, list[str]] = {}
    for c, r in sorted(cell_region.items()):
        by_region.setdefault(r, []).append(c)
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for r in sorted(by_region):
        cs = sorted(by_region[r])
        idx = list(range(len(cs)))
        rng.shuffle(idx)
        for pos, ci in enumerate(idx):
            fold_of[cs[ci]] = pos % k
    return fold_of


# --------------------------------------------------------------------------- metrics

def kloog_metrics(obs, pred, cell) -> dict:
    """The thesis metric set. `spatial_r2` and `temporal_r2` decompose pooled R^2 into
    between-cell and within-cell components (Kloog et al. convention).

    R^2 is predictive (1 - SSE/SST), never squared correlation.
    """
    import pandas as pd

    df = pd.DataFrame({"obs": np.asarray(obs, "float64"),
                       "pred": np.asarray(pred, "float64"),
                       "cell": np.asarray(cell)})
    df = df.dropna(subset=["pred"])

    def r2(y, p):
        y = np.asarray(y, "float64"); p = np.asarray(p, "float64")
        sst = float(((y - y.mean()) ** 2).sum())
        return float("nan") if sst == 0 else 1.0 - float(((y - p) ** 2).sum()) / sst

    cm = df.groupby("cell")[["obs", "pred"]].mean()
    oa = df["obs"] - df.groupby("cell")["obs"].transform("mean")
    pa = df["pred"] - df.groupby("cell")["pred"].transform("mean")
    per_cell = [r2(g["obs"], g["pred"]) for _, g in df.groupby("cell") if g["obs"].nunique() > 1]

    err = df["pred"].to_numpy() - df["obs"].to_numpy()
    return {
        "n": int(len(df)),
        "n_cells": int(df["cell"].nunique()),
        "pooled_r2": r2(df["obs"], df["pred"]),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "within5_pct": float(100 * np.mean(np.abs(err) <= 5)),
        "macro_r2": float(np.mean(per_cell)) if per_cell else float("nan"),
        "spatial_r2": r2(cm["obs"], cm["pred"]),
        "temporal_r2": r2(oa, pa),
    }
