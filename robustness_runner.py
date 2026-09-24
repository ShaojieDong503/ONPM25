#!/usr/bin/env python3
"""
Robustness / ablation suite. This file's ONLY job is to assign the data.

Every experiment runs the same two-stage fit as `run_lgbm_thesis_fold.py`, through
`thesis_core` -- same loaders, hyperparameters, corrector pool rule, fill values,
metrics and seed 2026. What changes per experiment is:

    * which FEATURES go in                    (ablations)
    * which BLOCKS are held out in each fold  (thesis folds vs alternatives)
    * which ROWS are held out                 (spatial CV, LOCO -- their own designs)

The primary folds are the thesis folds, read from `case_plans/GROUP_0N.json` by the
fold script's own `load_fold_plan`.

    python robustness_runner.py --list
    python robustness_runner.py --experiment baseline
    python robustness_runner.py --experiment ablation_no_fire --fold 3   # fan-out
    python robustness_runner.py --experiment loco --fold 0               # one cell
    python robustness_runner.py --merge loco
    python robustness_runner.py --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import thesis_core as TC  # noqa: E402

# LightGBM's n_jobs overrides OMP_NUM_THREADS, so when several experiments run
# concurrently each one would otherwise spawn a thread per core. At 4x
# oversubscription the OpenMP barriers spin instead of working and a 9-minute fit
# can stall for hours. Set the estimator's own thread count.
_LGBM_TREES = os.environ.get("PM25_LGBM_ESTIMATORS")
if _LGBM_TREES:                      # plumbing tests only, never scientific output
    TC.STAGE1_PARAMS["n_estimators"] = int(_LGBM_TREES)
    TC.CORRECTOR_PARAMS["n_estimators"] = int(_LGBM_TREES)

_LGBM_THREADS = os.environ.get("PM25_LGBM_THREADS")
if _LGBM_THREADS:
    n = max(1, int(_LGBM_THREADS))
    TC.STAGE1_PARAMS["n_jobs"] = n
    TC.CORRECTOR_PARAMS["n_jobs"] = n

from robustness_features import (  # noqa: E402
    balanced_assignment, group_sizes, kloog_metrics, select_features,
    spatial_fold_assignment,
)

N_GROUPS = 8

EXPERIMENTS = {
    "baseline":           {"kind": "block", "drop": [],         "keep": None,       "assign": "thesis"},
    "ablation_no_aod":    {"kind": "block", "drop": ["aod"],    "keep": None,       "assign": "thesis"},
    "ablation_no_fire":   {"kind": "block", "drop": ["fire"],   "keep": None,       "assign": "thesis"},
    "ablation_no_merra":  {"kind": "block", "drop": ["merra"],  "keep": None,       "assign": "thesis"},
    "ablation_met_only":  {"kind": "block", "drop": [],         "keep": "met_only", "assign": "thesis"},
    "ablation_no_burned": {"kind": "block", "drop": ["burned"], "keep": None,       "assign": "thesis"},
    "foldalt_1":          {"kind": "block", "drop": [], "keep": None, "assign": "alt", "seed": 101},
    "foldalt_2":          {"kind": "block", "drop": [], "keep": None, "assign": "alt", "seed": 202},
    "foldalt_3":          {"kind": "block", "drop": [], "keep": None, "assign": "alt", "seed": 303},
    "foldalt_4":          {"kind": "block", "drop": [], "keep": None, "assign": "alt", "seed": 404},
    "oof_corrector":      {"kind": "oof_corrector", "drop": [], "keep": None, "assign": "thesis"},
    "spatial_cv":         {"kind": "spatial_cv", "drop": [], "keep": None, "k": 7},
    # "loco" (leave-ONE-cell-out, 41 fits, ~6.5 h) is intentionally NOT registered.
    # Spatial generalisation is covered by spatial_cv (leave-cells-out, k=7), which
    # answers the same question at 1/9 the cost. run_loco() below is left in place so
    # the design can be reinstated by restoring this line.
}
FOLDALT = [f"foldalt_{i}" for i in range(1, 5)]


# --------------------------------------------------------------------------- fold plans

def thesis_folds(case_plans: Path) -> dict[int, dict]:
    """The thesis folds, via the fold script's own plan loader."""
    out = {}
    for g in range(1, N_GROUPS + 1):
        plan, _ = TC.load_fold_plan(case_plans, g)
        out[g] = {"train": list(plan["train_pair_blocks"]),
                  "test": list(plan["heldout_case_keys"])}
    return out


def block_rows(shard_root: Path, blocks: list[str]) -> dict[str, int]:
    import pyarrow.parquet as pq
    return {b: pq.ParquetFile(TC.pair_block_path(shard_root, b)).metadata.num_rows
            for b in blocks}


def alt_folds(shard_root: Path, case_plans: Path, seed: int) -> dict[int, dict]:
    """Alternative block -> group assignment: whole blocks, row-balanced.

    Leave-one-block-out is preserved; only which blocks share a fold changes.
    """
    blocks = sorted({b for f in thesis_folds(case_plans).values() for b in f["test"]})
    counts = block_rows(shard_root, blocks)
    assign = balanced_assignment([{"case_key": b, "rows": counts[b]} for b in blocks],
                                 N_GROUPS, seed)
    out = {}
    for g in range(1, N_GROUPS + 1):
        test = sorted(b for b, gg in assign.items() if gg == g)
        out[g] = {"train": [b for b in blocks if b not in set(test)], "test": test}
    return out


# --------------------------------------------------------------------------- experiments

def run_block(args, spec, logger) -> pd.DataFrame:
    """8 folds, each a full two-stage fit through the fold script's code path."""
    feats_all = TC.canonical_features(args.shard_root)
    feats = select_features(feats_all, drop=spec.get("drop"), keep_mode=spec.get("keep"))
    logger.info("features: %d of %d", len(feats), len(feats_all))

    folds = (alt_folds(args.shard_root, args.case_plans_dir, spec["seed"])
             if spec.get("assign") == "alt" else thesis_folds(args.case_plans_dir))

    todo = [args.fold] if args.fold else sorted(folds)
    parts = []
    # Stage-1 training predictions are cached alongside the held-out table so that a
    # later change to corrector handling can be answered by refitting correctors only.
    # Stage 1 is the expensive half and is unaffected by anything the corrector does.
    stage1_train_parts: list = []
    for g in todo:
        f = folds[g]
        logger.info("FOLD %d | train=%d test=%d", g, len(f["train"]), len(f["test"]))
        t0 = time.time()
        m = TC.fit_and_score(shard_root=args.shard_root, train_blocks=f["train"],
                             test_blocks=f["test"], feature_cols=feats, logger=logger)
        parts.append(TC.prediction_table(m, fold=g, fold_label=f"GROUP_{g:02d}"))
        stage1_train_parts.append(TC.stage1_train_table(m, fold=g))
        logger.info("  fold %d done in %.0fs | final rmse=%.4f",
                    g, time.time() - t0, m.final_metrics["rmse"])
    out = pd.concat(parts, ignore_index=True)
    if stage1_train_parts:
        out.attrs["stage1_train"] = pd.concat(stage1_train_parts, ignore_index=True)
    return out


def run_spatial_cv(args, spec, logger) -> pd.DataFrame:
    """Leave-cells-out, k folds. A held-out cell is absent from its fold's training."""
    feats = TC.canonical_features(args.shard_root)
    blocks = sorted(p.name for p in (args.shard_root / "pair_blocks").iterdir() if p.is_dir())
    df, X, _ = TC.load_blocks(args.shard_root, blocks, feats, logger)
    cell = df["CanOSSEM_RASTER_CELL"].astype(str).to_numpy()
    region = df["fold_region"].astype(str).to_numpy()

    cell_region = {}
    for c, r in zip(cell, region):
        cell_region.setdefault(c, r)
    k = int(spec.get("k", 7))
    fold_of = spatial_fold_assignment(cell_region, k=k,
                                      seed=TC.STAGE1_PARAMS["random_state"])
    cf = np.array([fold_of[c] for c in cell])
    todo = [args.fold] if args.fold is not None else list(range(k))
    return _row_mask_folds(df, X, cf, todo, feats, logger, label="spatial_cv")


def run_loco(args, spec, logger) -> pd.DataFrame:
    """Leave-ONE-cell-out: its own design, one fold per monitored cell."""
    feats = TC.canonical_features(args.shard_root)
    blocks = sorted(p.name for p in (args.shard_root / "pair_blocks").iterdir() if p.is_dir())
    df, X, _ = TC.load_blocks(args.shard_root, blocks, feats, logger)
    cell = df["CanOSSEM_RASTER_CELL"].astype(str).to_numpy()
    cells = sorted(set(cell))
    idx = np.array([cells.index(c) for c in cell])
    todo = [args.fold] if args.fold is not None else list(range(len(cells)))
    logger.info("LOCO | %d cells, running %d", len(cells), len(todo))
    return _row_mask_folds(df, X, idx, todo, feats, logger, label="loco",
                           names={i: c for i, c in enumerate(cells)})


def _row_mask_folds(df, X, fold_ids, todo, feats, logger, *, label, names=None) -> pd.DataFrame:
    """Driver for row-mask designs.

    Held-out units are rows, not blocks, so this calls thesis_core.fit_two_stage /
    score directly -- still the fold script's fit, just a different row selection.
    """
    parts = []
    for f in todo:
        te = fold_ids == f
        if not te.any():
            continue
        t0 = time.time()
        tr_df, te_df = df.loc[~te].copy(), df.loc[te].copy()
        model = TC.fit_two_stage(X_train=X[~te], train_df=tr_df,
                                 feature_cols=feats, logger=logger)
        p1, pc, pf = TC.score(model, X[te], te_df, logger)
        y = pd.to_numeric(te_df["pm25"], errors="raise").to_numpy("float64")
        parts.append(pd.DataFrame({
            "grid_cell_id": te_df["CanOSSEM_RASTER_CELL"].astype(str),
            "date": pd.to_datetime(te_df["date"]),
            "year": pd.to_numeric(te_df["year"], errors="raise").astype(int),
            "region": te_df["fold_region"].astype(str),
            "case_key": te_df["_source_block"].astype(str),
            "outer_fold": int(f),
            "fold_label": f"{label}_{names[f] if names else f}",
            "model_family": "lgbm", "obs_pm25": y,
            "pred_stage1": p1, "pred_corrector": pc, "pred_final": pf,
        }))
        logger.info("  %s fold %s: n=%d rmse=%.4f (%.0fs)", label,
                    names[f] if names else f, int(te.sum()),
                    TC.metrics(y, pf)["rmse"], time.time() - t0)
    return pd.concat(parts, ignore_index=True)


def run_oof_corrector(args, spec, logger) -> pd.DataFrame:
    """Corrector fitted on OUT-OF-FOLD Stage-1 residuals (nested holdout).

    The baseline corrector learns from IN-SAMPLE residuals, which are optimistically
    small because Stage 1 has already seen those rows. This variant asks what happens
    when the corrector is shown honest residuals instead.

    Per outer group g:
      1. split the 7 training groups; predict each with a Stage 1 trained on the
         other 6  -> out-of-fold residuals over all 63 training blocks   (7 fits)
      2. fit Stage 1 on all 63 training blocks, apply to g               (1 fit)
      3. fit the corrector on the OOF residuals, apply to g
    8 fits per group, 64 in total.
    """
    feats = TC.canonical_features(args.shard_root)
    folds = thesis_folds(args.case_plans_dir)

    # One load of everything; blocks are selected by mask per fold.
    all_blocks = sorted({b for f in folds.values() for b in f["test"]})
    df, X, _ = TC.load_blocks(args.shard_root, all_blocks, feats, logger)
    block = df["_source_block"].astype(str).to_numpy()
    y = pd.to_numeric(df["pm25"], errors="raise").to_numpy("float64")

    todo = [args.fold] if args.fold else sorted(folds)
    parts = []
    for g in todo:
        t0 = time.time()
        test_b = set(folds[g]["test"])
        te = np.isin(block, list(test_b))
        tr = ~te
        logger.info("OOF FOLD %d | train rows=%d test rows=%d", g, int(tr.sum()), int(te.sum()))

        # 1. out-of-fold Stage-1 predictions across the 7 training groups
        oof = np.full(int(tr.sum()), np.nan)
        tr_block = block[tr]
        X_tr, y_tr = X[tr], y[tr]
        inner = [h for h in sorted(folds) if h != g]
        for h in inner:
            ite = np.isin(tr_block, list(set(folds[h]["test"])))
            if not ite.any():
                continue
            m = TC.fit_stage1(X_tr[~ite], y_tr[~ite], feats)
            oof[ite] = m.predict(X_tr[ite]).astype("float64")
            logger.info("  inner group %d: %d rows scored out-of-fold", h, int(ite.sum()))
        if np.isnan(oof).any():
            raise AssertionError(f"fold {g}: {int(np.isnan(oof).sum())} training rows "
                                 f"never received an out-of-fold prediction")

        # 2. Stage 1 on all 63 training blocks -> the held-out group
        stage1 = TC.fit_stage1(X_tr, y_tr, feats)
        p1_te = stage1.predict(X[te]).astype("float64")

        # 3. corrector on the OOF residuals. Pool rule, fill values and transform are
        #    the fold script's; only the residual definition differs from baseline.
        pool_src = df.loc[tr].copy()
        pool_src["pred_stage1"] = oof
        pool_src["resid_stage1"] = y_tr - oof
        TC.F.validate_corrector_columns(pool_src)
        pool_df, _mask, diag = TC.F.build_corrector_pool(pool_src, logger)
        fill = TC.F.fit_corrector_fill_values(pool_df)
        Xc_tr, cols = TC.F.transform_corrector(pool_df, fill)
        y_corr = pd.to_numeric(pool_df["resid_stage1"], errors="raise").to_numpy("float64")

        from lightgbm import LGBMRegressor
        corr = LGBMRegressor(**TC.CORRECTOR_PARAMS)
        corr.fit(Xc_tr, y_corr, feature_name=cols)

        te_df = df.loc[te].copy()
        te_df["pred_stage1"] = p1_te
        TC.F.validate_corrector_columns(te_df)
        Xc_te, cols_te = TC.F.transform_corrector(te_df, fill)
        if cols_te != cols:
            raise AssertionError("corrector feature order differs between fit and score")
        pc = corr.predict(Xc_te).astype("float64")

        parts.append(pd.DataFrame({
            "grid_cell_id": te_df["CanOSSEM_RASTER_CELL"].astype(str),
            "date": pd.to_datetime(te_df["date"]),
            "year": pd.to_numeric(te_df["year"], errors="raise").astype(int),
            "region": te_df["fold_region"].astype(str),
            "case_key": te_df["_source_block"].astype(str),
            "outer_fold": int(g), "fold_label": f"GROUP_{g:02d}",
            "model_family": "lgbm", "obs_pm25": y[te],
            "pred_stage1": p1_te, "pred_corrector": pc, "pred_final": p1_te + pc,
        }))
        logger.info("  fold %d done in %.0fs | pool %d/%d | final rmse=%.4f",
                    g, time.time() - t0, len(pool_df), int(tr.sum()),
                    TC.metrics(y[te], p1_te + pc)["rmse"])
    return pd.concat(parts, ignore_index=True)


RUNNERS = {"block": run_block, "spatial_cv": run_spatial_cv, "loco": run_loco,
           "oof_corrector": run_oof_corrector}


# --------------------------------------------------------------------------- summary

def summarize(out_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out_dir.glob("*_metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("fold_subset") is not None:
            continue          # partial fan-out; merged separately
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    base = df[df.experiment == "baseline"]
    has_base = len(base) > 0
    df["delta_r2"] = df.pooled_r2 - (float(base.pooled_r2.iloc[0]) if has_base else np.nan)

    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    df = df.sort_values("experiment", key=lambda s: s.map(order))
    df.to_csv(out_dir / "robustness_summary.csv", index=False)

    lines = ["# Robustness / ablation results\n",
             "Two-stage (Stage 1 + corrector) via the run_lgbm_thesis_fold code path. "
             "R^2 is predictive (1 - SSE/SST).\n",
             "| Experiment | features | Pooled R2 | dR2 | RMSE | Spatial R2 |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        d = "-" if (r.experiment == "baseline" or not has_base) else f"{r.delta_r2:+.3f}"
        lines.append(f"| {r.experiment} | {int(r.n_features)} | {r.pooled_r2:.3f} | {d} "
                     f"| {r.rmse:.2f} | {r.spatial_r2:.3f} |")
    alt = df[df.experiment.isin(FOLDALT)]
    if len(alt) > 1:
        lines += ["", f"**Alternative fold assignments (x{len(alt)}):** pooled R2 "
                      f"{alt.pooled_r2.min():.3f}-{alt.pooled_r2.max():.3f}, "
                      f"RMSE {alt.rmse.min():.2f}-{alt.rmse.max():.2f}"]
    (out_dir / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def merge_partials(name: str, out_dir: Path, logger) -> dict | None:
    parts = sorted(out_dir.glob(f"{name}_fold*_predictions.parquet"))
    if not parts:
        logger.error("no partials matching %s_fold*_predictions.parquet", name)
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dup = int(df.duplicated(subset=["grid_cell_id", "date"]).sum())
    if dup:
        raise RuntimeError(f"{name}: {dup} cell-days predicted by more than one partial")
    metas = [json.loads(p.with_name(p.name.replace("_predictions.parquet", "_metrics.json"))
                        .read_text(encoding="utf-8")) for p in parts
             if p.with_name(p.name.replace("_predictions.parquet", "_metrics.json")).exists()]
    df.to_parquet(out_dir / f"{name}_predictions.parquet", index=False)
    m = kloog_metrics(df["obs_pm25"], df["pred_final"], df["grid_cell_id"])
    m.update({"experiment": name, "kind": EXPERIMENTS[name]["kind"],
              "n_features": metas[0]["n_features"] if metas else -1,
              "fold_subset": None, "merged_from": len(parts)})
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    logger.info("[merge] %s: %d partials -> %d rows | pooled_r2=%.3f rmse=%.2f",
                name, len(parts), len(df), m["pooled_r2"], m["rmse"])
    return m


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--fold", type=int, default=None,
                    help="single fold/cell (fan-out); the result is partial")
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--case-plans-dir", type=Path, default=ROOT / "case_plans")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "robustness")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--merge", default=None, metavar="EXPERIMENT")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"  {k:<20} {v['kind']:<11} drop={str(v.get('drop') or '-'):<10} "
                  f"keep={str(v.get('keep') or '-'):<9} assign={v.get('assign', '-')}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TC.configure_logger(args.out_dir / "robustness.log", args.verbose)

    if args.merge:
        merge_partials(args.merge, args.out_dir, logger)
        summarize(args.out_dir)
        return 0
    if args.summarize:
        df = summarize(args.out_dir)
        print(df.to_string(index=False) if len(df) else "(no results yet)")
        return 0
    if not args.experiment:
        ap.error("pass --experiment NAME, or --list / --summarize / --merge")
    if args.experiment not in EXPERIMENTS:
        ap.error(f"unknown experiment: {args.experiment}. Use --list.")

    spec = EXPERIMENTS[args.experiment]
    tag = args.experiment if args.fold is None else f"{args.experiment}_fold{args.fold}"
    pred_path = args.out_dir / f"{tag}_predictions.parquet"
    if pred_path.exists() and not args.force:
        print(f"[skip-existing] {tag}")
        return 0

    logger.info("=" * 90)
    logger.info("[run] %s (%s) fold=%s", args.experiment, spec["kind"], args.fold)
    feats_all = TC.canonical_features(args.shard_root)
    logger.info("feature groups: %s", group_sizes(feats_all))

    t0 = time.time()
    preds = RUNNERS[spec["kind"]](args, spec, logger)
    elapsed = time.time() - t0

    # Stage-1 training predictions, when the design produced them. These make the
    # corrector refittable on its own: the target is pm25 - pred_stage1 on the
    # training rows and everything else comes from the raw shards. Small (one float
    # per training row per fold) next to the hours a Stage-1 re-fit costs.
    #
    # Pop BEFORE to_parquet, not after: pyarrow serialises .attrs into the file's
    # metadata as JSON, so a DataFrame left in there fails the write itself.
    s1t = preds.attrs.pop("stage1_train", None)

    preds.to_parquet(pred_path, index=False)
    if s1t is not None and len(s1t):
        s1_path = args.out_dir / f"{tag}_stage1_train.parquet"
        s1t.to_parquet(s1_path, index=False)
        logger.info("cached Stage-1 training predictions: %d rows -> %s",
                    len(s1t), s1_path.name)

    feats = select_features(feats_all, drop=spec.get("drop"), keep_mode=spec.get("keep"))
    m = kloog_metrics(preds["obs_pm25"], preds["pred_final"], preds["grid_cell_id"])
    m.update({"experiment": args.experiment, "kind": spec["kind"],
              "n_features": len(feats), "fold_subset": args.fold,
              "elapsed_seconds": round(elapsed, 1), "shard_root": str(args.shard_root)})
    (args.out_dir / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    logger.info("[done] %s in %.1f min | features=%d pooled_r2=%.3f rmse=%.2f spatial_r2=%.3f",
                args.experiment, elapsed / 60, len(feats),
                m["pooled_r2"], m["rmse"], m["spatial_r2"])
    summarize(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
