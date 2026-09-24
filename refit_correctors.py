#!/usr/bin/env python3
"""
Refit correctors WITHOUT refitting Stage 1.

Why this is sound, not a shortcut
---------------------------------
`load_one_block` applies nan_to_num to the Stage-1 design matrix whether or not the
stored frame was pre-filled, so the Stage-1 matrix -- and therefore the fitted Stage-1
model -- is bit-identical between `Data/` and `Data_zerofilled/` (verified: max|dX| = 0
over 651 x 2,903, and Stage-1 holdout RMSE identical to six decimals on all 8 folds).
Only the corrector reads the frame's own NaN. So when corrector handling changes, the
correct response is to refit correctors and leave Stage 1 alone.

A corrector needs exactly two things beyond the raw shards:
  * the TRAINING rows' Stage-1 predictions  -> the residual target
  * the HELD-OUT rows' Stage-1 predictions  -> to score the refitted corrector
Both come either from a saved Stage-1 model or from a cached prediction table.

  # primary CV: reads GROUP_xx/stage1_model.txt, predicts the training rows itself
  python refit_correctors.py --mode folds --out-root outputs/lgbm_thesis

  # robustness: reads <experiment>_stage1_train.parquet written by robustness_runner
  python refit_correctors.py --mode cache --cache-dir outputs/robustness \\
      --experiment ablation_no_aod

  # verification: refit fold 1 and check it reproduces the full re-fit
  python refit_correctors.py --mode folds --out-root outputs/lgbm_thesis \\
      --folds 1 --verify

`--verify` compares the refitted corrector's held-out output against the saved
`pred_corrector` column and fails if they disagree beyond tolerance. That is the check
that makes this equivalent to a full re-fit rather than merely faster than one.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_lgbm_thesis_fold as F  # noqa: E402
import thesis_core as TC  # noqa: E402

KEY = ["grid_cell_id", "date"]
TOL = 1e-6

# Each family script carries its OWN copy of build_corrector_pool /
# fit_corrector_fill_values / transform_corrector / CORRECTOR_PARAMS -- they are not
# shared. So a refit must use the module that produced the run, not the LightGBM one,
# or the fill rules and feature order could silently diverge from what was fitted.
FAMILIES = {
    "lgbm": {"module": "run_lgbm_thesis_fold",
             "stage1_files": ["stage1_model.txt", "stage1_model.pkl"],
             "corrector_out": "corrector_model_refit"},
    "xgb":  {"module": "run_xgb_thesis_fold",
             "stage1_files": ["stage1_model.json", "stage1_model.pkl"],
             "corrector_out": "corrector_model_refit"},
    "rf":   {"module": "run_rf_thesis_fold",
             "stage1_files": ["stage1_estimator.joblib"],
             "corrector_out": "corrector_estimator_refit"},
}


def family_module(fam: str):
    import importlib
    return importlib.import_module(FAMILIES[fam]["module"])


def load_stage1(fam: str, gdir: Path):
    """Return an object with .predict(X) for the saved Stage-1 model."""
    spec = FAMILIES[fam]
    for fn in spec["stage1_files"]:
        path = gdir / fn
        if not path.exists():
            continue
        if fn.endswith(".txt"):
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        if fn.endswith(".joblib"):
            import joblib
            return joblib.load(path)
        if fn.endswith(".pkl"):
            obj = pickle.loads(path.read_bytes())
            return obj.get("model", obj) if isinstance(obj, dict) else obj
        if fn.endswith(".json"):
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(str(path))
            return m
    raise FileNotFoundError(
        f"no Stage-1 model for family {fam} in {gdir} "
        f"(looked for {spec['stage1_files']})")


def make_corrector(fam: str, mod):
    """A fresh corrector estimator with that family's own recorded parameters."""
    params = dict(mod.CORRECTOR_PARAMS)
    if fam == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if fam == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**params)


def quiet(name="refit") -> logging.Logger:
    lg = logging.getLogger(name)
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.ERROR)
    return lg


def keyed(df: pd.DataFrame, cell_col="CanOSSEM_RASTER_CELL") -> pd.DataFrame:
    out = df.copy()
    out["grid_cell_id"] = out[cell_col].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    return out


def fit_corrector(mod, fam: str, train_df: pd.DataFrame, lg: logging.Logger):
    """That family's own corrector, on a frame already carrying pred/resid_stage1."""
    mod.validate_corrector_columns(train_df)
    pool_df, _mask, diag = mod.build_corrector_pool(train_df, lg)
    y = pd.to_numeric(pool_df["resid_stage1"], errors="raise").to_numpy(dtype=np.float64)

    fills = mod.fit_corrector_fill_values(pool_df)     # aborts on a pre-filled pool
    X, cols = mod.transform_corrector(pool_df, fills)

    m = make_corrector(fam, mod)
    # Only LightGBM's sklearn wrapper takes feature_name= on fit.
    if fam == "lgbm":
        m.fit(X, y, feature_name=cols)
    else:
        m.fit(X, y)
    return m, cols, fills, diag


def apply_corrector(mod, model, cols, fills, test_df: pd.DataFrame,
                    pred_stage1: np.ndarray) -> np.ndarray:
    d = test_df.copy()
    d["pred_stage1"] = pred_stage1
    X, c = mod.transform_corrector(d, fills)
    if c != cols:
        raise AssertionError("corrector feature order differs between fit and apply")
    return np.asarray(model.predict(X), dtype=np.float64)


def flags_live(fam: str, model, cols: list[str]) -> tuple[int, int]:
    """How many __isna indicators the fitted corrector actually uses.

    Zero is the signature of the contaminated fit: the flags were constant, so no
    split could reference them. Each library exposes importance differently.
    """
    if fam == "lgbm":
        names = list(model.booster_.feature_name())
        imp = model.booster_.feature_importance("gain")
    elif fam == "xgb":
        names, imp = cols, model.feature_importances_
    else:
        names, imp = cols, model.feature_importances_
    isna = [(n, g) for n, g in zip(names, imp) if n.endswith("__isna")]
    return sum(1 for _, g in isna if g > 0), len(isna)


def refit_one_fold(fam: str, shard_root: Path, gdir: Path, fold: int,
                   case_plans: Path, lg: logging.Logger, verify: bool):
    """The saved Stage-1 model supplies both prediction sets; Stage 1 is never refit."""
    mod = family_module(fam)
    man = json.loads((gdir / "run_manifest.json").read_text(encoding="utf-8"))
    feats = list(TC.canonical_features(shard_root))
    plan, _ = TC.load_fold_plan(case_plans, fold)
    train_blocks, test_blocks = list(plan["train_pair_blocks"]), list(man["holdout_blocks"])

    s1 = load_stage1(fam, gdir)

    train_df, Xtr, _ = mod.load_blocks(shard_root, train_blocks, feats, lg)
    y = pd.to_numeric(train_df["pm25"], errors="raise").to_numpy(dtype=np.float64)
    p_tr = np.asarray(s1.predict(Xtr), dtype=np.float64)
    train_df = train_df.copy()
    train_df["pred_stage1"] = p_tr
    train_df["resid_stage1"] = y - p_tr

    model, cols, fills, diag = fit_corrector(mod, fam, train_df, lg)

    test_df, Xte, _ = mod.load_blocks(shard_root, test_blocks, feats, lg)
    p_te = np.asarray(s1.predict(Xte), dtype=np.float64)
    corr = apply_corrector(mod, model, cols, fills, test_df, p_te)

    live, total = flags_live(fam, model, cols)
    res = {"family": fam, "fold": fold, "train_rows": int(len(train_df)),
           "pool_rows": diag["pool_rows"], "pool_fraction": diag["pool_fraction"],
           "isna_live": live, "isna_total": total,
           "burned_nearest_km_500km_fill": fills.get("burned_nearest_km_500km")}

    obs = pd.to_numeric(test_df["pm25"], errors="raise").to_numpy(dtype=np.float64)
    res["final_rmse_refit"] = float(np.sqrt((((p_te + corr) - obs) ** 2).mean()))
    res["stage1_rmse"] = float(np.sqrt(((p_te - obs) ** 2).mean()))

    saved = pd.read_parquet(gdir / "holdout_predictions.parquet",
                            columns=KEY + ["pred_stage1", "pred_corrector", "pred_final"])
    saved["date"] = pd.to_datetime(saved["date"])
    got = keyed(test_df)[KEY].assign(pred_stage1=p_te, pred_corrector=corr)
    j = got.merge(saved, on=KEY, how="left", suffixes=("_new", "_saved"),
                  validate="one_to_one")
    if j["pred_corrector_saved"].isna().any():
        raise AssertionError(f"fold {fold}: rows missing from the saved table")
    # Stage 1 must reproduce exactly -- that is the premise the whole shortcut rests
    # on. The corrector is EXPECTED to differ when the old run was contaminated.
    res["stage1_max_abs_diff"] = float(np.abs(j.pred_stage1_new - j.pred_stage1_saved).max())
    res["corrector_max_abs_diff"] = float(
        np.abs(j.pred_corrector_new - j.pred_corrector_saved).max())
    res["final_rmse_saved"] = float(np.sqrt(((j.pred_final.to_numpy() - obs) ** 2).mean()))
    if verify:
        res["verify_ok"] = bool(res["corrector_max_abs_diff"] <= TOL)
    res["stage1_reproduced"] = bool(res["stage1_max_abs_diff"] <= TOL)

    # The corrected held-out table. Without this the refit would leave
    # holdout_predictions.parquet holding the contaminated pred_corrector/pred_final,
    # and every downstream comparison would quietly keep reading the old numbers.
    preds = keyed(test_df)[KEY].copy()
    preds["year"] = pd.to_datetime(test_df["date"]).dt.year.astype(int).to_numpy()
    preds["model_family"] = fam
    preds["outer_fold"] = fold
    preds["obs_pm25"] = obs
    preds["pred_stage1"] = p_te
    preds["pred_corrector"] = corr
    preds["pred_final"] = p_te + corr
    return res, model, cols, fills, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="lgbm")
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="directory holding GROUP_01..08 for this family")
    ap.add_argument("--case-plans-dir", type=Path, default=ROOT / "case_plans")
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--verify", action="store_true",
                    help="require the refit to REPRODUCE the saved corrector "
                         "(only meaningful when the saved run was already clean)")
    ap.add_argument("--save", action="store_true",
                    help="write the refitted corrector into each GROUP dir")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    lg = quiet()
    fam = args.family
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    results, all_preds, t0 = [], [], time.time()

    print(f"[refit] family={fam}  shards={args.shard_root}  runs={args.out_root}")
    print("[refit] Stage 1 is NOT refitted; saved models supply both prediction sets.")
    print()

    for f in folds:
        gdir = args.out_root / f"GROUP_{f:02d}"
        if not gdir.exists():
            print(f"  [skip] fold {f}: {gdir} missing")
            continue
        t = time.time()
        res, model, cols, fills, preds = refit_one_fold(
            fam, args.shard_root, gdir, f, args.case_plans_dir, lg, args.verify)
        res["seconds"] = round(time.time() - t, 1)
        results.append(res)
        s1 = "OK" if res["stage1_reproduced"] else f"MOVED {res['stage1_max_abs_diff']:.2e}"
        print(f"  fold {f}: __isna live {res['isna_live']:>2}/{res['isna_total']}  "
              f"burned_500km fill {res['burned_nearest_km_500km_fill']}  "
              f"rmse {res['final_rmse_saved']:.4f} -> {res['final_rmse_refit']:.4f}  "
              f"stage1 {s1}  {res['seconds']}s", flush=True)
        if args.save:
            stem = FAMILIES[fam]["corrector_out"]
            pickle.dump({"model": model, "corrector_cols": cols, "fill_values": fills},
                        (gdir / f"{stem}.pkl").open("wb"))
            preds.to_parquet(gdir / "holdout_predictions_refit.parquet", index=False)
            (gdir / "corrector_refit_diagnostics.json").write_text(
                json.dumps(res, indent=2), encoding="utf-8")
        all_preds.append(preds)

    print()
    print(f"[done] {len(results)} fold(s) in {time.time()-t0:.0f}s")
    if results:
        moved = [r for r in results if not r["stage1_reproduced"]]
        print(f"[stage1] reproduced exactly on {len(results)-len(moved)}/{len(results)} "
              f"fold(s) -- the premise of refitting the corrector alone")
        if moved:
            print(f"[stage1] MOVED on folds {[r['fold'] for r in moved]}; "
                  f"do not trust these refits")
        live = [r["isna_live"] for r in results]
        print(f"[flags]  __isna indicators used: min {min(live)}, max {max(live)} of "
              f"{results[0]['isna_total']}  (0 everywhere = still contaminated)")
        d = np.mean([r["final_rmse_refit"] - r["final_rmse_saved"] for r in results])
        print(f"[rmse]   mean change vs the saved run: {d:+.6f}")
    if all_preds:
        a = pd.concat(all_preds, ignore_index=True)
        y = a.obs_pm25.to_numpy(); sst = ((y - y.mean()) ** 2).sum()
        for col in ("pred_stage1", "pred_final"):
            e = a[col].to_numpy() - y
            print(f"[pooled] {col:<12} n={len(a):,}  rmse={np.sqrt((e**2).mean()):.6f}  "
                  f"r2={1 - (e**2).sum()/sst:.6f}")
        if args.save:
            out = args.out_root / "holdout_predictions_refit_all.parquet"
            a.to_parquet(out, index=False)
            print(f"[wrote] {out}")
    if args.verify and results:
        bad = [r for r in results if not r.get("verify_ok")]
        print(f"[verify] {len(results)-len(bad)}/{len(results)} reproduce the saved "
              f"corrector within {TOL:g}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[wrote] {args.out_json}")
    return 1 if any(not r["stage1_reproduced"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
