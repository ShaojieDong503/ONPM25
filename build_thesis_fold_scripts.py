#!/usr/bin/env python3
"""Generate run_xgb_thesis_fold.py and run_rf_thesis_fold.py from the LightGBM one.

run_lgbm_thesis_fold.py is the single source of truth for the fold logic: plan
validation, column resolution, NaN policy, the corrector pool rule, the 80-column
encoding, the out-of-fold table and every post-condition. Keeping three hand-edited
copies in sync is how those drift apart, so the other two are derived from it.

The substitution table below IS the documentation of what legitimately differs
between learners. Every entry asserts its target was found, so a rename upstream
fails loudly here rather than silently producing a stale file.

    python build_thesis_fold_scripts.py [--check]

--check verifies the files on disk match what would be generated, and exits
non-zero if not -- use it to detect a hand-edit that bypassed this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_lgbm_thesis_fold.py"

# --------------------------------------------------------------------------------
# Per-learner differences. Anything NOT listed here is identical across all three.
#
# Six categories, and only the first two are what people expect:
#   1. hyperparameters          STAGE1_PARAMS
#   2. estimator class          + its import
#   3. fit() signature          LightGBM accepts feature_name=, XGBoost/sklearn do not
#   4. native model export      .txt booster / .json / joblib
#   5. version recorded         which library goes in run_manifest.json
#   6. labels                   docstring, logger name, banner, model_family, out root
# --------------------------------------------------------------------------------

XGB_PARAMS = '''STAGE1_PARAMS = {
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
}'''

RF_PARAMS = '''STAGE1_PARAMS = {
    "n_estimators": 900,
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": 0.6,
    "bootstrap": True,
    "random_state": SEED,
    "n_jobs": -1,
}'''


def lgbm_params_block(src: str) -> str:
    """Extract the LightGBM STAGE1_PARAMS literal, whatever its current contents."""
    start = src.index("STAGE1_PARAMS = {")
    end = src.index("\n}", start) + 2
    return src[start:end]


def common_edits(name_long: str, name_short: str, family: str) -> list[tuple[str, str]]:
    """Label-only substitutions: docstrings, logger, banner, model_family, out root."""
    return [
        ("Run ONE thesis LightGBM outer fold for the Ontario PM2.5 model.",
         f"Run ONE thesis {name_long} outer fold for the Ontario PM2.5 model."),
        ("- LightGBM regression", f"- {name_long} regression"),
        ("- same LightGBM hyperparameters as Stage 1",
         f"- same {name_long} hyperparameters as Stage 1"),
        ('    r"\\thesis_lgbm_grouped_runs"', f'    r"\\thesis_{family}_grouped_runs"'),
        # The non-Windows fallback root, which must differ per family or all three
        # would write into outputs/lgbm_thesis and overwrite each other.
        ('FAMILY_OUT_DIRNAME = "lgbm_thesis"', f'FAMILY_OUT_DIRNAME = "{family}_thesis"'),
        ('logging.getLogger("thesis_lgbm_fold")', f'logging.getLogger("thesis_{family}_fold")'),
        ('description="Run one thesis Table-A5 LightGBM fold (Stage 1 + corrector)."',
         f'description="Run one thesis Table-A5 {name_long} fold (Stage 1 + corrector)."'),
        ('logger.info("THESIS LIGHTGBM OUTER FOLD START")',
         f'logger.info("THESIS {name_short} OUTER FOLD START")'),
        ('logger.info("Fitting Stage-1 LightGBM...")',
         f'logger.info("Fitting Stage-1 {name_long}...")'),
        ('logger.info("Fitting residual-corrector LightGBM...")',
         f'logger.info("Fitting residual-corrector {name_long}...")'),
        # three occurrences: the prediction table, and the two saved bundles
        ('            "model_family": "lgbm",', f'            "model_family": "{family}",'),
    ]


# fit(): LightGBM takes feature_name=, the other two reject it.
FIT_EDITS = [
    ("""        stage1.fit(
            X_train,
            y_train,
            feature_name=list(canonical_features),
        )""",
     """        # Neither XGBoost's sklearn API nor RandomForestRegressor accepts
        # feature_name in fit(); the names are preserved in the saved bundle instead.
        stage1.fit(
            X_train,
            y_train,
        )"""),
    ("""        corrector.fit(
            X_corr_train,
            y_corr,
            feature_name=corrector_feature_names,
        )""",
     """        corrector.fit(
            X_corr_train,
            y_corr,
        )"""),
]

LGBM_EXPORT = """        # Native LightGBM model representations as an additional robust artifact.
        stage1.booster_.save_model(str(fold_out / "stage1_model.txt"))
        corrector.booster_.save_model(str(fold_out / "corrector_model.txt"))"""

XGB_EXPORT = """        # Native XGBoost JSON as an additional robust artifact: unlike the pickle
        # it survives an xgboost upgrade.
        stage1.save_model(str(fold_out / "stage1_model.json"))
        corrector.save_model(str(fold_out / "corrector_model.json"))"""

RF_EXPORT = """        # scikit-learn has no portable native format, so joblib is the only option.
        # It is version-sensitive in the same way the pickle is -- the run_manifest
        # records scikit_learn.__version__ so a later reader knows what wrote it.
        joblib.dump(stage1, fold_out / "stage1_estimator.joblib", compress=3)
        joblib.dump(corrector, fold_out / "corrector_estimator.joblib", compress=3)"""

LGBM_DOC_ARTIFACTS = """  stage1_model.txt
  corrector_model.txt"""


def build(kind: str, src: str) -> str:
    out = src
    edits: list[tuple[str, str]] = []

    if kind == "xgb":
        edits += common_edits("XGBoost", "XGBOOST", "xgb")
        edits += [
            ("import lightgbm\nfrom lightgbm import LGBMRegressor",
             "import xgboost\nfrom xgboost import XGBRegressor"),
            (lgbm_params_block(src), XGB_PARAMS),
            ("stage1 = LGBMRegressor(**STAGE1_PARAMS)", "stage1 = XGBRegressor(**STAGE1_PARAMS)"),
            ("corrector = LGBMRegressor(**CORRECTOR_PARAMS)",
             "corrector = XGBRegressor(**CORRECTOR_PARAMS)"),
            (LGBM_EXPORT, XGB_EXPORT),
            (LGBM_DOC_ARTIFACTS, "  stage1_model.json\n  corrector_model.json"),
            ('                "lightgbm": lightgbm.__version__,',
             '                "xgboost": xgboost.__version__,'),
        ] + FIT_EDITS
    elif kind == "rf":
        edits += common_edits("Random Forest", "RANDOM FOREST", "rf")
        edits += [
            ("import lightgbm\nfrom lightgbm import LGBMRegressor",
             "import joblib\nfrom sklearn.ensemble import RandomForestRegressor"),
            (lgbm_params_block(src), RF_PARAMS),
            ("stage1 = LGBMRegressor(**STAGE1_PARAMS)",
             "stage1 = RandomForestRegressor(**STAGE1_PARAMS)"),
            ("corrector = LGBMRegressor(**CORRECTOR_PARAMS)",
             "corrector = RandomForestRegressor(**CORRECTOR_PARAMS)"),
            (LGBM_EXPORT, RF_EXPORT),
            (LGBM_DOC_ARTIFACTS, "  stage1_estimator.joblib\n  corrector_estimator.joblib"),
            # scikit-learn is already recorded; drop the lightgbm line entirely.
            ('                "lightgbm": lightgbm.__version__,\n', ""),
        ] + FIT_EDITS
    else:
        raise ValueError(kind)

    for old, new in edits:
        n = out.count(old)
        if n == 0:
            raise SystemExit(
                f"[{kind}] substitution target not found -- run_lgbm_thesis_fold.py "
                f"changed and this table is stale:\n  {old.strip().splitlines()[0][:100]}")
        out = out.replace(old, new)

    header = (f'# GENERATED FILE -- do not edit by hand.\n'
              f'# Derived from run_lgbm_thesis_fold.py by build_thesis_fold_scripts.py.\n'
              f'# Edit the LightGBM script or the substitution table, then regenerate.\n')
    lines = out.split("\n")
    return lines[0] + "\n" + header + "\n".join(lines[1:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify on-disk files match; exit non-zero if not")
    args = ap.parse_args()

    src = SOURCE.read_text(encoding="utf-8")
    rc = 0
    for kind in ("xgb", "rf"):
        target = HERE / f"run_{kind}_thesis_fold.py"
        built = build(kind, src)
        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            same = current == built
            print(f"  {target.name:<28} {'in sync' if same else 'OUT OF SYNC'}")
            rc |= 0 if same else 1
        else:
            target.write_text(built, encoding="utf-8", newline="\n")
            print(f"  wrote {target.name}  ({len(built.splitlines()):,} lines)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
