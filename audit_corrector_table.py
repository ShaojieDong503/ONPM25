#!/usr/bin/env python3
"""
Build and audit the corrector design matrix, and prove its missingness is real.

Reviewer point 1 asks that the missingness indicators be created BEFORE filling and
that fill values come only from the training portion of a fold. Both were already true
in `transform_corrector` / `fit_corrector_fill_values`. What defeated them was the
input: `Data_zerofilled/` had NaN already replaced by 0, so `~np.isfinite(x)` found
nothing, every `__isna` flag was constant zero, and every median was dragged toward 0.

Stage 1 hid this. `load_one_block` applies nan_to_num to the design matrix on both
paths, so the Stage-1 matrix is bit-identical between the two roots and nothing
downstream of Stage 1 looked wrong. Only the corrector reads the frame's own NaN.

The pre-filled copy has since been deleted, so this is no longer a before/after
comparison -- it is a standing check that whatever shard root is handed to it carries
real missingness:

  python audit_corrector_table.py --shard-root Data
  python audit_corrector_table.py --shard-root Data --fold-blocks all
  python audit_corrector_table.py --shard-root Data_external/qc --fold-blocks all

Exit status is 0 only if the root passes. A root whose 40 corrector inputs contain no
missing values at all is reported as CONTAMINATED and exits 2, because that is the
signature of a pre-filled copy rather than of clean data.

Outputs (under --out-dir):
  corrector_column_audit.csv   per-column missing counts, fill value, flag verdict
  corrector_table_audit.json   totals, verdict, the fold plan the fills came from
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_lgbm_thesis_fold as F  # noqa: E402
import thesis_core as TC  # noqa: E402


def quiet() -> logging.Logger:
    lg = logging.getLogger("audit")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.ERROR)
    return lg


def is_distance(col: str) -> bool:
    """Columns whose fill is a 'nothing in range' sentinel rather than a central value."""
    return "dist_nearest" in col or "nearest_km" in col


def radius_km(col: str) -> int | None:
    """The neighbourhood radius named in the column, e.g. ..._500km -> 500."""
    m = re.search(r"_(\d+)km", col)
    return int(m.group(1)) if m else None


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness and the fill each column would receive."""
    rows = []
    n = len(df)
    for c in F.CORRECTOR_RAW_COLS:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "n_rows": n, "n_missing": None,
                         "missing_frac": None, "fill_value": None, "fill_rule": None,
                         "verdict": "ABSENT"})
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        finite = x[np.isfinite(x)]
        n_missing = int(len(x) - len(finite))

        if is_distance(c):
            fill = float(finite.max() + 10.0) if len(finite) else 9999.0
            rule = "max(finite)+10  [sentinel: nothing in range]"
        else:
            fill = float(np.median(finite)) if len(finite) else 0.0
            rule = "median(finite)"

        # A distance sentinel is meaningful only if it sits beyond the real distances.
        # When zeros were substituted for missing, max(finite) became 0 and the
        # sentinel collapsed to 10 km -- telling the model that a cell with no fire in
        # range was 10 km from one. Flag that explicitly rather than by eye.
        # Both conditions must be able to fire together: on a pre-filled root the
        # substituted zeros BOTH flatten the flag AND collapse the sentinel, and the
        # collapse is the more alarming of the two. An elif would hide it.
        flags = []
        if n_missing == 0:
            flags.append(f"NO MISSING -> derived {c}__isna is constant 0")
        # Compare the sentinel against the search radius named in the column, not
        # against the column's own values: on a contaminated root the observed
        # distances are corrupted too, so the data carries no clean reference. A
        # "nearest km within 500 km" "nothing in range" marker must exceed 500.
        r = radius_km(c)
        if is_distance(c) and r and fill <= r:
            flags.append(f"SENTINEL COLLAPSED (fill {fill:g} <= {r} km search radius)")
        verdict = "; ".join(flags) if flags else "ok"

        # Degeneracy: an input can be present-and-constant, or entirely absent. Either
        # way it contributes nothing, and "the flag is constant" is a different failure
        # from "the flag is wrong" -- report it so it is not mistaken for contamination.
        n_unique = int(len(np.unique(finite))) if len(finite) else 0
        if c == "pred_stage1":
            # load_pool stubs this to 0.0 -- there is no Stage-1 model in an audit
            # context. Its degeneracy here is an artifact of this script, not of the
            # data, so say so instead of reporting a finding that does not exist.
            flags.append("STUBBED BY THIS AUDIT (no Stage-1 model); not a data finding")
        elif n_missing == n:
            flags.append("ALL MISSING -> value and flag both constant")
        elif n_unique == 1 and n_missing:
            flags.append(f"VALUE CONSTANT ({finite[0]:g}) -> only the flag varies")
        elif n_unique == 1:
            flags.append(f"VALUE CONSTANT ({finite[0]:g}) and never missing -> inert")
        verdict = "; ".join(flags) if flags else "ok"

        rows.append({"column": c, "present": True, "n_rows": n, "n_missing": n_missing,
                     "n_unique_finite": n_unique,
                     "missing_frac": round(n_missing / n, 6) if n else None,
                     "fill_value": fill, "fill_rule": rule,
                     # fill_value is what WOULD be substituted; with n_missing == 0 it
                     # is computed but never applied. Say so, so a median of 0 on a
                     # 0/1 column is not misread as the column being all zeros.
                     "fill_applied": bool(n_missing > 0),
                     "mean_value": float(finite.mean()) if len(finite) else None,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def load_pool(shard_root: Path, blocks: list[str], lg: logging.Logger) -> pd.DataFrame:
    """The frame the corrector trains from, with Stage-1 residual columns stubbed.

    build_corrector_pool needs pm25 and resid_stage1. The pool rule is measured here
    for its own sake (it is reviewer point 5's evidence), so a zero residual is used:
    it exercises the smoke/high-pm clauses without inventing a Stage-1 model.
    """
    feats = TC.canonical_features(shard_root)
    df, _X, _ = F.load_blocks(shard_root, blocks, feats, lg)
    df = df.copy()
    if "pm25" not in df.columns:
        raise SystemExit("[error] frame has no pm25 column")
    df["pred_stage1"] = 0.0
    df["resid_stage1"] = 0.0
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--case-plans-dir", type=Path, default=ROOT / "case_plans")
    ap.add_argument("--fold", type=int, default=1,
                    help="audit this fold's TRAINING blocks (fills are train-only)")
    ap.add_argument("--fold-blocks", default=None,
                    help="'all' to audit every block instead of one fold's training set")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "corrector_audit")
    args = ap.parse_args()

    lg = quiet()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def blocks_for(root: Path) -> tuple[list[str], str]:
        if args.fold_blocks == "all":
            bs = sorted(p.name for p in (root / "pair_blocks").iterdir() if p.is_dir())
            return bs, "all blocks"
        plan, _ = TC.load_fold_plan(args.case_plans_dir, args.fold)
        train = list(plan["train_pair_blocks"])
        return train, f"fold {args.fold} training blocks ({len(train)})"

    results = {}
    for label, root in [("primary", args.shard_root)]:
        blocks, scope = blocks_for(root)
        df = load_pool(root, blocks, lg)
        pool, _mask, diag = F.build_corrector_pool(df, lg)
        a = audit(pool)

        present = a[a.present]
        total_missing = int(present.n_missing.sum())
        cols_with_missing = int((present.n_missing > 0).sum())
        collapsed = present[present.verdict.str.contains("SENTINEL")]

        verdict = ("CONTAMINATED (pre-filled: no missing values anywhere)"
                   if total_missing == 0 else "OK")

        print(f"\n{'='*74}\n[{label}] {root}\n  scope: {scope}\n{'='*74}")
        print(f"  pool rows            {len(pool):,} of {len(df):,} "
              f"(pool_fraction {diag['pool_fraction']:.4f})")
        print(f"  corrector inputs     {int(present.present.sum())} of "
              f"{len(F.CORRECTOR_RAW_COLS)} present")
        print(f"  total missing values {total_missing:,}")
        print(f"  columns with missing {cols_with_missing} "
              f"-> that many __isna flags can vary")
        print(f"  collapsed sentinels  {len(collapsed)}")
        for _, r in collapsed.iterrows():
            print(f"      {r.column:<42} fill={r.fill_value:<10.4g} {r.verdict}")
        print(f"  VERDICT: {verdict}")

        a.insert(0, "shard_root", str(root))
        results[label] = {"root": str(root), "scope": scope,
                          "pool_rows": int(len(pool)), "frame_rows": int(len(df)),
                          "pool_fraction": diag["pool_fraction"],
                          "total_missing": total_missing,
                          "columns_with_missing": cols_with_missing,
                          "collapsed_sentinels": collapsed.column.tolist(),
                          "verdict": verdict, "table": a}

    frames = [r.pop("table") for r in results.values()]
    pd.concat(frames, ignore_index=True).to_csv(
        args.out_dir / "corrector_column_audit.csv", index=False)

    if len(results) == 2:
        p, c = results["primary"], results["compare"]
        print(f"\n{'='*74}\n[side by side]\n{'='*74}")
        print(f"{'':<26}{'primary':>16}{'compare':>16}")
        for k in ("total_missing", "columns_with_missing", "pool_rows"):
            print(f"  {k:<24}{p[k]:>16,}{c[k]:>16,}")

    TC.save_json({"folds_source": str(args.case_plans_dir), **results},
                 args.out_dir / "corrector_table_audit.json")
    print(f"\n[wrote] {args.out_dir / 'corrector_column_audit.csv'}")
    print(f"[wrote] {args.out_dir / 'corrector_table_audit.json'}")

    return 2 if results["primary"]["verdict"] != "OK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
