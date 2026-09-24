#!/usr/bin/env python3
"""Assemble the Ontario-train / Quebec-test external validation so it runs through
the production fold script unchanged.

The fold script trains on the plan's `train_pair_blocks` and scores its
`heldout_case_keys`, both read from one shard root. External validation is therefore
just a plan whose training blocks are all 72 Ontario blocks and whose held-out blocks
are Quebec's -- no new code, provided both provinces sit in the same root.

    Data/pair_blocks/PAIR_<year>_<region>/     72 Ontario blocks   169,882 rows
    Data_external/qc/pair_blocks/QC_BLOCK_*/    4 Quebec blocks    181,401 rows
    -> Data_on_plus_qc/pair_blocks/            76 blocks

Quebec never appears in training, so this measures transfer to a province the model
has never seen. That is a stronger test than the region-year CV, where the same grid
cells always appear in training under a different year.

    python make_external_qc_plan.py

Verifies before building that the QC blocks carry the same 651 predictors and resolve
under the same name mapping -- a silent feature mismatch would make the comparison
meaningless rather than merely wrong.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
KEY_COLUMNS = ["date", "year", "fold_region", "CanOSSEM_RASTER_CELL", "pm25",
               "_year_region_pair", "naps_id", "station_name"]


def shard_frame_column(c: str) -> str:
    if "_roll3_" in c or "_roll7_" in c:
        return c.replace("_roll3_", "_lag1_roll3_").replace("_roll7_", "_lag1_roll7_")
    return c


def check_compatible(on_root: Path, qc_root: Path) -> list[str]:
    """The two roots must declare the same predictors, and QC must actually carry them."""
    on_feats = json.loads((on_root / "manifest.json").read_text(encoding="utf-8"))["feature_cols"]
    qc_feats = json.loads((qc_root / "manifest.json").read_text(encoding="utf-8"))["feature_cols"]
    if on_feats != qc_feats:
        raise SystemExit("manifests declare different feature_cols; the comparison "
                         "would not be like-for-like")
    for b in sorted((qc_root / "pair_blocks").iterdir()):
        names = set(pq.read_schema(b / "frame.parquet").names)
        missing_keys = [c for c in KEY_COLUMNS if c not in names]
        if missing_keys:
            raise SystemExit(f"{b.name}: missing key column(s) {missing_keys}")
        absent = [c for c in on_feats
                  if c not in names and shard_frame_column(c) not in names]
        if absent:
            raise SystemExit(f"{b.name}: {len(absent)} predictor(s) absent, e.g. {absent[:5]}")
    print(f"  compatibility: {len(on_feats)} predictors resolve in every QC block")
    return on_feats


def build_root(on_root: Path, qc_root: Path, out: Path) -> tuple[list[str], list[str]]:
    if out.exists():
        shutil.rmtree(out)
    (out / "pair_blocks").mkdir(parents=True)
    shutil.copy2(on_root / "manifest.json", out / "manifest.json")

    on_blocks, qc_blocks = [], []
    for src in sorted((on_root / "pair_blocks").iterdir()):
        shutil.copytree(src, out / "pair_blocks" / src.name)
        on_blocks.append(src.name)
    for src in sorted((qc_root / "pair_blocks").iterdir()):
        shutil.copytree(src, out / "pair_blocks" / src.name)
        qc_blocks.append(src.name)

    rows = lambda b: pq.ParquetFile(out / "pair_blocks" / b / "frame.parquet").metadata.num_rows
    print(f"  train (Ontario): {len(on_blocks)} blocks, {sum(rows(b) for b in on_blocks):,} rows")
    print(f"  test  (Quebec) : {len(qc_blocks)} blocks, {sum(rows(b) for b in qc_blocks):,} rows")
    return on_blocks, qc_blocks


def write_plan(on_blocks: list[str], qc_blocks: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "fold_no": 1,
        "fold_label": "GROUP_01",
        "assignment_source": "external validation: train all Ontario, test all Quebec",
        "assignment_mode": "province_holdout",
        # Declared so the production validator checks THIS design, not the 63/9/72
        # it assumes for the thesis folds.
        "expected_train_count": len(on_blocks),
        "expected_heldout_count": len(qc_blocks),
        "expected_total_count": len(on_blocks) + len(qc_blocks),
        "heldout_case_keys": qc_blocks,
        "train_pair_blocks": on_blocks,
        # Quebec is the TEST set here, not support data. support_blocks must stay empty
        # or the fold script would fold it into training and silently destroy the test.
        "support_blocks": [],
    }
    (out / "GROUP_01.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  wrote 1 plan -> {out.name}/GROUP_01.json")


def validate(plans_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location("F", HERE / "run_lgbm_thesis_fold.py")
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    plan, _ = F.load_fold_plan(plans_dir, 1)
    print(f"  production loader accepts it: train={len(plan['train_pair_blocks'])} "
          f"test={len(plan['heldout_case_keys'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on-root", type=Path, default=HERE / "Data")
    ap.add_argument("--qc-root", type=Path, default=HERE / "Data_external" / "qc")
    ap.add_argument("--out-root", type=Path, default=HERE / "Data_on_plus_qc")
    ap.add_argument("--plans-dir", type=Path, default=HERE / "case_plans_external_qc")
    args = ap.parse_args()

    check_compatible(args.on_root, args.qc_root)
    on_blocks, qc_blocks = build_root(args.on_root, args.qc_root, args.out_root)
    write_plan(on_blocks, qc_blocks, args.plans_dir)
    validate(args.plans_dir)
    print(f"\n  run with:\n    python run_lgbm_thesis_fold.py --fold 1 "
          f"--shard-root {args.out_root.name} --case-plans-dir {args.plans_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
