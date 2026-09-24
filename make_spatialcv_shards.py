#!/usr/bin/env python3
"""Re-partition the shards by monitored CELL so leave-cells-out runs through the
production fold script unchanged.

The fold script holds out named directories under <shard_root>/pair_blocks/. Nothing
in it requires those directories to be region-year blocks -- that is a property of
the data, not of the code. So leave-cells-out needs no code change, only a shard root
whose partition unit is the cell:

    Data/pair_blocks/PAIR_<year>_<region>/frame.parquet    72 region-year blocks
    Data_by_cell/pair_blocks/CELL_<id>/frame.parquet       41 cell blocks

Every row keeps its columns and its values; only the file it lives in changes. A held-
out cell is then absent from its fold's training entirely, which is the spatial claim
the province-wide surface actually makes.

    python make_spatialcv_shards.py            # build the shard root + 7 fold plans
    python make_spatialcv_shards.py --k 7      # number of spatial folds

Outputs
  Data_by_cell/manifest.json            copied verbatim -- same 651 predictors
  Data_by_cell/pair_blocks/CELL_*/      one parquet per monitored cell
  case_plans_spatialcv/GROUP_0N.json    k plans, each declaring its own counts
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CELL = "CanOSSEM_RASTER_CELL"
SEED = 2026


def build_shards(src: Path, dst: Path) -> dict[str, str]:
    """One parquet per cell. Returns cell -> region, for stratified fold assignment."""
    blocks = sorted(p for p in (src / "pair_blocks").iterdir() if p.is_dir())
    print(f"  reading {len(blocks)} region-year blocks from {src.name}/")
    df = pd.concat([pd.read_parquet(b / "frame.parquet") for b in blocks], ignore_index=True)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    out = dst / "pair_blocks"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(src / "manifest.json", dst / "manifest.json")

    cell_region: dict[str, str] = {}
    written = 0
    for cell, g in df.groupby(df[CELL].astype(str), sort=True):
        d = out / f"CELL_{cell}"
        d.mkdir()
        g.reset_index(drop=True).to_parquet(d / "frame.parquet", index=False)
        cell_region[str(cell)] = str(g["fold_region"].iloc[0])
        written += 1

    total = sum(pd.read_parquet(p / "frame.parquet", columns=[CELL]).shape[0]
                for p in sorted(out.iterdir()))
    if total != len(df):
        raise SystemExit(f"row count changed: {len(df):,} -> {total:,}")
    print(f"  wrote {written} cell blocks, {total:,} rows preserved exactly")
    return cell_region


def balanced_spatial_folds(cell_region: dict[str, str], k: int, seed: int) -> dict[str, int]:
    """Assign cells to k folds so the fold SIZES differ by at most one.

    robustness_features.spatial_fold_assignment stratifies by region but lets the
    sizes fall where they may -- for 41 cells over 7 folds it produced 8,8,6,6,6,4,3,
    so the last fold's estimate rested on 3 monitors.

    Here the cells are ordered region-by-region (shuffled within a region, seeded) and
    then DEALT round-robin. Consecutive cells in the ordering land in different folds,
    so regions still spread across folds, and 41 = 6*6 + 5 falls out by construction.
    Every cell is held out exactly once: no cell is reused to pad a fold to equal size,
    which would give that monitor two out-of-fold predictions and double-count it in
    the pooled metric.
    """
    rng = random.Random(seed)
    by_region: dict[str, list[str]] = {}
    for cell, region in sorted(cell_region.items()):
        by_region.setdefault(region, []).append(cell)

    ordered: list[str] = []
    for region in sorted(by_region):
        cells = sorted(by_region[region])
        rng.shuffle(cells)
        ordered.extend(cells)

    return {cell: i % k for i, cell in enumerate(ordered)}


def build_plans(cell_region: dict[str, str], k: int, out: Path) -> None:
    """k spatial folds over the cell blocks, stratified by region and size-balanced."""
    fold_of = balanced_spatial_folds(cell_region, k=k, seed=SEED)
    cells = sorted(cell_region)

    sizes = [sum(1 for c in cells if fold_of[c] == f) for f in range(k)]
    if max(sizes) - min(sizes) > 1:
        raise SystemExit(f"fold sizes differ by more than one: {sizes}")
    if sum(sizes) != len(cells):
        raise SystemExit(f"assignment covers {sum(sizes)} of {len(cells)} cells")
    print(f"  fold sizes: {sizes}  (sum {sum(sizes)} = {len(cells)} cells, each held out once)")
    names = {c: f"CELL_{c}" for c in cells}

    out.mkdir(parents=True, exist_ok=True)
    for f in range(k):
        test = sorted(names[c] for c in cells if fold_of[c] == f)
        train = [names[c] for c in cells if names[c] not in set(test)]
        if not test:
            raise SystemExit(f"fold {f} holds out no cell; reduce --k")
        plan = {
            "fold_no": f + 1,
            "fold_label": f"GROUP_{f + 1:02d}",
            "assignment_source": f"spatial_cv, leave-cells-out k={k}, seed {SEED}",
            "assignment_mode": "whole_cells_region_stratified",
            # Declared so the production validator checks THIS design strictly,
            # instead of the 63/9/72 it assumes for the thesis folds.
            "expected_train_count": len(train),
            "expected_heldout_count": len(test),
            "expected_total_count": len(cells),
            "heldout_case_keys": test,
            "train_pair_blocks": train,
            "support_blocks": [],
            "heldout_regions": sorted({cell_region[c] for c in cells
                                       if names[c] in set(test)}),
        }
        (out / f"GROUP_{f + 1:02d}.json").write_text(json.dumps(plan, indent=2),
                                                     encoding="utf-8")
    print(f"  wrote {k} fold plans -> {out.name}/")


def validate(plans_dir: Path, k: int) -> None:
    """Validate with the PRODUCTION loader, not a copy of its rules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("F", HERE / "run_lgbm_thesis_fold.py")
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    for f in range(1, k + 1):
        plan, _ = F.load_fold_plan(plans_dir, f)
        print(f"    GROUP_{f:02d}: train={len(plan['train_pair_blocks'])} "
              f"test={len(plan['heldout_case_keys'])} "
              f"regions={','.join(plan['heldout_regions'])}  OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", type=Path, default=HERE / "Data")
    ap.add_argument("--out-root", type=Path, default=HERE / "Data_by_cell")
    ap.add_argument("--plans-dir", type=Path, default=HERE / "case_plans_spatialcv")
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()

    cell_region = build_shards(args.shard_root, args.out_root)
    build_plans(cell_region, args.k, args.plans_dir)
    print("  validating with the production loader:")
    validate(args.plans_dir, args.k)
    print(f"\n  run with:\n    python run_lgbm_thesis_fold.py --fold 1 "
          f"--shard-root {args.out_root.name} --case-plans-dir {args.plans_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
