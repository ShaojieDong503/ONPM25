#!/usr/bin/env python3
"""
Move fold outputs that landed under a literal Windows path name into outputs/.

Why this exists: run_*_thesis_fold.py used to default --out-root to
`Path(r"D:\\lambda\\...\\thesis_lgbm_grouped_runs")`. On Linux that is not an
absolute path and not an error -- a backslash is an ordinary filename character,
so it is a RELATIVE name containing backslashes. A VM run silently created one
directory called `D:\\lambda\\...\\thesis_rf_grouped_runs` under the cwd and wrote
104 GB into it, where compare_models.py does not look.

The fold scripts now fall back to `outputs/<family>_thesis` off-Windows, so this
is a one-shot repair for runs made before that fix. It moves rather than copies
(same filesystem, so it is a rename) and refuses to clobber an existing
destination.

    python relocate_win_outputs.py [--root .] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

LEAF_TO_FAMILY = {
    "thesis_lgbm_grouped_runs": "lgbm_thesis",
    "thesis_xgb_grouped_runs": "xgb_thesis",
    "thesis_rf_grouped_runs": "rf_thesis",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = args.root / "outputs"
    moved = 0

    for d in sorted(args.root.iterdir()):
        if not d.is_dir() or "\\" not in d.name:
            continue
        leaf = d.name.split("\\")[-1]
        family = LEAF_TO_FAMILY.get(leaf)
        if family is None:
            print(f"[skip] unrecognised backslash directory: {d.name}")
            continue

        groups = sorted(p for p in d.iterdir() if p.name.startswith("GROUP_"))
        complete = [g for g in groups if (g / "metrics.json").exists()]
        print(f"[found] {leaf:30s} {len(groups)} groups, "
              f"{len(complete)} with metrics.json  ->  outputs/{family}")

        # A partial group is a fold that died mid-write; leave it in place and say
        # so rather than moving something downstream will read as complete.
        if len(complete) != len(groups):
            incomplete = [g.name for g in groups if g not in complete]
            print(f"   WARNING incomplete: {incomplete}")

        dst = out / family
        if dst.exists():
            print(f"   destination already exists, NOT overwriting: {dst}")
            continue
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dst))
        moved += 1

    print(f"\n{'would move' if args.dry_run else 'moved'} {moved} directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
