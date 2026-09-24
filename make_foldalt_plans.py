#!/usr/bin/env python3
"""Generate alternative fold-assignment plans for the foldalt robustness experiment.

Writes a case_plans directory that run_lgbm_thesis_fold.py accepts unchanged, so a
fold-sensitivity run is a change of INPUT, not of code path:

    python run_lgbm_thesis_fold.py --fold 1 --shard-root Data \
        --case-plans-dir case_plans_foldalt_seed101

Leave-one-block-out is preserved exactly -- whole region-year blocks, 9 held out per
fold, 63 training, no block split across the boundary. Only WHICH blocks share a fold
changes.

    python make_foldalt_plans.py --seed 101
    python make_foldalt_plans.py --seed 101 202 303 404
    python make_foldalt_plans.py --seed 101 --compare      # overlap vs the thesis folds

Output per seed: 8 GROUP_0N.json + ALL_GROUPS_INDEX.json, matching case_plans/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robustness_features import balanced_assignment  # noqa: E402

N_GROUPS = 8


def block_rows(shard_root: Path, blocks: list[str]) -> dict[str, int]:
    return {b: pq.ParquetFile(shard_root / "pair_blocks" / b / "frame.parquet").metadata.num_rows
            for b in blocks}


def build(shard_root: Path, seed: int) -> list[dict]:
    """One plan dict per fold, in the schema load_fold_plan validates."""
    blocks = sorted(p.name for p in (shard_root / "pair_blocks").iterdir() if p.is_dir())
    if len(blocks) != 72:
        raise SystemExit(f"expected 72 region-year blocks, found {len(blocks)}")
    rows = block_rows(shard_root, blocks)
    assign = balanced_assignment([{"case_key": b, "rows": rows[b]} for b in blocks],
                                 N_GROUPS, seed)
    plans = []
    for g in range(1, N_GROUPS + 1):
        test = sorted(b for b, gg in assign.items() if gg == g)
        plans.append({
            "fold_no": g,
            "fold_label": f"GROUP_{g:02d}",
            "assignment_source": f"foldalt, balanced_assignment(seed={seed})",
            "assignment_mode": "row_balanced_whole_pair_blocks",
            "heldout_case_count": len(test),
            "train_case_count": len(blocks) - len(test),
            "heldout_case_keys": test,
            "train_pair_blocks": [b for b in blocks if b not in set(test)],
            "support_blocks": [],
            "heldout_rows": sum(rows[b] for b in test),
        })
    return plans


def write(plans: list[dict], out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for p in plans:
        (out / f"{p['fold_label']}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    index = {
        "source": f"foldalt, balanced_assignment(seed={seed})",
        "group_count": len(plans),
        "case_count": sum(len(p["heldout_case_keys"]) for p in plans),
        "groups": plans,
    }
    (out / "ALL_GROUPS_INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def validate(out: Path) -> None:
    """Validate with the PRODUCTION loader, not a reimplementation of its rules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("F", HERE / "run_lgbm_thesis_fold.py")
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    for g in range(1, N_GROUPS + 1):
        plan, _ = F.load_fold_plan(out, g)
        assert len(plan["train_pair_blocks"]) == 63 and len(plan["heldout_case_keys"]) == 9


def compare(plans: list[dict], case_plans: Path) -> tuple[int, float]:
    """How different is this assignment from the thesis one?

    A 'fold sensitivity' experiment that mostly reproduces the thesis split measures
    nothing, so the overlap is reported rather than assumed.
    """
    thesis = {g: set(json.loads((case_plans / f"GROUP_{g:02d}.json").read_text())["heldout_case_keys"])
              for g in range(1, N_GROUPS + 1)}
    identical, shared = 0, 0
    for p in plans:
        g = p["fold_no"]
        ov = len(thesis[g] & set(p["heldout_case_keys"]))
        shared += ov
        identical += (ov == 9)
        print(f"    GROUP_{g:02d}: {ov}/9 shared with the thesis fold"
              + ("   IDENTICAL" if ov == 9 else ""))
    return identical, shared / (9 * N_GROUPS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[101, 202, 303, 404])
    ap.add_argument("--shard-root", type=Path, default=HERE / "Data")
    ap.add_argument("--case-plans", type=Path, default=HERE / "case_plans")
    ap.add_argument("--out-prefix", type=Path, default=HERE / "case_plans_foldalt_seed")
    ap.add_argument("--compare", action="store_true",
                    help="report overlap with the thesis assignment")
    args = ap.parse_args()

    for seed in args.seed:
        out = Path(f"{args.out_prefix}{seed}")
        plans = build(args.shard_root, seed)
        write(plans, out, seed)
        validate(out)
        n = len(list(out.glob("*.json")))
        print(f"  seed {seed}: wrote {n} json files -> {out.name}  (validated by the production loader)")
        if args.compare:
            identical, frac = compare(plans, args.case_plans)
            print(f"    -> {identical}/8 folds identical to the thesis, "
                  f"{100*frac:.0f}% of held-out blocks shared overall\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
