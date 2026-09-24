#!/usr/bin/env python3
"""Generate the Stage-1 feature-subset JSONs for the ablation experiments.

Writes one JSON list per ablation, consumable by the production fold script, so an
ablation is a change of INPUT rather than of code path:

    python run_lgbm_thesis_fold.py --fold 1 --shard-root Data \
        --case-plans-dir case_plans \
        --features-file features_ablation/ablation_no_aod.json

Group membership comes from robustness_features.feature_tokens(), the same function
robustness_runner uses, so these files and the in-memory experiments cannot disagree.

    python make_ablation_features.py
    python make_ablation_features.py --out-dir features_ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robustness_features import (  # noqa: E402
    feature_tokens, group_sizes, select_features,
)

# Mirrors the EXPERIMENTS registry in robustness_runner.py.
ABLATIONS = {
    "ablation_no_aod":    {"drop": ["aod"],    "keep": None,       "why": "drop satellite AOD"},
    "ablation_no_fire":   {"drop": ["fire"],   "keep": None,       "why": "drop VIIRS/HMS/burned"},
    "ablation_no_merra":  {"drop": ["merra"],  "keep": None,       "why": "drop MERRA-2 reanalysis"},
    "ablation_no_burned": {"drop": ["burned"], "keep": None,       "why": "drop burned area"},
    "ablation_met_only":  {"drop": [],         "keep": "met_only", "why": "keep NARR met + calendar + land/road"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", type=Path, default=HERE / "Data")
    ap.add_argument("--out-dir", type=Path, default=HERE / "features_ablation")
    args = ap.parse_args()

    manifest = json.loads((args.shard_root / "manifest.json").read_text(encoding="utf-8"))
    feats = list(manifest["feature_cols"])
    if len(feats) != 651:
        raise SystemExit(f"manifest has {len(feats)} features, expected 651")

    sizes = group_sizes(feats)
    print(f"  manifest: {sizes['total']} Stage-1 predictors")
    print(f"    aod {sizes['aod']}   merra {sizes['merra']}   fire {sizes['fire']}   "
          f"burned {sizes['burned']}   met_only keeps {sizes['met_only_kept']}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {'experiment':<22}{'kept':>6}{'dropped':>9}   {'note'}")
    for name, spec in ABLATIONS.items():
        kept = select_features(feats, drop=spec["drop"], keep_mode=spec["keep"])
        if not kept:
            raise SystemExit(f"{name}: would keep 0 predictors")
        if len(kept) == len(feats):
            raise SystemExit(f"{name}: dropped nothing -- the token rule matched no "
                             f"predictor, so this ablation would silently be a baseline")
        # the fold script requires a subset of the manifest, in manifest order
        assert kept == [f for f in feats if f in set(kept)]
        (args.out_dir / f"{name}.json").write_text(json.dumps(kept, indent=0), encoding="utf-8")
        print(f"  {name:<22}{len(kept):>6}{len(feats) - len(kept):>9}   {spec['why']}")

    # baseline is the absence of the flag, not a file -- recorded so the set is complete
    print(f"  {'baseline':<22}{len(feats):>6}{0:>9}   run with NO --features-file\n")
    print(f"  wrote {len(ABLATIONS)} json file(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
