#!/usr/bin/env python3
"""
Re-run the whole post-defense pipeline on the RAW shards, locally.

Why this exists
---------------
The published runs read `Data_zerofilled/`, whose parquet frames already had NaN
replaced by 0. Stage 1 is unaffected -- `load_one_block` calls nan_to_num on X either
way, so the Stage-1 design matrix is bit-identical between the two roots (verified:
max|dX| = 0 over 651 x 2,903). The corrector is NOT unaffected: it reads `df`, which
keeps its NaN, so a pre-filled root zeroes every `__isna` flag and drags every
training-pool median toward 0. That is reviewer point 1, and it is why everything
downstream of the corrector has to be produced again.

Everything here therefore points at `Data/`. `fit_corrector_fill_values` now refuses a
pool with zero missing values across all 40 raw inputs, so this cannot silently revert.

Order is dependency order. Each stage is skipped when its completion marker already
exists, so an interrupted run resumes by being started again.

  python run_revision_local.py --dry-run
  python run_revision_local.py
  python run_revision_local.py --only folds final_model
  python run_revision_local.py --skip raster
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARD = ROOT / "Data"
PLANS = ROOT / "case_plans"
OUT = ROOT / "outputs"
FOLDS_OUT = OUT / "lgbm_thesis"
FINAL_OUT = OUT / "final_ontario_model"
# Windows workstation default; on the VM the grid lives beside the code, so this is
# overridable by --grid-dir or PM25_GRID_DIR rather than hardcoded to one machine.
GRID = Path(os.environ.get(
    "PM25_GRID_DIR",
    r"D:\lambda\Ontario_RealTarget_GPD\outputs\ontario_surface_build"))


def stages(args: argparse.Namespace) -> dict:
    return {
        # 8 outer folds. These carry the headline CV metrics and are what SHAP explains.
        "folds": {
            "script": "run_thesis_folds.py",
            "argv": ["--model", "lgbm", "--from", "1", "--to", "8", "--resume",
                     "--shard-root", str(SHARD), "--case-plans-dir", str(PLANS),
                     "--out-root", str(FOLDS_OUT)],
            "done": FOLDS_OUT / "GROUP_08" / "metrics.json",
            "needs": None, "note": "8 outer folds, raw shards",
        },
        # One fit on all 72 blocks. The surface and the external test both use it.
        "final_model": {
            "script": "train_final_ontario_model.py",
            "argv": ["--shard-root", str(SHARD), "--out-dir", str(FINAL_OUT)],
            "done": FINAL_OUT / "stage1_model_bundle.pkl",
            "needs": None, "note": "all 72 blocks, no holdout",
        },
        # Quebec shards were clipped from raw SUPPORT_SHARED and already carry real
        # NaN, so only the model changes here.
        "quebec": {
            "script": "quebec_external_test.py",
            "argv": ["--province", "QC", "--shard-root", str(SHARD),
                     "--external-root", str(ROOT / "Data_external" / "qc"),
                     "--model-dir", str(FINAL_OUT),
                     "--out-dir", str(OUT / "external_qc")],
            "done": OUT / "external_qc" / "qc_metrics.json",
            "needs": "final_model", "note": "monitored QC cells, 2010-2024",
        },
        # Reviewer point 4. Two sample sizes so the sample-size claim is shown, not
        # asserted: if the AOD share or the top-10 membership moves between them, the
        # larger one is the one to report.
        "shap_500": {
            "script": "make_fold_shap.py",
            "argv": ["--folds", "1,2,3,4,5,6,7,8", "--rows-per-fold", "500",
                     "--shard-root", str(SHARD), "--out-root", str(FOLDS_OUT),
                     "--out-dir", str(OUT / "shap_all8_n500")],
            "done": OUT / "shap_all8_n500" / "shap_report.json",
            "needs": "folds", "note": "8-fold SHAP, 500 rows/fold",
        },
        # The headline SHAP run: 1000 held-out rows per fold, both stages, on the
        # repaired correctors. 500 and 2000 bracket it for the sample-size check.
        "shap_1000": {
            "script": "make_fold_shap.py",
            "argv": ["--folds", "1,2,3,4,5,6,7,8", "--rows-per-fold", "1000",
                     "--shard-root", str(SHARD), "--out-root", str(FOLDS_OUT),
                     "--out-dir", str(OUT / "shap_all8_n1000")],
            "done": OUT / "shap_all8_n1000" / "shap_report.json",
            "needs": "folds", "note": "8-fold SHAP, 1000 rows/fold (headline)",
        },
        "shap_2000": {
            "script": "make_fold_shap.py",
            "argv": ["--folds", "1,2,3,4,5,6,7,8", "--rows-per-fold", "2000",
                     "--shard-root", str(SHARD), "--out-root", str(FOLDS_OUT),
                     "--out-dir", str(OUT / "shap_all8_n2000")],
            "done": OUT / "shap_all8_n2000" / "shap_report.json",
            "needs": "folds", "note": "8-fold SHAP, 2000 rows/fold (sensitivity)",
        },
        # The long pole: 12 experiments x 8 folds, plus 41 leave-one-cell-out fits.
        "robustness": {
            "script": "run_all_robustness.py",
            "argv": ["--shard-root", str(SHARD), "--case-plans-dir", str(PLANS),
                     "--out-dir", str(OUT / "robustness"),
                     "--jobs", str(args.jobs)],
            "done": OUT / "robustness" / "robustness_summary.csv",
            "needs": None, "note": "12 ablations/fold-variants + LOCO",
        },
        # 46.7M cell-days. Grid inputs were never zero-filled; only the model changed.
        # cells-per-batch is the memory dial: one cell-decade is ~11 MB at 651 float32
        # and build_derivatives roughly doubles that during the concat.
        "raster": {
            "script": "predict_raster_cells.py",
            "argv": ["--grid-dir", str(args.grid_dir), "--shard-root", str(SHARD),
                     "--model-dir", str(FINAL_OUT),
                     "--cells-per-batch", str(args.cells_per_batch),
                     "--out-dir", str(OUT / "raster_prediction")],
            "done": OUT / "raster_prediction" / "grid_predictions_lgbm.parquet",
            "needs": "final_model", "note": "46.7M Ontario cell-days",
            "requires": [args.grid_dir / "static_features.parquet",
                         args.grid_dir / "temporal_tables"],
        },
    }


def hms(s: float) -> str:
    s = int(round(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n# {' '.join(cmd)}\n# started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            # The console is cp1252 on this box and child output carries U+FFFD (from
            # its own decoding of box-drawing/plus-minus bytes). Writing that straight
            # to stdout raises UnicodeEncodeError and kills the DRIVER while the child
            # has already succeeded -- which is exactly what it did. Never let a
            # relay-encoding problem fail a stage that worked.
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()
            fh.write(line); fh.flush()
        code = p.wait()
        fh.write(f"# exit={code} finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=3,
                    help="robustness concurrency; threads per job are clamped to match")
    ap.add_argument("--cells-per-batch", type=int, default=120,
                    help="raster memory dial (16 GB box: 120; 125 GB VM: 600)")
    ap.add_argument("--grid-dir", type=Path, default=GRID,
                    help="ontario_surface_build (temporal_tables/ + static_features.parquet)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Belt and braces alongside the per-line guard in run(): ask for UTF-8 with
    # replacement so the relay degrades instead of raising.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    st = stages(args)
    names = [n for n in st if (not args.only or n in set(args.only))
             and n not in set(args.skip)]

    logs = OUT / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Several fits run at once only inside run_all_robustness; LightGBM's n_jobs
    # overrides OMP_NUM_THREADS, so the per-job clamp has to be passed explicitly or
    # every job grabs all 12 cores and they spin against each other at OMP barriers.
    env_threads = max(1, (os.cpu_count() or 4) // max(1, args.jobs))
    os.environ["PM25_LGBM_THREADS"] = str(env_threads)

    plan = []
    for n in names:
        s = st[n]
        missing = [str(q) for q in s.get("requires", []) if not q.exists()]
        if s["done"].exists() and not args.force:
            state = "done"
        elif missing:
            state = "MISSING INPUT"
        else:
            state = "todo"
        plan.append((n, s, state, missing))

    print("=" * 76)
    print(f"[plan] shard-root {SHARD}")
    print(f"[plan] robustness jobs={args.jobs} threads/job={env_threads}  "
          f"raster cells/batch={args.cells_per_batch}")
    for n, s, state, missing in plan:
        print(f"  {n:<13} {state:<14} {s['note']}")
        for m in missing:
            print(f"                 needs {m}")
    print("=" * 76, flush=True)
    if args.dry_run:
        return 0

    results, t0, failed = [], time.perf_counter(), set()
    for n, s, state, missing in plan:
        if state == "MISSING INPUT":
            print(f"[unavailable] {n}: missing {missing[0]}", flush=True)
            results.append({"stage": n, "status": "unavailable", "seconds": 0.0})
            failed.add(n)
            continue
        if state == "done":
            print(f"[skip-existing] {n}", flush=True)
            results.append({"stage": n, "status": "skipped", "seconds": 0.0})
            continue
        if s["needs"] in failed:
            print(f"[blocked] {n}: {s['needs']} did not succeed", flush=True)
            results.append({"stage": n, "status": "blocked", "seconds": 0.0})
            failed.add(n)
            continue
        print(f"\n{'='*76}\n[run] {n}  ({s['note']})  {datetime.now():%H:%M:%S}\n{'='*76}",
              flush=True)
        t = time.perf_counter()
        code = run([sys.executable, str(ROOT / s["script"])] + s["argv"], logs / f"{n}.log")
        el = time.perf_counter() - t
        ok = code == 0 and s["done"].exists()
        if not ok:
            failed.add(n)
        results.append({"stage": n, "status": "ok" if ok else "failed",
                        "code": code, "seconds": round(el, 1)})
        print(f"[{'ok' if ok else 'failed'}] {n} in {hms(el)}", flush=True)

    print("\n" + "=" * 76)
    for r in results:
        print(f"  {r['stage']:<13} {r['status']:<8} {hms(r['seconds']):>10}")
    bad = [r for r in results if r["status"] in ("failed", "blocked")]
    print(f"[done] total {hms(time.perf_counter()-t0)}  failures={len(bad)}")
    (logs / "run_revision_local_summary.json").write_text(
        json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                    "results": results, "failures": len(bad)}, indent=2),
        encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
