#!/usr/bin/env python3
"""
Run the whole robustness suite end to end, in one command.

Drives `robustness_runner.py` through every experiment in cost order (cheapest
first), fans LOCO out over its 41 cells, merges the fan-out, and writes the summary
table. Everything is resumable: an experiment whose predictions already exist is
skipped, so the suite can be stopped and restarted freely.

    python run_all_robustness.py                    # everything, ~20 h
    python run_all_robustness.py --dry-run          # show the plan and exit
    python run_all_robustness.py --skip oof_corrector   # leave out the 6 h one
    python run_all_robustness.py --only baseline spatial_cv
    python run_all_robustness.py --stage cheap      # the 7 quick table rows only

Failure policy
--------------
The experiments are independent, so a failure does NOT stop the rest -- every one is
attempted, then a summary lists what failed with its log path and the process exits 1.
`[done] ... failures=0` is the only success signal. Use --stop-on-error to halt early.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "robustness_runner.py"

#: name -> (stage, rough minutes, note). Order here is execution order.
PLAN = [
    ("baseline",           "cheap", 50,  "all 651 predictors, 8-fold block CV"),
    ("ablation_no_aod",    "cheap", 50,  "drop 12 AOD predictors"),
    ("ablation_no_burned", "cheap", 50,  "drop 6 burned-area predictors"),
    ("ablation_no_fire",   "cheap", 45,  "drop 82 VIIRS/HMS/burned predictors"),
    ("ablation_no_merra",  "cheap", 25,  "drop 532 MERRA-2 predictors"),
    ("ablation_met_only",  "cheap", 10,  "keep 25 meteorology/calendar/land predictors"),
    ("spatial_cv",         "cheap", 45,  "leave-cells-out, k=7"),
    ("foldalt_1",          "folds", 50,  "alternative block->group assignment, seed 101"),
    ("foldalt_2",          "folds", 50,  "alternative block->group assignment, seed 202"),
    ("foldalt_3",          "folds", 50,  "alternative block->group assignment, seed 303"),
    ("foldalt_4",          "folds", 50,  "alternative block->group assignment, seed 404"),
    ("oof_corrector",      "heavy", 360, "corrector on out-of-fold residuals, 64 fits"),
]

N_LOCO_CELLS = 41
STAGES = ("cheap", "folds", "heavy")


def hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, metavar="NAME",
                    help="run only these experiments")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME",
                    help="skip these experiments")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="run only one cost tier")
    ap.add_argument("--shard-root", type=Path, default=ROOT / "Data")
    ap.add_argument("--case-plans-dir", type=Path, default=ROOT / "case_plans")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "robustness")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="run N experiments concurrently. LightGBM flattens past ~8 threads "
                         "per fit, so on a big box 4 jobs x 8 threads beats 1 job x 32.")
    ap.add_argument("--total-threads", type=int, default=None,
                    help="threads to divide across jobs (default: CPU count)")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    ap.add_argument("--quick", action="store_true",
                    help="n_estimators=100 plumbing test; NOT scientific output")
    return ap.parse_args()


def select(args: argparse.Namespace) -> list[tuple]:
    plan = PLAN
    if args.stage:
        plan = [p for p in plan if p[1] == args.stage]
    if args.only:
        unknown = [n for n in args.only if n not in {p[0] for p in PLAN}]
        if unknown:
            raise SystemExit(f"[error] unknown experiment(s): {unknown}")
        plan = [p for p in plan if p[0] in set(args.only)]
    if args.skip:
        plan = [p for p in plan if p[0] not in set(args.skip)]
    return plan


def base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(RUNNER),
           "--shard-root", str(args.shard_root),
           "--case-plans-dir", str(args.case_plans_dir),
           "--out-dir", str(args.out_dir)]
    if args.force:
        cmd.append("--force")
    if args.quick:
        cmd.append("--quick")
    return cmd


def already_done(name: str, out_dir: Path) -> bool:
    return (out_dir / f"{name}_predictions.parquet").exists()


def run(cmd: list[str], log_path: Path, threads: int | None = None,
        quiet: bool = False) -> int:
    """Run a step, streaming to its log (and the console unless quiet)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
            env[k] = str(threads)
        # LightGBM ignores OMP_NUM_THREADS when n_jobs is set (-1 by default), so
        # the child must clamp the estimator itself or concurrent jobs oversubscribe.
        env["PM25_LGBM_THREADS"] = str(threads)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n# {' '.join(cmd)}\n# started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            fh.write(line)
            fh.flush()
        code = proc.wait()
        fh.write(f"# exit={code}  finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return code


def job_split(args: argparse.Namespace) -> tuple[int, int]:
    """How many concurrent jobs, and how many threads each gets.

    LightGBM's within-tree parallelism is flat past ~8 threads (3.26x at 8, 3.30x at
    12), so on a large box several narrow jobs finish far sooner than one wide one.
    """
    total = args.total_threads or os.cpu_count() or 8
    jobs = max(1, int(args.jobs))
    return jobs, max(1, total // jobs)


def run_loco(args: argparse.Namespace, log_dir: Path) -> int:
    """LOCO is fanned out over its 41 cells so a failure costs one cell, not 6 hours."""
    out_dir = args.out_dir
    if already_done("loco", out_dir) and not args.force:
        print("[skip-existing] loco (merged result present)", flush=True)
        return 0

    todo = [i for i in range(N_LOCO_CELLS)
            if args.force or not (out_dir / f"loco_fold{i}_predictions.parquet").exists()]
    if N_LOCO_CELLS - len(todo):
        print(f"[skip-existing] {N_LOCO_CELLS - len(todo)} loco cell(s) already computed",
              flush=True)

    jobs, threads = job_split(args)
    failures = 0

    def one(i: int) -> tuple[int, int]:
        # One log per cell: concurrent writers would otherwise interleave.
        return i, run(base_cmd(args) + ["--experiment", "loco", "--fold", str(i)],
                      log_dir / f"loco_cell{i}.log", threads=threads, quiet=jobs > 1)

    print(f"[loco] {len(todo)} cell(s), {jobs} job(s) x {threads} threads", flush=True)
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for n, (i, code) in enumerate(ex.map(one, todo), 1):
            if code == 0:
                print(f"[loco] cell {i} ok  ({n}/{len(todo)})", flush=True)
            else:
                failures += 1
                print(f"[fail] loco cell {i} exited {code}  "
                      f"see {log_dir / f'loco_cell{i}.log'}", flush=True)

    if failures:
        print(f"[loco] {failures} cell(s) failed; NOT merging a partial result", flush=True)
        return 1

    print("\n[loco] merging 41 partials", flush=True)
    return run(base_cmd(args) + ["--merge", "loco"], log_dir / "loco.log")


def main() -> int:
    args = parse_args()
    plan = select(args)
    out_dir = args.out_dir
    log_dir = out_dir / "_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = [p for p in plan if args.force or not already_done(p[0], out_dir)]
    est = sum(p[2] for p in pending)

    print("=" * 78)
    print(f"[plan] {len(plan)} experiment(s), {len(pending)} still to run")
    print(f"[plan] estimated {est} min (~{est/60:.1f} h) for the outstanding ones")
    print(f"[shards]   {args.shard_root}")
    print(f"[out_dir]  {out_dir}")
    print("=" * 78)
    for name, stage, mins, note in plan:
        mark = "done" if already_done(name, out_dir) and not args.force else f"~{mins}m"
        print(f"  {name:<20} {stage:<6} {mark:<7} {note}")
    print("=" * 78, flush=True)

    if args.quick:
        print("[warning] --quick: n_estimators=100. Plumbing test, NOT results.\n", flush=True)
    if args.dry_run:
        return 0

    results, t_all = [], time.perf_counter()
    jobs, threads = job_split(args)

    todo = [p for p in plan if args.force or not already_done(p[0], out_dir)]
    for p in plan:
        if p not in todo:
            print(f"[skip-existing] {p[0]}", flush=True)
            results.append({"experiment": p[0], "status": "skipped", "code": 0, "seconds": 0.0})

    # LOCO runs on its own: it manages a 41-way fan-out internally, so giving it one
    # slot alongside the others would leave most of the machine idle.
    loco = [p for p in todo if p[0] == "loco"]
    todo = [p for p in todo if p[0] != "loco"]

    def one(item) -> tuple[str, int, float]:
        name, _stage, _mins, note = item
        t0 = time.perf_counter()
        print(f"[start] {name}  ({note})  {datetime.now():%H:%M:%S}", flush=True)
        code = run(base_cmd(args) + ["--experiment", name], log_dir / f"{name}.log",
                   threads=threads, quiet=jobs > 1)
        return name, code, time.perf_counter() - t0

    if todo:
        print(f"\n{'=' * 78}\n[run] {len(todo)} experiment(s), "
              f"{jobs} job(s) x {threads} threads\n{'=' * 78}", flush=True)
        try:
            with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
                for name, code, elapsed in ex.map(one, todo):
                    status = "ok" if code == 0 else "failed"
                    results.append({"experiment": name, "status": status, "code": code,
                                    "seconds": elapsed,
                                    "log": str(log_dir / f"{name}.log")})
                    print(f"[{status}] {name} in {hms(elapsed)}", flush=True)
        except KeyboardInterrupt:
            print("\n[abort] interrupted", flush=True)

    for name, _s, _m, note in loco:
        print(f"\n{'=' * 78}\n[run] loco  ({note})", flush=True)
        t0 = time.perf_counter()
        code = run_loco(args, log_dir)
        elapsed = time.perf_counter() - t0
        status = "ok" if code == 0 else "failed"
        results.append({"experiment": "loco", "status": status, "code": code,
                        "seconds": elapsed, "log": str(log_dir / "loco.log")})
        print(f"[{status}] loco in {hms(elapsed)}", flush=True)

    # Final summary table, built from whatever completed.
    print(f"\n{'=' * 78}\n[summary] building the table", flush=True)
    run(base_cmd(args) + ["--summarize"], log_dir / "summarize.log")

    total = time.perf_counter() - t_all
    failed = [r for r in results if r["status"] in ("failed", "interrupted")]
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print("\n" + "=" * 78)
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['experiment']:<20} ok         {hms(r['seconds']):>10}")
        elif r["status"] == "skipped":
            print(f"  {r['experiment']:<20} skipped    (already done)")
        else:
            print(f"  {r['experiment']:<20} {r['status'].upper():<10} exit={r['code']}  "
                  f"{r.get('log', '')}")
    print("-" * 78)
    print(f"[done] {len(ok)} ran, {len(skipped)} skipped, failures={len(failed)}  "
          f"total {hms(total)}")
    print(f"[table] {out_dir / 'robustness_summary.md'}")
    print(f"[logs]  {log_dir}")
    print("=" * 78, flush=True)

    (log_dir / "run_all_summary.json").write_text(json.dumps({
        "finished": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total, 1), "results": results,
        "failures": len(failed), "shard_root": str(args.shard_root),
    }, indent=2), encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
