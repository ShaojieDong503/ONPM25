#!/usr/bin/env python3
"""
Run a range of thesis outer folds, one after another.

Drives run_<model>_thesis_fold.py once per fold, sequentially, and reports a single
pass/fail summary at the end.

Quick start
-----------
  python run_thesis_folds.py --model lgbm --from 1 --to 8 \
      --shard-root     D:\\lambda\\Post_Defense_Code\\Data_zerofilled \
      --case-plans-dir D:\\lambda\\Post_Defense_Code\\case_plans \
      --out-root       D:\\lambda\\Post_Defense_Code\\outputs\\lgbm_thesis_zerofilled

  python run_thesis_folds.py --model lgbm --from 3 --to 5        # a sub-range
  python run_thesis_folds.py --model lgbm --folds 2,4,7          # specific folds
  python run_thesis_folds.py --model lgbm --resume               # skip finished folds
  python run_thesis_folds.py --model lgbm --dry-run              # print commands only

Failure policy
--------------
The 8 folds are independent, so by default a failing fold does NOT stop the others --
every requested fold is attempted. At the end one [fail] line is printed per failed
fold with its exit code and log path, then `failures=N`, and the process exits 1 if
N > 0. `[done] ... failures=0` with exit 0 is the only success signal. Use
--stop-on-error to halt at the first failure instead.

Logs
----
Per-fold stdout/stderr is streamed to the console and written to
  <out-root>/_logs/<model>_fold_<n>.log
Exit codes are recorded in
  <out-root>/_logs/.status/<n>
so a run killed mid-way (OOM, reboot) is reported as `missing` rather than silently
counting as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = {
    "lgbm": ROOT / "run_lgbm_thesis_fold.py",
    "xgb": ROOT / "run_xgb_thesis_fold.py",
    "rf": ROOT / "run_rf_thesis_fold.py",
}

MIN_FOLD, MAX_FOLD = 1, 8

#: Written by each fold script when it finishes; used by --resume.
COMPLETION_MARKER = "metrics.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a range of thesis outer folds sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Quick start")[1] if "Quick start" in __doc__ else None,
    )
    ap.add_argument("--model", choices=sorted(MODELS), default="lgbm",
                    help="which learner's fold script to drive (default: lgbm)")
    ap.add_argument("--from", dest="fold_from", type=int, default=MIN_FOLD,
                    metavar="N", help=f"first fold, {MIN_FOLD}-{MAX_FOLD} (default: {MIN_FOLD})")
    ap.add_argument("--to", dest="fold_to", type=int, default=MAX_FOLD,
                    metavar="N", help=f"last fold, inclusive (default: {MAX_FOLD})")
    ap.add_argument("--folds", default=None, metavar="LIST",
                    help="explicit comma-separated folds, e.g. 2,4,7. Overrides --from/--to")

    ap.add_argument("--shard-root", type=Path, default=None,
                    help="passed through; omit to use the fold script's own default")
    ap.add_argument("--case-plans-dir", type=Path, default=None,
                    help="passed through; needed when case_plans/ is not under --shard-root")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="passed through; also where logs and status files are written")

    ap.add_argument("--resume", action="store_true",
                    help=f"skip folds whose <out-root>/GROUP_0N/{COMPLETION_MARKER} already exists")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="halt at the first failing fold instead of attempting the rest")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, then exit")
    ap.add_argument("--verbose", action="store_true",
                    help="pass --verbose to each fold script")
    return ap.parse_args()


def resolve_folds(args: argparse.Namespace) -> list[int]:
    if args.folds:
        try:
            folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"[error] --folds must be a comma-separated list of integers: "
                             f"{args.folds!r}")
    else:
        if args.fold_from > args.fold_to:
            raise SystemExit(f"[error] --from {args.fold_from} is greater than "
                             f"--to {args.fold_to}")
        folds = list(range(args.fold_from, args.fold_to + 1))

    bad = [f for f in folds if not (MIN_FOLD <= f <= MAX_FOLD)]
    if bad:
        raise SystemExit(f"[error] fold(s) {bad} outside the valid range "
                         f"{MIN_FOLD}-{MAX_FOLD}")
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(folds))


def build_command(args: argparse.Namespace, fold: int) -> list[str]:
    cmd = [sys.executable, str(MODELS[args.model]), "--fold", str(fold)]
    if args.shard_root is not None:
        cmd += ["--shard-root", str(args.shard_root)]
    if args.case_plans_dir is not None:
        cmd += ["--case-plans-dir", str(args.case_plans_dir)]
    if args.out_root is not None:
        cmd += ["--out-root", str(args.out_root)]
    if args.verbose:
        cmd += ["--verbose"]
    return cmd


def fold_is_complete(out_root: Path | None, fold: int) -> bool:
    if out_root is None:
        return False
    return (out_root / f"GROUP_{fold:02d}" / COMPLETION_MARKER).exists()


def hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"


def preflight(args: argparse.Namespace) -> None:
    """Fail before launching anything rather than on fold 1 after a long wait."""
    script = MODELS[args.model]
    if not script.exists():
        raise SystemExit(f"[error] fold script not found: {script}")

    if args.shard_root is not None:
        if not args.shard_root.exists():
            raise SystemExit(f"[error] --shard-root does not exist: {args.shard_root}")
        for required in ("manifest.json", "pair_blocks"):
            if not (args.shard_root / required).exists():
                raise SystemExit(
                    f"[error] --shard-root is missing {required}: {args.shard_root}\n"
                    f"        the shard root is the directory CONTAINING pair_blocks/, "
                    f"not pair_blocks/ itself")

    # case_plans defaults to <shard-root>/case_plans; check whichever applies.
    plans_dir = args.case_plans_dir
    if plans_dir is None and args.shard_root is not None:
        plans_dir = args.shard_root / "case_plans"
    if plans_dir is not None:
        if not plans_dir.exists():
            raise SystemExit(
                f"[error] case plans directory not found: {plans_dir}\n"
                f"        pass --case-plans-dir explicitly if case_plans/ lives "
                f"elsewhere than under --shard-root")
        missing = [f"GROUP_{f:02d}.json" for f in resolve_folds(args)
                   if not (plans_dir / f"GROUP_{f:02d}.json").exists()]
        if missing:
            raise SystemExit(f"[error] missing case plan(s) in {plans_dir}: {missing}")


def main() -> int:
    args = parse_args()
    folds = resolve_folds(args)
    preflight(args)

    out_root = args.out_root
    log_dir = (out_root / "_logs") if out_root else (ROOT / "_logs")
    status_dir = log_dir / ".status"
    if not args.dry_run:
        status_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[plan] {args.model} folds {folds[0]}-{folds[-1]}"
          f"{'' if folds == list(range(folds[0], folds[-1] + 1)) else f' (explicit: {folds})'}"
          f"   sequential, one at a time")
    print(f"[script]      {MODELS[args.model]}")
    print(f"[shard_root]  {args.shard_root or '(fold script default)'}")
    print(f"[case_plans]  {args.case_plans_dir or '(<shard-root>/case_plans)'}")
    print(f"[out_root]    {out_root or '(fold script default)'}")
    print(f"[logs]        {log_dir}")
    print("=" * 78, flush=True)

    if args.dry_run:
        # Honour --resume here too, or the dry run advertises work that the real run
        # would skip.
        for f in folds:
            if args.resume and fold_is_complete(out_root, f):
                print(f"# fold {f}: SKIPPED by --resume "
                      f"({out_root / f'GROUP_{f:02d}' / COMPLETION_MARKER} exists)")
                continue
            print(" ".join(f'"{c}"' if " " in c else c for c in build_command(args, f)))
        return 0

    results: list[dict] = []
    run_start = time.perf_counter()
    interrupted = False

    for i, fold in enumerate(folds, 1):
        tag = f"fold {fold}"
        if args.resume and fold_is_complete(out_root, fold):
            print(f"[skip-existing] {tag}   {out_root / f'GROUP_{fold:02d}' / COMPLETION_MARKER}",
                  flush=True)
            results.append({"fold": fold, "status": "skipped", "code": 0, "seconds": 0.0})
            continue

        cmd = build_command(args, fold)
        log_path = log_dir / f"{args.model}_fold_{fold}.log"
        status_path = status_dir / str(fold)
        status_path.unlink(missing_ok=True)

        print(f"\n[launch] {args.model} {tag}  ({i}/{len(folds)})  "
              f"{datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        print(f"[log]    {log_path}", flush=True)
        # The fold scripts write their detail to their own training.log rather than to
        # stdout, so the console here stays quiet for minutes during a fit. Say where
        # the live detail actually is, or a healthy run looks like a hung one.
        if out_root is not None:
            print(f"[watch]  {out_root / f'GROUP_{fold:02d}' / 'training.log'}"
                  f"   (live per-block and fitting progress)", flush=True)

        t0 = time.perf_counter()
        code = 1
        try:
            # Stream to console and to the log file at the same time, so a long run is
            # watchable live and still leaves a complete record.
            with log_path.open("w", encoding="utf-8", errors="replace") as fh:
                fh.write(f"# {' '.join(cmd)}\n")
                fh.write(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
                fh.flush()
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", bufsize=1)
                try:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        fh.write(line)
                        # Flush per line: without this the log sits in the buffer for
                        # the whole fit, so `tail -f` on it shows nothing until the
                        # fold ends -- which reads as a hung run.
                        fh.flush()
                    code = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
                finally:
                    fh.write(f"\n# exit={code}  finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        except KeyboardInterrupt:
            interrupted = True
            elapsed = time.perf_counter() - t0
            status_path.write_text("interrupted\n", encoding="utf-8")
            results.append({"fold": fold, "status": "interrupted", "code": 130,
                            "seconds": elapsed, "log": str(log_path)})
            print(f"\n[abort] interrupted during {tag} after {hms(elapsed)}", flush=True)
            break

        elapsed = time.perf_counter() - t0
        status_path.write_text(f"{code}\n", encoding="utf-8")
        results.append({"fold": fold, "status": "ok" if code == 0 else "failed",
                        "code": code, "seconds": elapsed, "log": str(log_path)})

        if code == 0:
            print(f"[ok]   {args.model} {tag} in {hms(elapsed)}", flush=True)
        else:
            print(f"[fail] {args.model} {tag} exited {code} after {hms(elapsed)}   "
                  f"see {log_path}", flush=True)
            if args.stop_on_error:
                print("[abort] --stop-on-error set; not attempting the remaining folds",
                      flush=True)
                break

    total = time.perf_counter() - run_start

    # Any fold that never recorded a status was killed without reporting; count it as a
    # failure so a truncated run can never look like a clean one.
    attempted = {r["fold"] for r in results}
    for fold in folds:
        if fold not in attempted:
            results.append({"fold": fold, "status": "not run", "code": None, "seconds": 0.0})

    failed = [r for r in results if r["status"] in ("failed", "interrupted")]
    not_run = [r for r in results if r["status"] == "not run"]
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print()
    print("=" * 78)
    for r in results:
        if r["status"] == "ok":
            print(f"  fold {r['fold']}  ok          {hms(r['seconds']):>10}")
        elif r["status"] == "skipped":
            print(f"  fold {r['fold']}  skipped     (already complete)")
        elif r["status"] == "not run":
            print(f"  fold {r['fold']}  NOT RUN")
        else:
            print(f"  fold {r['fold']}  {r['status'].upper():<11} exit={r['code']}   "
                  f"{r.get('log', '')}")
    print("-" * 78)
    print(f"[done] {args.model} folds {folds[0]}-{folds[-1]} in {hms(total)}   "
          f"ok={len(ok)} skipped={len(skipped)} failures={len(failed) + len(not_run)}")
    print(f"[logs] {log_dir}")
    print("=" * 78, flush=True)

    summary_path = log_dir / f"{args.model}_run_summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "folds_requested": folds,
        "shard_root": str(args.shard_root) if args.shard_root else None,
        "case_plans_dir": str(args.case_plans_dir) if args.case_plans_dir else None,
        "out_root": str(out_root) if out_root else None,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total, 1),
        "interrupted": interrupted,
        "results": results,
        "failures": len(failed) + len(not_run),
    }, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)

    return 1 if (failed or not_run) else 0


if __name__ == "__main__":
    raise SystemExit(main())
