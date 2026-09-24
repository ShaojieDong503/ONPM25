#!/usr/bin/env python3
"""
Ontario PM2.5 trainable-shard data audit.

Purpose
-------
Validate the already-built trainable region-year blocks BEFORE model fitting.
This script does not retrain models and does not modify input data.

Default shard layout:
  <shard_root>/
    manifest.json
    pair_blocks/
      PAIR_<year>_<region>/
        frame.parquet

Checks include:
- expected 2012-2023 x 6 regions = 72 blocks
- row counts and duplicate cell-days
- required key columns and target integrity
- 651-feature width
- canonical feature -> stored parquet-name resolution
- historical rolling-name repair (canonical _roll*_ -> stored _lag1_roll*_)
- target-leakage name scan
- lag/rolling feature-count sanity
- schema consistency across blocks
- NaN / inf / all-NaN / constant-column diagnostics
- station/cell/date reconciliation
- cross-block duplicate cell-day detection
- machine-readable JSON summary + CSV block/feature audits + human-readable log

The data are NEVER changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EXPECTED_YEARS = list(range(2012, 2024))
EXPECTED_REGIONS = ["Central", "East", "North", "Toronto", "West_NH", "West_SW"]
EXPECTED_BLOCKS = 72
EXPECTED_ROWS = 169_882
EXPECTED_FEATURES = 651
EXPECTED_TEMPORAL_BASES = 73
EXPECTED_ROLLING = 438
EXPECTED_TEMPORAL_DERIVED = 511
EXPECTED_GRID_CELLS = 41

KEY_COLS = [
    "date",
    "year",
    "fold_region",
    "CanOSSEM_RASTER_CELL",
    "pm25",
    "_year_region_pair",
    "naps_id",
    "station_name",
]

FORBIDDEN_FEATURE_PATTERNS = [
    # Detect response/target leakage without rejecting legitimate predictor names
    # such as aod_obs_flag.
    re.compile(r"(^|_)pm25($|_)", re.I),
    re.compile(r"(^|_)obs_pm25($|_)", re.I),
    re.compile(r"(^|_)observed_pm25($|_)", re.I),
    re.compile(r"(^|_)target(?:_pm25)?($|_)", re.I),
    re.compile(r"(^|_)resid(?:ual)?(?:_stage1)?($|_)", re.I),
]


def shard_frame_column(canonical: str) -> str:
    """Canonical feature name -> stored repaired-shard name."""
    if "_roll3_" in canonical or "_roll7_" in canonical:
        return canonical.replace("_roll3_", "_lag1_roll3_").replace(
            "_roll7_", "_lag1_roll7_"
        )
    return canonical


def canonical_from_stored(stored: str) -> str:
    """Reverse helper used only when auditing a standalone CSV without manifest."""
    return stored.replace("_lag1_roll3_", "_roll3_").replace(
        "_lag1_roll7_", "_roll7_"
    )


def expected_pair_aliases(year: int, region: str) -> set[str]:
    """Accept the common pair-key spellings used by the shard builder."""
    return {
        f"{year}|{region}",
        f"PAIR_{year}_{region}",
        f"{year}_{region}",
    }


@dataclass
class Check:
    name: str
    status: str  # PASS / WARN / FAIL
    observed: object
    expected: object
    detail: str = ""


class Audit:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.checks: List[Check] = []

    def add(self, name: str, ok: bool, observed, expected, detail="", warn=False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        c = Check(name, status, observed, expected, detail)
        self.checks.append(c)
        msg = f"[{status}] {name} | observed={observed!r} | expected={expected!r}"
        if detail:
            msg += f" | {detail}"
        if status == "FAIL":
            self.logger.error(msg)
        elif status == "WARN":
            self.logger.warning(msg)
        else:
            self.logger.info(msg)

    @property
    def fail_count(self):
        return sum(c.status == "FAIL" for c in self.checks)

    @property
    def warn_count(self):
        return sum(c.status == "WARN" for c in self.checks)


def configure_logging(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("pm25_data_audit")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_manifest(shard_root: Path) -> Optional[dict]:
    p = shard_root / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def enumerate_parquet_blocks(shard_root: Path) -> List[Tuple[str, Path]]:
    pair_root = shard_root / "pair_blocks"
    if not pair_root.exists():
        return []
    out = []
    for d in sorted(pair_root.glob("PAIR_*")):
        p = d / "frame.parquet"
        if p.exists():
            out.append((d.name, p))
    return out


def parse_block_name(block_name: str) -> Tuple[Optional[int], Optional[str]]:
    m = re.fullmatch(r"PAIR_(\d{4})_(.+)", block_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def numeric_summary(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    finite = x[np.isfinite(x)]
    if finite.empty:
        return {
            "count": int(len(s)),
            "missing": int(x.isna().sum()),
            "finite": 0,
            "zero": 0,
            "n_unique": int(s.nunique(dropna=True)),
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
        }
    q = finite.quantile([0.01, 0.5, 0.99])
    return {
        "count": int(len(s)),
        "missing": int(x.isna().sum()),
        "finite": int(len(finite)),
        "zero": int((finite == 0).sum()),
        "n_unique": int(s.nunique(dropna=True)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(finite.min()),
        "p01": float(q.loc[0.01]),
        "p50": float(q.loc[0.5]),
        "p99": float(q.loc[0.99]),
        "max": float(finite.max()),
    }


def scan_forbidden_feature_names(feature_names: Sequence[str]) -> List[str]:
    bad = []
    for c in feature_names:
        if any(p.search(c) for p in FORBIDDEN_FEATURE_PATTERNS):
            bad.append(c)
    return bad


def resolve_features(
    available_cols: Sequence[str], canonical_features: Sequence[str]
) -> Tuple[List[str], List[str], List[str]]:
    available = set(available_cols)
    direct, translated, absent = [], [], []
    for c in canonical_features:
        if c in available:
            direct.append(c)
        else:
            stored = shard_frame_column(c)
            if stored in available:
                translated.append(c)
            else:
                absent.append(c)
    return direct, translated, absent


def audit_one_frame(
    df: pd.DataFrame,
    block_name: str,
    canonical_features: Sequence[str],
    audit: Audit,
    expected_year: Optional[int],
    expected_region: Optional[str],
    collect_feature_stats: bool = True,
) -> Tuple[dict, List[dict], set]:
    row = {"block": block_name, "rows": int(len(df))}
    feature_stats: List[dict] = []

    missing_keys = [c for c in KEY_COLS if c not in df.columns]
    audit.add(
        f"{block_name}: required key columns",
        not missing_keys,
        missing_keys,
        "none missing",
        "Missing keys prevent a valid training block.",
    )
    if missing_keys:
        return row, feature_stats, set()

    # Basic identity
    if expected_year is not None:
        years = sorted(pd.Series(df["year"]).dropna().astype(int).unique().tolist())
        audit.add(
            f"{block_name}: year identity",
            years == [expected_year],
            years,
            [expected_year],
        )

    if expected_region is not None:
        regions = sorted(df["fold_region"].dropna().astype(str).unique().tolist())
        audit.add(
            f"{block_name}: region identity",
            regions == [expected_region],
            regions,
            [expected_region],
        )

    # Date parsing and agreement
    dates = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(dates.isna().sum())
    audit.add(f"{block_name}: parseable dates", bad_dates == 0, bad_dates, 0)

    if bad_dates == 0:
        year_mismatch = int((dates.dt.year != pd.to_numeric(df["year"], errors="coerce")).sum())
        audit.add(
            f"{block_name}: date.year == year",
            year_mismatch == 0,
            year_mismatch,
            0,
        )

    # Pair key
    if expected_year is not None and expected_region is not None:
        vals = set(df["_year_region_pair"].dropna().astype(str).unique())
        aliases = expected_pair_aliases(expected_year, expected_region)
        audit.add(
            f"{block_name}: pair-key identity",
            bool(vals) and vals.issubset(aliases),
            sorted(vals),
            sorted(aliases),
            "Accepts common builder spellings such as 2012|Central and PAIR_2012_Central.",
        )

    # Duplicates
    dup_n = int(df.duplicated(["CanOSSEM_RASTER_CELL", "date"]).sum())
    audit.add(
        f"{block_name}: duplicate cell-days",
        dup_n == 0,
        dup_n,
        0,
        "Duplicate spatial-day rows are a hard data-integrity failure.",
    )

    # Target
    target_nan = int(df["pm25"].isna().sum())
    audit.add(f"{block_name}: target NaN", target_nan == 0, target_nan, 0)
    target_finite = np.isfinite(pd.to_numeric(df["pm25"], errors="coerce"))
    target_nonfinite = int((~target_finite).sum())
    audit.add(
        f"{block_name}: target finite",
        target_nonfinite == 0,
        target_nonfinite,
        0,
    )

    # Feature width and name resolution
    non_keys = [c for c in df.columns if c not in KEY_COLS]
    audit.add(
        f"{block_name}: stored feature width",
        len(non_keys) == EXPECTED_FEATURES,
        len(non_keys),
        EXPECTED_FEATURES,
    )

    direct, translated, absent = resolve_features(df.columns, canonical_features)
    audit.add(
        f"{block_name}: canonical features resolved",
        len(absent) == 0 and len(direct) + len(translated) == len(canonical_features),
        {
            "direct": len(direct),
            "translated": len(translated),
            "absent": len(absent),
        },
        {"resolved": len(canonical_features), "absent": 0},
        f"Absent examples: {absent[:10]}",
    )
    if len(canonical_features) == EXPECTED_FEATURES:
        audit.add(
            f"{block_name}: historical 213/438 resolver sanity",
            len(direct) == 213 and len(translated) == 438,
            {"direct": len(direct), "translated": len(translated)},
            {"direct": 213, "translated": 438},
            "If this warns but absent=0, inspect manifest/version before calling it an error.",
            warn=True,
        )

    bad_feature_names = scan_forbidden_feature_names(canonical_features)
    audit.add(
        f"{block_name}: no target-derived feature names",
        len(bad_feature_names) == 0,
        bad_feature_names[:20],
        [],
    )

    # Temporal storage structure
    stored_lag1 = sum(c.endswith("_lag1") for c in non_keys)
    stored_roll = sum(
        "_lag1_roll3_" in c or "_lag1_roll7_" in c for c in non_keys
    )
    audit.add(
        f"{block_name}: lag1 feature count",
        stored_lag1 == EXPECTED_TEMPORAL_BASES,
        stored_lag1,
        EXPECTED_TEMPORAL_BASES,
    )
    audit.add(
        f"{block_name}: rolling feature count",
        stored_roll == EXPECTED_ROLLING,
        stored_roll,
        EXPECTED_ROLLING,
    )
    audit.add(
        f"{block_name}: total temporal derivatives",
        stored_lag1 + stored_roll == EXPECTED_TEMPORAL_DERIVED,
        stored_lag1 + stored_roll,
        EXPECTED_TEMPORAL_DERIVED,
    )

    # Numeric integrity
    numeric = df[non_keys].select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if len(numeric.columns) else 0
    audit.add(
        f"{block_name}: infinite feature values",
        inf_count == 0,
        inf_count,
        0,
    )

    missing_cells = int(df[non_keys].isna().sum().sum())
    all_nan_cols = [c for c in non_keys if df[c].isna().all()]
    constant_cols = [c for c in non_keys if df[c].nunique(dropna=False) <= 1]

    # These are WARN diagnostics because the production Stage-1 loader zero-fills.
    audit.add(
        f"{block_name}: raw feature NaN cells",
        missing_cells == 0,
        missing_cells,
        0,
        "Raw trainable shards may legitimately contain missing values; production preprocessing must handle them consistently.",
        warn=True,
    )
    audit.add(
        f"{block_name}: all-NaN features",
        len(all_nan_cols) == 0,
        all_nan_cols[:25],
        [],
        "Investigate whether these are structurally unavailable in this block or accidentally lost.",
        warn=True,
    )
    audit.add(
        f"{block_name}: constant/all-NaN features",
        len(constant_cols) == 0,
        constant_cols[:25],
        [],
        "Constant columns are diagnostic warnings, not automatically corruption.",
        warn=True,
    )

    row.update(
        {
            "year": expected_year if expected_year is not None else (
                int(df["year"].iloc[0]) if len(df) else None
            ),
            "region": expected_region if expected_region is not None else (
                str(df["fold_region"].iloc[0]) if len(df) else None
            ),
            "cells": int(df["CanOSSEM_RASTER_CELL"].nunique()),
            "naps_ids": int(df["naps_id"].nunique()),
            "station_names": int(df["station_name"].nunique()),
            "target_min": float(pd.to_numeric(df["pm25"], errors="coerce").min()),
            "target_mean": float(pd.to_numeric(df["pm25"], errors="coerce").mean()),
            "target_max": float(pd.to_numeric(df["pm25"], errors="coerce").max()),
            "feature_cols": len(non_keys),
            "resolved_direct": len(direct),
            "resolved_translated": len(translated),
            "resolved_absent": len(absent),
            "raw_feature_nan_cells": missing_cells,
            "all_nan_feature_count": len(all_nan_cols),
            "constant_feature_count": len(constant_cols),
        }
    )

    if collect_feature_stats:
        for c in non_keys:
            s = df[c]
            rec = {
                "block": block_name,
                "stored_feature": c,
                "canonical_feature": canonical_from_stored(c),
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(s):
                ns = numeric_summary(s)
                rec.update(ns)
                rec["zero_pct"] = (
                    float(ns["zero"] / ns["finite"]) if ns["finite"] else None
                )
            feature_stats.append(rec)

    keys = set(zip(df["CanOSSEM_RASTER_CELL"].astype(str), df["date"].astype(str)))
    return row, feature_stats, keys


def infer_canonical_features_from_csv(df: pd.DataFrame) -> List[str]:
    stored = [c for c in df.columns if c not in KEY_COLS]
    return [canonical_from_stored(c) for c in stored]


def write_summary_md(path: Path, audit: Audit, summary: dict):
    lines = [
        "# Ontario PM2.5 trainable-data audit",
        "",
        f"Overall status: **{summary['overall_status']}**",
        f"- FAIL: {audit.fail_count}",
        f"- WARN: {audit.warn_count}",
        f"- PASS: {sum(c.status == 'PASS' for c in audit.checks)}",
        "",
        "## Key totals",
    ]
    for k, v in summary.get("totals", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Failures"]
    fails = [c for c in audit.checks if c.status == "FAIL"]
    if not fails:
        lines.append("- None")
    else:
        for c in fails:
            lines.append(f"- **{c.name}**: observed={c.observed!r}; expected={c.expected!r}. {c.detail}")
    lines += ["", "## Warnings"]
    warns = [c for c in audit.checks if c.status == "WARN"]
    if not warns:
        lines.append("- None")
    else:
        for c in warns:
            lines.append(f"- **{c.name}**: observed={c.observed!r}; expected={c.expected!r}. {c.detail}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard-root",
        type=Path,
        default=Path(
            r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle\Ontario_RealTarget_GPD\outputs\materialized_support_family_shards_pruned_temporal_x_repaired"
        ),
        help="Root containing manifest.json and pair_blocks/.",
    )
    ap.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Audit one CSV block instead of the parquet shard root.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_audit_output"),
        help="Output directory for log/reports.",
    )
    ap.add_argument(
        "--no-feature-stats",
        action="store_true",
        help="Skip per-feature distribution CSV for faster execution.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "data_check.log"
    logger = configure_logging(log_path, args.verbose)
    audit = Audit(logger)

    logger.info("=" * 90)
    logger.info("ONTARIO PM2.5 TRAINABLE-DATA AUDIT START")
    logger.info("=" * 90)

    block_rows: List[dict] = []
    feature_stats_all: List[dict] = []
    global_keys: set = set()
    cross_block_duplicates = 0
    all_cells: set = set()
    all_naps: set = set()
    all_station_names: set = set()
    total_rows = 0
    all_target_values: List[np.ndarray] = []
    schema_signature = None
    schema_mismatch_blocks: List[str] = []

    if args.sample_csv is not None:
        logger.info(f"Mode: standalone CSV sample")
        logger.info(f"Input: {args.sample_csv}")
        df = pd.read_csv(args.sample_csv)

        canonical_features = infer_canonical_features_from_csv(df)
        audit.add(
            "sample: inferred canonical feature count",
            len(canonical_features) == EXPECTED_FEATURES,
            len(canonical_features),
            EXPECTED_FEATURES,
        )
        year = int(df["year"].iloc[0]) if "year" in df.columns and len(df) else None
        region = str(df["fold_region"].iloc[0]) if "fold_region" in df.columns and len(df) else None
        block_name = f"PAIR_{year}_{region}" if year is not None and region is not None else args.sample_csv.stem

        row, stats, keys = audit_one_frame(
            df,
            block_name,
            canonical_features,
            audit,
            year,
            region,
            collect_feature_stats=not args.no_feature_stats,
        )
        block_rows.append(row)
        feature_stats_all.extend(stats)
        global_keys |= keys
        total_rows += len(df)
        all_cells |= set(df["CanOSSEM_RASTER_CELL"].astype(str))
        all_naps |= set(df["naps_id"].dropna().astype(str))
        all_station_names |= set(df["station_name"].dropna().astype(str))
        all_target_values.append(pd.to_numeric(df["pm25"], errors="coerce").to_numpy())

        logger.info("Standalone sample mode: global 72-block totals are not asserted.")

    else:
        root = args.shard_root
        logger.info("Mode: full parquet shard audit")
        logger.info(f"Shard root: {root}")

        audit.add("shard root exists", root.exists(), str(root), "existing path")
        if not root.exists():
            logger.error("Shard root does not exist. Exiting.")
            sys.exit(2)

        manifest = load_manifest(root)
        audit.add(
            "manifest.json exists",
            manifest is not None,
            str(root / "manifest.json"),
            "present",
        )
        if manifest is None:
            logger.error("Cannot perform canonical feature verification without manifest.json.")
            sys.exit(2)

        canonical_features = manifest.get("feature_cols", [])
        audit.add(
            "manifest canonical feature count",
            len(canonical_features) == EXPECTED_FEATURES,
            len(canonical_features),
            EXPECTED_FEATURES,
        )
        audit.add(
            "manifest feature names unique",
            len(set(canonical_features)) == len(canonical_features),
            len(set(canonical_features)),
            len(canonical_features),
        )

        bad = scan_forbidden_feature_names(canonical_features)
        audit.add("manifest target-leakage name scan", len(bad) == 0, bad, [])

        blocks = enumerate_parquet_blocks(root)
        audit.add("parquet block count", len(blocks) == EXPECTED_BLOCKS, len(blocks), EXPECTED_BLOCKS)

        expected_names = {
            f"PAIR_{y}_{r}" for y in EXPECTED_YEARS for r in EXPECTED_REGIONS
        }
        observed_names = {name for name, _ in blocks}
        missing_blocks = sorted(expected_names - observed_names)
        extra_blocks = sorted(observed_names - expected_names)
        audit.add("missing expected blocks", not missing_blocks, missing_blocks, [])
        audit.add("unexpected blocks", not extra_blocks, extra_blocks, [])

        for i, (block_name, p) in enumerate(blocks, 1):
            logger.info("-" * 90)
            logger.info(f"[{i}/{len(blocks)}] Reading {block_name}: {p}")
            year, region = parse_block_name(block_name)

            df = pd.read_parquet(p)

            # Stored schema consistency (column names + order)
            sig = tuple(df.columns)
            if schema_signature is None:
                schema_signature = sig
            elif sig != schema_signature:
                schema_mismatch_blocks.append(block_name)

            row, stats, keys = audit_one_frame(
                df,
                block_name,
                canonical_features,
                audit,
                year,
                region,
                collect_feature_stats=not args.no_feature_stats,
            )
            block_rows.append(row)
            feature_stats_all.extend(stats)

            overlap = global_keys.intersection(keys)
            cross_block_duplicates += len(overlap)
            global_keys.update(keys)

            total_rows += len(df)
            all_cells |= set(df["CanOSSEM_RASTER_CELL"].astype(str))
            all_naps |= set(df["naps_id"].dropna().astype(str))
            all_station_names |= set(df["station_name"].dropna().astype(str))
            all_target_values.append(pd.to_numeric(df["pm25"], errors="coerce").to_numpy())

        audit.add(
            "all block schemas identical",
            not schema_mismatch_blocks,
            schema_mismatch_blocks,
            [],
        )
        audit.add("total rows", total_rows == EXPECTED_ROWS, total_rows, EXPECTED_ROWS)
        audit.add(
            "global duplicate cell-days across blocks",
            cross_block_duplicates == 0,
            cross_block_duplicates,
            0,
        )
        audit.add(
            "distinct evaluated grid cells",
            len(all_cells) == EXPECTED_GRID_CELLS,
            len(all_cells),
            EXPECTED_GRID_CELLS,
        )

    # Global / sample target summary
    target = np.concatenate(all_target_values) if all_target_values else np.array([])
    finite_target = target[np.isfinite(target)]
    totals = {
        "rows": int(total_rows),
        "grid_cells": int(len(all_cells)),
        "naps_ids": int(len(all_naps)),
        "station_names": int(len(all_station_names)),
        "target_min": float(np.min(finite_target)) if finite_target.size else None,
        "target_mean": float(np.mean(finite_target)) if finite_target.size else None,
        "target_median": float(np.median(finite_target)) if finite_target.size else None,
        "target_p95": float(np.quantile(finite_target, 0.95)) if finite_target.size else None,
        "target_p99": float(np.quantile(finite_target, 0.99)) if finite_target.size else None,
        "target_max": float(np.max(finite_target)) if finite_target.size else None,
        "target_negative_count": int((finite_target < 0).sum()),
        "target_ge15_count": int((finite_target >= 15).sum()),
        "target_ge25_count": int((finite_target >= 25).sum()),
        "target_ge50_count": int((finite_target >= 50).sum()),
        "target_ge100_count": int((finite_target >= 100).sum()),
    }

    # Outputs
    pd.DataFrame(block_rows).to_csv(args.out_dir / "block_audit.csv", index=False)

    if feature_stats_all:
        pd.DataFrame(feature_stats_all).to_csv(
            args.out_dir / "feature_distribution_audit.csv", index=False
        )

    # All checks table
    pd.DataFrame([asdict(c) for c in audit.checks]).to_csv(
        args.out_dir / "checks.csv", index=False
    )

    overall = "INVALID" if audit.fail_count else (
        "VALID_WITH_WARNINGS" if audit.warn_count else "VALID"
    )
    summary = {
        "overall_status": overall,
        "fail_count": audit.fail_count,
        "warn_count": audit.warn_count,
        "pass_count": sum(c.status == "PASS" for c in audit.checks),
        "totals": totals,
        "checks": [asdict(c) for c in audit.checks],
    }
    (args.out_dir / "data_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    write_summary_md(args.out_dir / "data_audit_summary.md", audit, summary)

    logger.info("=" * 90)
    logger.info(f"OVERALL STATUS: {overall}")
    logger.info(f"PASS={summary['pass_count']} WARN={audit.warn_count} FAIL={audit.fail_count}")
    logger.info(f"TOTALS: {json.dumps(totals, default=str)}")
    logger.info(f"Outputs written to: {args.out_dir.resolve()}")
    logger.info("=" * 90)

    # Exit nonzero only on hard failures.
    return 1 if audit.fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
