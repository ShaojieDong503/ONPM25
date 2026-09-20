#!/usr/bin/env python3
"""
Zero-fill residual NaNs in Ontario PM2.5 trainable feature blocks and log every fill.

Scope
-----
This script starts from the already-built trainable data tables/shards.
It fills NaN values with 0 ONLY in model feature columns.

It does NOT fill:
- pm25 target
- date/year/region/cell/station metadata
- other key columns

It records:
- block name
- total rows
- columns requiring filling
- NaN count per column before fill
- row count affected per column
- total cells filled
- total rows affected in each block
- verification that no feature NaNs remain afterward

By default it writes a mirrored COPY to a separate output root.
Use --in-place only if you intentionally want to overwrite the original parquet files.

Expected input layout:
  <shard_root>/
    manifest.json
    pair_blocks/
      PAIR_<year>_<region>/
        frame.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


KEY_COLUMNS = {
    "date",
    "year",
    "fold_region",
    "CanOSSEM_RASTER_CELL",
    "pm25",
    "_year_region_pair",
    "naps_id",
    "station_name",
}


def configure_logger(log_path: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("pm25_zero_fill")
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


def read_feature_columns(shard_root: Path, df: pd.DataFrame) -> List[str]:
    """
    Prefer manifest.json["feature_cols"] if present.
    Otherwise define features as all columns except known keys.
    """
    manifest_path = shard_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cols = manifest.get("feature_cols")
        if cols:
            # Canonical manifest names may differ from stored parquet names for rolling features.
            resolved = []
            for c in cols:
                if c in df.columns:
                    resolved.append(c)
                    continue
                stored = c.replace("_roll3_", "_lag1_roll3_").replace(
                    "_roll7_", "_lag1_roll7_"
                )
                if stored in df.columns:
                    resolved.append(stored)
                    continue
                raise KeyError(
                    f"Feature from manifest cannot be resolved in dataframe: {c}"
                )
            return resolved

    return [c for c in df.columns if c not in KEY_COLUMNS]


def zero_fill_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    block_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict, List[dict]]:
    """
    Fill feature NaNs with 0 and return:
      modified df,
      block summary,
      per-column fill records.
    """
    missing_feature_cols = [c for c in feature_cols if c not in df.columns]
    if missing_feature_cols:
        raise KeyError(
            f"{block_name}: missing expected feature columns: "
            f"{missing_feature_cols[:20]}"
        )

    # Never touch target or metadata.
    forbidden = sorted(set(feature_cols) & KEY_COLUMNS)
    if forbidden:
        raise ValueError(
            f"{block_name}: feature list unexpectedly includes key/target columns: {forbidden}"
        )

    feature_df = df[feature_cols]

    # Count NaN only, not +/-inf.
    nan_mask = feature_df.isna()
    nan_counts = nan_mask.sum()
    cols_to_fill = nan_counts[nan_counts > 0].sort_values(ascending=False)

    rows_affected = int(nan_mask.any(axis=1).sum())
    cells_filled = int(nan_mask.to_numpy().sum())

    column_records = []
    for col, count in cols_to_fill.items():
        affected_rows = int(nan_mask[col].sum())
        column_records.append(
            {
                "block": block_name,
                "column": col,
                "nan_cells_filled": int(count),
                "rows_affected": affected_rows,
                "row_fraction_affected": float(affected_rows / len(df)) if len(df) else 0.0,
                "dtype_before": str(df[col].dtype),
            }
        )

    logger.info("-" * 100)
    logger.info(
        "%s | rows=%d | features=%d | columns_needing_fill=%d | "
        "rows_affected=%d | cells_filled=%d",
        block_name,
        len(df),
        len(feature_cols),
        len(cols_to_fill),
        rows_affected,
        cells_filled,
    )

    for rec in column_records:
        logger.info(
            "FILL | block=%s | column=%s | nan_cells=%d | rows=%d | row_fraction=%.6f",
            rec["block"],
            rec["column"],
            rec["nan_cells_filled"],
            rec["rows_affected"],
            rec["row_fraction_affected"],
        )

    # Copy to avoid modifying caller's frame unexpectedly.
    out = df.copy()

    # Fill only feature columns. pandas keeps numeric columns numeric.
    if len(cols_to_fill):
        out.loc[:, feature_cols] = out[feature_cols].fillna(0)

    remaining = int(out[feature_cols].isna().to_numpy().sum())
    if remaining != 0:
        raise RuntimeError(
            f"{block_name}: zero fill incomplete; {remaining} feature NaNs remain."
        )

    # Confirm target was untouched, including NaN pattern and values.
    if "pm25" in df.columns:
        before = df["pm25"]
        after = out["pm25"]
        same_target = before.equals(after)
        if not same_target:
            raise RuntimeError(f"{block_name}: pm25 target changed during zero-fill.")

    summary = {
        "block": block_name,
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "columns_needing_fill": int(len(cols_to_fill)),
        "rows_affected": rows_affected,
        "row_fraction_affected": float(rows_affected / len(df)) if len(df) else 0.0,
        "cells_filled": cells_filled,
        "remaining_feature_nan_cells": remaining,
    }

    return out, summary, column_records


def copy_support_files(shard_root: Path, output_root: Path, logger: logging.Logger):
    """
    Copy top-level metadata files needed to preserve the shard bundle structure.
    Does not copy original pair block parquet files; those are rewritten after filling.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for name in [
        "manifest.json",
        "pair_blocks_summary.csv",
        "case_plans_summary.csv",
        "support_blocks_summary.csv",
        "kept_base_feature_selection.txt",
        "temporal_feature_selection.txt",
    ]:
        src = shard_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)
            logger.info("Copied metadata: %s", name)

    # Copy case plans if present.
    src_case_plans = shard_root / "case_plans"
    dst_case_plans = output_root / "case_plans"
    if src_case_plans.exists() and not dst_case_plans.exists():
        shutil.copytree(src_case_plans, dst_case_plans)
        logger.info("Copied case_plans/")


def process_parquet_root(
    shard_root: Path,
    output_root: Path | None,
    in_place: bool,
    logger: logging.Logger,
) -> tuple[List[dict], List[dict]]:
    pair_root = shard_root / "pair_blocks"
    if not pair_root.exists():
        raise FileNotFoundError(f"pair_blocks directory not found: {pair_root}")

    blocks = sorted(pair_root.glob("PAIR_*/frame.parquet"))
    if not blocks:
        raise FileNotFoundError(f"No PAIR_*/frame.parquet files found under {pair_root}")

    if not in_place:
        assert output_root is not None
        copy_support_files(shard_root, output_root, logger)

    block_summaries: List[dict] = []
    column_records: List[dict] = []

    for idx, src in enumerate(blocks, 1):
        block_name = src.parent.name
        logger.info("[%d/%d] Reading %s", idx, len(blocks), src)

        df = pd.read_parquet(src)
        feature_cols = read_feature_columns(shard_root, df)

        filled, summary, records = zero_fill_frame(
            df=df,
            feature_cols=feature_cols,
            block_name=block_name,
            logger=logger,
        )

        if in_place:
            dst = src
            # Write temporary file first, then replace atomically.
            tmp = src.with_name("frame.zero_fill_tmp.parquet")
            filled.to_parquet(tmp, index=False)
            tmp.replace(src)
        else:
            dst = output_root / "pair_blocks" / block_name / "frame.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            filled.to_parquet(dst, index=False)

            # Preserve per-block meta.json if present.
            meta_src = src.parent / "meta.json"
            if meta_src.exists():
                shutil.copy2(meta_src, dst.parent / "meta.json")

        logger.info("WROTE | %s", dst)

        # Read back and verify persisted output.
        verify = pd.read_parquet(dst)
        persisted_remaining = int(verify[feature_cols].isna().to_numpy().sum())
        if persisted_remaining != 0:
            raise RuntimeError(
                f"{block_name}: persisted output still has "
                f"{persisted_remaining} feature NaN cells."
            )

        summary["output_path"] = str(dst)
        block_summaries.append(summary)
        column_records.extend(records)

    return block_summaries, column_records


def process_single_csv(
    csv_path: Path,
    output_csv: Path,
    logger: logging.Logger,
) -> tuple[List[dict], List[dict]]:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in KEY_COLUMNS]
    block_name = csv_path.stem

    filled, summary, records = zero_fill_frame(
        df=df,
        feature_cols=feature_cols,
        block_name=block_name,
        logger=logger,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(output_csv, index=False)
    logger.info("WROTE | %s", output_csv)

    verify = pd.read_csv(output_csv)
    persisted_remaining = int(verify[feature_cols].isna().to_numpy().sum())
    if persisted_remaining != 0:
        raise RuntimeError(
            f"{block_name}: persisted CSV still has {persisted_remaining} feature NaNs."
        )

    summary["output_path"] = str(output_csv)
    return [summary], records


def write_reports(
    report_dir: Path,
    block_summaries: List[dict],
    column_records: List[dict],
    logger: logging.Logger,
):
    report_dir.mkdir(parents=True, exist_ok=True)

    blocks_df = pd.DataFrame(block_summaries)
    cols_df = pd.DataFrame(column_records)

    blocks_df.to_csv(report_dir / "zero_fill_block_summary.csv", index=False)
    cols_df.to_csv(report_dir / "zero_fill_column_log.csv", index=False)

    total_rows = int(blocks_df["rows"].sum()) if len(blocks_df) else 0
    total_cells = int(blocks_df["cells_filled"].sum()) if len(blocks_df) else 0
    total_rows_affected = int(blocks_df["rows_affected"].sum()) if len(blocks_df) else 0
    blocks_affected = int((blocks_df["cells_filled"] > 0).sum()) if len(blocks_df) else 0

    if len(cols_df):
        aggregate = (
            cols_df.groupby("column", as_index=False)
            .agg(
                blocks_affected=("block", "nunique"),
                nan_cells_filled=("nan_cells_filled", "sum"),
                rows_affected_sum=("rows_affected", "sum"),
            )
            .sort_values(["nan_cells_filled", "column"], ascending=[False, True])
        )
    else:
        aggregate = pd.DataFrame(
            columns=["column", "blocks_affected", "nan_cells_filled", "rows_affected_sum"]
        )

    aggregate.to_csv(report_dir / "zero_fill_column_aggregate.csv", index=False)

    summary = {
        "status": "SUCCESS",
        "blocks_processed": int(len(block_summaries)),
        "blocks_with_any_fill": blocks_affected,
        "rows_processed": total_rows,
        "block_row_occurrences_affected": total_rows_affected,
        "feature_nan_cells_filled": total_cells,
        "remaining_feature_nan_cells": int(
            blocks_df["remaining_feature_nan_cells"].sum()
        ) if len(blocks_df) else 0,
    }

    (report_dir / "zero_fill_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# PM2.5 zero-fill summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Blocks processed: {summary['blocks_processed']}",
        f"- Blocks with any fill: {summary['blocks_with_any_fill']}",
        f"- Rows processed: {summary['rows_processed']}",
        f"- Block-row occurrences affected: {summary['block_row_occurrences_affected']}",
        f"- Feature NaN cells filled with zero: {summary['feature_nan_cells_filled']}",
        f"- Remaining feature NaN cells: {summary['remaining_feature_nan_cells']}",
        "",
        "Target and metadata columns were not modified.",
        "",
        "See:",
        "- `zero_fill_block_summary.csv`",
        "- `zero_fill_column_log.csv`",
        "- `zero_fill_column_aggregate.csv`",
        "- `zero_fill.log`",
    ]
    (report_dir / "zero_fill_summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    logger.info("=" * 100)
    logger.info("ZERO-FILL COMPLETE")
    logger.info("Blocks processed: %d", summary["blocks_processed"])
    logger.info("Blocks with fills: %d", summary["blocks_with_any_fill"])
    logger.info("Feature NaN cells filled: %d", summary["feature_nan_cells_filled"])
    logger.info("Remaining feature NaNs: %d", summary["remaining_feature_nan_cells"])
    logger.info("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Zero-fill NaNs in PM2.5 trainable feature shards with detailed logging."
    )
    ap.add_argument(
        "--shard-root",
        type=Path,
        default=Path(
            r"D:\lambda\Ontario_RealTarget_GPD\dist\gcloud_repaired_bundle\Ontario_RealTarget_GPD\outputs\materialized_support_family_shards_pruned_temporal_x_repaired"
        ),
        help="Input shard root containing manifest.json and pair_blocks/.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output shard root. If omitted, defaults to sibling directory "
            "'materialized_support_family_shards_pruned_temporal_x_repaired_zerofilled'."
        ),
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original frame.parquet files. Use deliberately.",
    )
    ap.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Optional: process one CSV instead of the parquet shard root.",
    )
    ap.add_argument(
        "--sample-output-csv",
        type=Path,
        default=None,
        help="Output path when --sample-csv is used.",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=Path("zero_fill_reports"),
        help="Directory for logs and audit reports.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.report_dir / "zero_fill.log", args.verbose)

    logger.info("=" * 100)
    logger.info("ONTARIO PM2.5 ZERO-FILL START")
    logger.info("=" * 100)

    try:
        if args.sample_csv is not None:
            output_csv = args.sample_output_csv
            if output_csv is None:
                output_csv = args.sample_csv.with_name(
                    args.sample_csv.stem + "_zerofilled.csv"
                )

            logger.info("Mode: single CSV")
            logger.info("Input CSV: %s", args.sample_csv)
            logger.info("Output CSV: %s", output_csv)

            block_summaries, column_records = process_single_csv(
                args.sample_csv,
                output_csv,
                logger,
            )
        else:
            shard_root = args.shard_root

            if args.in_place:
                output_root = None
                logger.warning("MODE: IN-PLACE OVERWRITE")
                logger.warning("Original frame.parquet files WILL be replaced.")
            else:
                output_root = args.output_root
                if output_root is None:
                    output_root = shard_root.with_name(
                        shard_root.name + "_zerofilled"
                    )
                if output_root.resolve() == shard_root.resolve():
                    raise ValueError(
                        "output-root equals shard-root. Use --in-place explicitly if intended."
                    )
                logger.info("Mode: copy")
                logger.info("Input shard root: %s", shard_root)
                logger.info("Output shard root: %s", output_root)

            block_summaries, column_records = process_parquet_root(
                shard_root=shard_root,
                output_root=output_root,
                in_place=args.in_place,
                logger=logger,
            )

        write_reports(
            report_dir=args.report_dir,
            block_summaries=block_summaries,
            column_records=column_records,
            logger=logger,
        )

        return 0

    except Exception:
        logger.exception("ZERO-FILL FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
