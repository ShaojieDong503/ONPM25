#!/usr/bin/env bash
# Package and upload ONLY what a cloud fold run needs.
#
#   bash cloud/upload_inputs.sh
#
# Uploads ~408 MB: the zero-filled shards, the case plans and the four scripts.
# Deliberately does NOT upload outputs/ (1.5 GB of results), Data/ (the pre-zero-fill
# copy) or zero_fill_reports/ -- none of it is an input to a fold run.
set -euo pipefail

BUCKET="${BUCKET:-gs://my-project-pm25predictionon-ontario-out}"
PREFIX="${PREFIX:-post_defense}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST="$BUCKET/$PREFIX"

echo "=============================================================="
echo "[upload] source  $HERE"
echo "[upload] dest    $DEST"
echo "=============================================================="

for required in Data_zerofilled/manifest.json Data_zerofilled/pair_blocks \
                case_plans/GROUP_01.json run_thesis_folds.py \
                run_xgb_thesis_fold.py run_rf_thesis_fold.py; do
  if [[ ! -e "$HERE/$required" ]]; then
    echo "[error] missing required input: $HERE/$required" >&2
    exit 1
  fi
done

n_blocks=$(find "$HERE/Data_zerofilled/pair_blocks" -name frame.parquet | wc -l)
n_plans=$(find "$HERE/case_plans" -name 'GROUP_*.json' | wc -l)
echo "[check] pair blocks: $n_blocks   case plans: $n_plans"
if [[ "$n_blocks" -ne 72 ]]; then
  echo "[error] expected 72 pair blocks, found $n_blocks" >&2
  exit 1
fi
if [[ "$n_plans" -ne 8 ]]; then
  echo "[error] expected 8 GROUP_*.json case plans, found $n_plans" >&2
  exit 1
fi

# Scripts first: tiny, and lets a re-upload of code alone be quick.
echo
echo "### scripts"
gcloud storage cp \
  "$HERE/run_thesis_folds.py" \
  "$HERE/run_lgbm_thesis_fold.py" \
  "$HERE/run_xgb_thesis_fold.py" \
  "$HERE/run_rf_thesis_fold.py" \
  "$DEST/code/"

echo
echo "### case plans"
gcloud storage rsync -r "$HERE/case_plans" "$DEST/case_plans"

echo
echo "### zero-filled shards (~408 MB, the slow part)"
gcloud storage rsync -r "$HERE/Data_zerofilled" "$DEST/Data_zerofilled"

echo
echo "=============================================================="
echo "[done] inputs at $DEST"
gcloud storage du -s "$DEST"
echo "=============================================================="
