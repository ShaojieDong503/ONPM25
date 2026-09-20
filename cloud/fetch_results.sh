#!/usr/bin/env bash
# Download a model's fold results from GCS.
#
#   bash cloud/fetch_results.sh xgb            # metrics + predictions only (small)
#   MODELS=1 bash cloud/fetch_results.sh rf    # include the .pkl/.txt model files (large)
#
# By default the fitted-model files are EXCLUDED. For RF at the thesis configuration
# they are several GB per fold; you rarely need them locally to write up results.
set -euo pipefail

MODEL="${1:-}"
case "$MODEL" in
  lgbm|xgb|rf) ;;
  *) echo "Usage: bash cloud/fetch_results.sh <lgbm|xgb|rf>" >&2; exit 2 ;;
esac

BUCKET="${BUCKET:-gs://my-project-pm25predictionon-ontario-out}"
PREFIX="${PREFIX:-post_defense}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="$BUCKET/$PREFIX/runs/$MODEL"
DST="$HERE/outputs/${MODEL}_thesis_zerofilled_cloud"

echo "[fetch] $SRC  ->  $DST"
mkdir -p "$DST"

if [[ "${MODELS:-0}" == "1" ]]; then
  gcloud storage rsync -r "$SRC" "$DST"
else
  gcloud storage rsync -r --exclude='.*\.(pkl|txt)$' "$SRC" "$DST"
  echo "[note] fitted-model .pkl/.txt files skipped; re-run with MODELS=1 to include them"
fi

echo
echo "[status]"
gcloud storage cat "$SRC/_status/progress.log" 2>/dev/null | tail -20 || echo "  (no progress log yet)"

echo
echo "[folds present locally]"
for g in 01 02 03 04 05 06 07 08; do
  f="$DST/GROUP_$g/metrics.json"
  if [[ -f "$f" ]]; then
    python -c "
import json,sys
m=json.load(open(r'$f'))
s=m['stage1_holdout']; fi=m['final_holdout']
print(f\"  GROUP_$g  n={m['holdout_rows']:>6}  stage1 rmse={s['rmse']:.4f} r2={s['r2_predictive']:.4f}   final rmse={fi['rmse']:.4f} r2={fi['r2_predictive']:.4f}\")
" 2>/dev/null || echo "  GROUP_$g  (metrics.json unreadable)"
  else
    echo "  GROUP_$g  -- missing --"
  fi
done
