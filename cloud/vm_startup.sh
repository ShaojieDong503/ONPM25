#!/usr/bin/env bash
# VM startup script. Runs as root on first boot; gcloud passes it via
# --metadata-from-file startup-script=... and supplies MODEL / FOLD_FROM / FOLD_TO
# / BUCKET / PREFIX as instance metadata.
#
# Results are synced to GCS AFTER EVERY FOLD, not at the end, so a preemption or a
# crash loses at most one fold. Re-running the same VM resumes from what is already
# in the bucket.
#
# Progress is readable without SSH:
#   gcloud storage cat gs://<bucket>/<prefix>/runs/<model>/_status/progress.log
set -uo pipefail

exec > >(tee -a /var/log/pm25_run.log) 2>&1
echo "=============================================================="
echo "[startup] $(date -u +%FT%TZ)  $(hostname)"
echo "=============================================================="

meta() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || echo "$2"
}

MODEL="$(meta model lgbm)"
FOLD_FROM="$(meta fold_from 1)"
FOLD_TO="$(meta fold_to 8)"
BUCKET="$(meta bucket gs://my-project-pm25predictionon-ontario-out)"
PREFIX="$(meta prefix post_defense)"
SHUTDOWN="$(meta shutdown_when_done 1)"
THREADS="$(meta threads 0)"

SRC="$BUCKET/$PREFIX"
DST="$SRC/runs/$MODEL"
WORK=/opt/pm25
STATUS_LOCAL="$WORK/outputs/_status"

echo "[cfg] model=$MODEL folds=$FOLD_FROM-$FOLD_TO"
echo "[cfg] inputs=$SRC"
echo "[cfg] results=$DST"

mkdir -p "$WORK" "$STATUS_LOCAL"
cd "$WORK"

note() {   # one-line progress breadcrumb, visible from GCS without SSH
  echo "$(date -u +%FT%TZ) | $*" | tee -a "$STATUS_LOCAL/progress.log"
  gcloud storage cp "$STATUS_LOCAL/progress.log" "$DST/_status/progress.log" -q 2>/dev/null || true
}

fail() {
  note "FAILED: $*"
  gcloud storage cp /var/log/pm25_run.log "$DST/_status/pm25_run.log" -q 2>/dev/null || true
  [[ "$SHUTDOWN" == "1" ]] && shutdown -h +5
  exit 1
}

# ---------------------------------------------------------------- 1. environment
note "installing python environment"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip libgomp1 >/dev/null || fail "apt-get"

python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip

# Versions pinned to the ones the thesis runs used. Changing them changes binning and
# tie-breaking in the tree learners, i.e. changes the model.
pip install --quiet \
  "numpy" "pandas" "pyarrow" \
  "scikit-learn==1.9.0" "lightgbm==4.6.0" "xgboost==3.2.0" || fail "pip install"

python3 - <<'PY'
import importlib
for m in ("numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "xgboost"):
    print(f"  {m:<12} {importlib.import_module(m).__version__}")
PY

# ---------------------------------------------------------------- 2. inputs
note "downloading inputs"
gcloud storage rsync -r "$SRC/Data_zerofilled" "$WORK/Data_zerofilled" || fail "rsync shards"
gcloud storage rsync -r "$SRC/case_plans"      "$WORK/case_plans"      || fail "rsync case_plans"
gcloud storage rsync -r "$SRC/code"            "$WORK/code"            || fail "rsync code"

n=$(find "$WORK/Data_zerofilled/pair_blocks" -name frame.parquet | wc -l)
[[ "$n" -eq 72 ]] || fail "expected 72 pair blocks, got $n"
note "inputs ready ($n blocks)"

# Resume: pull back any folds an earlier attempt already finished.
gcloud storage rsync -r "$DST" "$WORK/outputs" 2>/dev/null || true

# ---------------------------------------------------------------- 3. threads
if [[ "$THREADS" != "0" ]]; then
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
         OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_MAX_THREADS="$THREADS"
  note "threads pinned to $THREADS"
fi

# ---------------------------------------------------------------- 4. folds
FAILURES=0
for (( fold=FOLD_FROM; fold<=FOLD_TO; fold++ )); do
  if [[ -f "$WORK/outputs/GROUP_$(printf '%02d' "$fold")/metrics.json" ]]; then
    note "fold $fold already complete; skipping"
    continue
  fi

  note "fold $fold START"
  t0=$(date +%s)
  set +e
  python3 "$WORK/code/run_thesis_folds.py" \
    --model "$MODEL" --folds "$fold" \
    --shard-root     "$WORK/Data_zerofilled" \
    --case-plans-dir "$WORK/case_plans" \
    --out-root       "$WORK/outputs"
  code=$?
  set -e
  elapsed=$(( $(date +%s) - t0 ))

  # Sync after every fold so a preemption costs at most this one.
  gcloud storage rsync -r "$WORK/outputs" "$DST" || note "WARNING: sync failed for fold $fold"

  if [[ "$code" -eq 0 ]]; then
    note "fold $fold DONE in ${elapsed}s"
  else
    FAILURES=$(( FAILURES + 1 ))
    note "fold $fold FAILED exit=$code after ${elapsed}s"
  fi
done

# ---------------------------------------------------------------- 5. finish
gcloud storage cp /var/log/pm25_run.log "$DST/_status/pm25_run.log" -q || true
note "ALL DONE model=$MODEL folds=$FOLD_FROM-$FOLD_TO failures=$FAILURES"
echo "DONE failures=$FAILURES" > "$STATUS_LOCAL/COMPLETE"
gcloud storage cp "$STATUS_LOCAL/COMPLETE" "$DST/_status/COMPLETE" -q || true

if [[ "$SHUTDOWN" == "1" ]]; then
  note "shutting down in 3 minutes (set shutdown_when_done=0 to keep the VM alive)"
  shutdown -h +3
fi
