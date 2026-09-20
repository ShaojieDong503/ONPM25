#!/usr/bin/env bash
# Create a VM that runs a range of folds and shuts itself down.
#
#   bash cloud/launch_vm.sh xgb 1 8
#   bash cloud/launch_vm.sh rf  1 2
#   SPOT=1 bash cloud/launch_vm.sh xgb 1 8        # ~70% cheaper, can be preempted
#   KEEP_ALIVE=1 bash cloud/launch_vm.sh rf 1 1   # don't auto-shutdown (for debugging)
#
# Machine sizes differ sharply between the two learners -- see cloud/README.md.
set -euo pipefail

MODEL="${1:-}"
FOLD_FROM="${2:-1}"
FOLD_TO="${3:-8}"

case "$MODEL" in
  lgbm|xgb|rf) ;;
  *) echo "Usage: bash cloud/launch_vm.sh <lgbm|xgb|rf> [from] [to]" >&2; exit 2 ;;
esac

BUCKET="${BUCKET:-gs://my-project-pm25predictionon-ontario-out}"
PREFIX="${PREFIX:-post_defense}"
ZONE="${ZONE:-us-central1-c}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sizing. XGB is CPU-bound with a modest footprint. RF at the thesis configuration
# (900 trees, max_depth=None, min_samples_leaf=2) grows ~150k nodes per tree on
# 148,785 rows, so the forest alone is several GB in RAM and again on disk.
case "$MODEL" in
  lgbm) MACHINE="${MACHINE:-n2-standard-16}"; DISK="${DISK:-200}"; THREADS="${THREADS:-0}" ;;
  xgb)  MACHINE="${MACHINE:-n2-standard-16}"; DISK="${DISK:-200}"; THREADS="${THREADS:-0}" ;;
  rf)   MACHINE="${MACHINE:-n2-highmem-16}"; DISK="${DISK:-800}"; THREADS="${THREADS:-0}" ;;
esac

SHUTDOWN=1
[[ "${KEEP_ALIVE:-0}" == "1" ]] && SHUTDOWN=0

NAME="${NAME:-pm25-$MODEL-$(date +%m%d-%H%M)}"

PROVISION=()
if [[ "${SPOT:-0}" == "1" ]]; then
  # Spot capacity draws on the PREEMPTIBLE_CPUS quota, which is 0 on a fresh project.
  # Check before creating, or the failure arrives as an opaque quota error.
  limit=$(gcloud compute regions describe "${ZONE%-*}" --format=json 2>/dev/null \
          | python -c "import json,sys;print(next((int(q['limit']) for q in json.load(sys.stdin).get('quotas',[]) if q['metric']=='PREEMPTIBLE_CPUS'),0))" 2>/dev/null || echo 0)
  if [[ "$limit" -lt 1 ]]; then
    echo "[error] SPOT=1 requested but PREEMPTIBLE_CPUS quota in ${ZONE%-*} is $limit." >&2
    echo "        Request an increase at IAM & Admin > Quotas, or drop SPOT=1 to use" >&2
    echo "        a standard on-demand VM (roughly 3x the cost, no preemption risk)." >&2
    exit 1
  fi
  PROVISION=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

echo "=============================================================="
echo "[launch] name      $NAME"
echo "[launch] model     $MODEL   folds $FOLD_FROM-$FOLD_TO"
echo "[launch] machine   $MACHINE   disk ${DISK}GB   zone $ZONE"
echo "[launch] spot      ${SPOT:-0}    auto-shutdown $SHUTDOWN"
echo "[launch] results   $BUCKET/$PREFIX/runs/$MODEL"
echo "=============================================================="

gcloud compute instances create "$NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size="${DISK}GB" \
  --boot-disk-type=pd-balanced \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script="$HERE/vm_startup.sh" \
  --metadata="model=$MODEL,fold_from=$FOLD_FROM,fold_to=$FOLD_TO,bucket=$BUCKET,prefix=$PREFIX,shutdown_when_done=$SHUTDOWN,threads=$THREADS" \
  "${PROVISION[@]}"

cat <<EOF

Watch progress without SSH:
  gcloud storage cat $BUCKET/$PREFIX/runs/$MODEL/_status/progress.log

Or attach to the VM's log:
  gcloud compute ssh $NAME --zone=$ZONE --command="sudo tail -f /var/log/pm25_run.log"

When it finishes it writes _status/COMPLETE and shuts down (billing for CPU stops;
the disk still bills until deleted):
  gcloud compute instances delete $NAME --zone=$ZONE

Fetch results:
  bash cloud/fetch_results.sh $MODEL
EOF
