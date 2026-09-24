#!/usr/bin/env bash
# =============================================================================
# Package this directory and ship it to the GCloud VM.
#
#   ./vm_deploy.sh inventory      # what is on the VM now, and what it costs
#   ./vm_deploy.sh clean          # remove the old deployment  (asks first)
#   ./vm_deploy.sh push           # upload code + data
#   ./vm_deploy.sh push-code      # upload code only (fast, for a fix-up)
#   ./vm_deploy.sh setup          # create the venv and install requirements
#   ./vm_deploy.sh run            # start the pipeline under tmux, detached
#   ./vm_deploy.sh tail           # follow the running pipeline's log
#
# Run from Git Bash on Windows or any Linux shell. Requires gcloud, or set
# USE_SSH=1 to fall back to plain ssh/scp against the external IP.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

VM_NAME=${VM_NAME:-instance-20260923-184043}
VM_ZONE=${VM_ZONE:-us-central1-a}
VM_IP=${VM_IP:-34.30.154.225}
VM_USER=${VM_USER:-$(whoami)}
USE_SSH=${USE_SSH:-0}

# The 10 GB boot disk cannot hold this pipeline -- Random Forest alone writes
# ~6.4 GB per fold. Everything lives on the 200 GB data disk (/dev/sdc), and the
# raster inputs stay on the 60 GB disk (/dev/sdb) where they already are. Both
# are in /etc/fstab, so a reboot remounts them.
REMOTE=${REMOTE:-/mnt/work/pm25}          # deployment root, 200 GB disk
GRID=${GRID:-/mnt/grid/grid}              # 144 monthly rasters, 2012-01..2023-12

# Data roots to ship. Everything else in this directory is code or output.
DATA_DIRS=(Data Data_by_cell Data_on_plus_qc)
PLAN_DIRS=(case_plans case_plans_spatialcv case_plans_external_qc
           case_plans_foldalt_seed101 case_plans_foldalt_seed202
           case_plans_foldalt_seed303 case_plans_foldalt_seed404
           features_ablation features_policy)

if [[ "$USE_SSH" == "1" || -z "$VM_NAME" ]]; then
  rsh() { ssh "${VM_USER}@${VM_IP}" "$@"; }
  put() { scp -C "$1" "${VM_USER}@${VM_IP}:$2"; }
else
  rsh() { gcloud compute ssh "$VM_NAME" --zone "$VM_ZONE" --command "$*"; }
  put() { gcloud compute scp --zone "$VM_ZONE" --compress "$1" "$VM_NAME:$2"; }
fi

case "${1:-}" in

inventory)
  # Look before deleting. Shows what a clean would remove and what it buys back.
  rsh "echo '--- disk ---'; df -h / | tail -1;
       echo; echo '--- home, one level ---'; du -sh ~/* 2>/dev/null | sort -rh | head -20;
       echo; echo '--- deployment root ---';
       if [ -d ${REMOTE} ]; then du -sh ${REMOTE}/* 2>/dev/null | sort -rh | head -30;
       else echo '${REMOTE} does not exist yet'; fi;
       echo; echo '--- anything still running? ---';
       pgrep -af 'python|run_everything' || echo '  (nothing)'"
  ;;

clean)
  # Destructive and irreversible. It prints the target and waits for a typed
  # confirmation -- a previous run of a build script in this project deleted
  # results that were nested under its output root, so nothing here deletes on
  # the strength of a flag alone.
  echo "About to REMOVE ${REMOTE} on ${VM_NAME:-$VM_IP}, including any results in it."
  rsh "du -sh ${REMOTE} 2>/dev/null || echo '  (${REMOTE} not present)'"
  echo
  read -r -p "Type the word DELETE to proceed: " confirm
  [[ "$confirm" == "DELETE" ]] || { echo "aborted, nothing changed"; exit 1; }
  rsh "pgrep -f run_everything >/dev/null && { echo 'REFUSING: the pipeline is still running'; exit 1; };
       rm -rf ${REMOTE} && echo 'removed ${REMOTE}'; df -h / | tail -1"
  ;;

push|push-code)
  echo "[1/3] packaging code"
  tar czf /tmp/pm25_code.tgz \
      --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='Data' --exclude='Data_by_cell' --exclude='Data_on_plus_qc' \
      *.py *.sh *.md requirements.txt 2>/dev/null
  ls -lh /tmp/pm25_code.tgz

  echo "[2/3] uploading code"
  rsh "mkdir -p ${REMOTE}"
  put /tmp/pm25_code.tgz "${REMOTE}/"
  rsh "cd ${REMOTE} && tar xzf pm25_code.tgz && rm pm25_code.tgz && chmod +x *.sh && ls *.py | wc -l"

  if [[ "$1" == "push-code" ]]; then echo "code only, done"; exit 0; fi

  echo "[3/3] uploading data and plans  (~1.6 GB, this is the slow part)"
  for d in "${DATA_DIRS[@]}" "${PLAN_DIRS[@]}"; do
    [[ -d "$d" ]] || { echo "  skip $d (not present locally)"; continue; }
    echo "  $d  ($(du -sh "$d" | cut -f1))"
    tar czf "/tmp/pm25_$d.tgz" "$d"
    put "/tmp/pm25_$d.tgz" "${REMOTE}/"
    rsh "cd ${REMOTE} && tar xzf pm25_$d.tgz && rm pm25_$d.tgz"
    rm -f "/tmp/pm25_$d.tgz"
  done
  rsh "cd ${REMOTE} && du -sh . && ls -d Data* case_plans* features_* 2>/dev/null"
  ;;

setup)
  rsh "cd ${REMOTE} &&
       sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip tmux libgomp1 &&
       python3 -m venv .venv &&
       .venv/bin/pip install --quiet --upgrade pip &&
       .venv/bin/pip install --quiet -r requirements.txt &&
       .venv/bin/python -c 'import lightgbm,xgboost,sklearn,pandas,pyarrow,shap;
print(\"lightgbm\", lightgbm.__version__); print(\"xgboost\", xgboost.__version__);
print(\"sklearn\", sklearn.__version__); print(\"pandas\", pandas.__version__)' &&
       nproc && free -g | head -2"
  ;;

run)
  # tmux, so the pipeline survives the ssh session closing. A 12-step run that
  # dies because a laptop slept is a bad way to spend a day of compute.
  shift || true
  rsh "cd ${REMOTE} && tmux kill-session -t pm25 2>/dev/null;
       tmux new-session -d -s pm25 \
         'source .venv/bin/activate && ./run_everything.sh ${*:-} 2>&1 | tee -a pipeline.log' &&
       sleep 3 && tail -20 ${REMOTE}/pipeline.log"
  echo
  echo "detached. follow it with:  ./vm_deploy.sh tail"
  ;;

tail)
  rsh "tail -f ${REMOTE}/pipeline.log"
  ;;

*)
  sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
  ;;
esac
