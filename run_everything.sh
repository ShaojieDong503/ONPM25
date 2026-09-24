#!/usr/bin/env bash
# =============================================================================
# Post-defence pipeline for the e2-standard-32 VM (32 vCPU / 128 GB, Broadwell).
#
#   ./run_everything.sh            # all steps (0-12)
#   ./run_everything.sh 5          # start at step 5
#   ./run_everything.sh 5 8        # steps 5 through 8 only
#
# Linux counterpart of run_everything.ps1, with the folds run concurrently.
# One command per step, in dependency order; read it top to bottom and you have
# the whole pipeline.
#
# THE ONE THING THAT MATTERS FOR CONCURRENCY
#   The fold scripts default to n_jobs=-1, and LightGBM's n_jobs overrides
#   OMP_NUM_THREADS -- exporting that variable does nothing. Four folds left at
#   the default would each claim all 32 cores, and at 4x oversubscription the
#   OpenMP barriers spin instead of working: a 9-minute fit can take hours.
#   Every launch below therefore pairs --jobs with an explicit --threads, and
#   --jobs x --threads never exceeds the vCPU count. run_thesis_folds.py refuses
#   --jobs > 1 without --threads, so this cannot be forgotten by accident.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

FROM=${1:-0}
TO=${2:-12}
STARTED=$(date +%s)

VCPU=$(nproc)
# LightGBM and XGBoost stop scaling around 8 threads (measured 3.26x at 8, 3.30x
# at 12), so more thinner jobs beat fewer fat ones.
JOBS=4;    THREADS=8
# Random Forest scales better per process (5.00x at 8, 5.91x at 12) but that is
# still sublinear, so for total throughput over 8 folds more processes beat wider
# ones: 4x8 aggregates ~20x against ~14x for 2x16. Each fit holds roughly 6 GB
# (a 900-tree, unlimited-depth forest measured 73.6M nodes, plus the training
# matrix), so 4 concurrent is ~24 GB against 125 GB of RAM.
RF_JOBS=4; RF_THREADS=8

# Flags applied to EVERY fold run. Must match what the LightGBM primary folds
# used, or compare_models.py is not comparing like with like.
COMMON=(--shard-root Data --case-plans-dir case_plans)
# COMMON=(--shard-root Data --case-plans-dir case_plans \
#         --fill-policy tiered --features-file features_policy/aod_filled_only.json)

# Reads commands on stdin, one per line, and runs $1 of them at a time.
# Exits non-zero if any command fails, which set -e turns into a halt.
pool() { xargs -P "$1" -d '\n' -I CMD bash -c CMD; }

step() {
  local n=$1 text=$2
  if (( n < FROM || n > TO )); then echo "[$n] skip   $text"; return 1; fi
  echo; printf '=%.0s' {1..78}; echo
  echo "[$n] $text   $(date +%H:%M:%S)"
  printf '=%.0s' {1..78}; echo
  return 0
}

# The 144 monthly rasters for step 12. They live on the 60 GB data disk, not
# beside the code, and predict_raster_cells.py's default points at a Windows path.
GRID_DIR=${GRID_DIR:-/mnt/grid/grid}

echo "host $(hostname)   ${VCPU} vCPU   $(free -g | awk '/^Mem:/{print $2}') GB RAM"
echo "folds ${JOBS}x${THREADS}   rf ${RF_JOBS}x${RF_THREADS}   steps ${FROM}-${TO}"
echo "disk  $(df -h . | awk 'NR==2{print $4" free on "$6}')"
(( JOBS * THREADS <= VCPU )) || echo "WARNING: ${JOBS}x${THREADS} oversubscribes ${VCPU} vCPU"

# Fail now, not eleven steps from now. Measured on the VM: Random Forest writes
# ~13 GB per fold (104 GB for the family), everything else together ~27 GB. So
# the requirement depends entirely on whether step 2 is still ahead of us.
avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if (( FROM <= 2 )); then need_gb=135; else need_gb=30; fi
(( avail_gb >= need_gb )) || {
  echo "ERROR: ${avail_gb} GB free, steps ${FROM}-${TO} need ~${need_gb} GB"; exit 1; }
if (( TO >= 12 )) && [[ ! -d "$GRID_DIR/temporal_tables" ]]; then
  echo "ERROR: step 12 needs the rasters, but $GRID_DIR/temporal_tables is missing."
  echo "       Set GRID_DIR, or run ./run_everything.sh $FROM 11 to stop before it."
  exit 1
fi

# -----------------------------------------------------------------------------
if step 0 "LightGBM, 8 folds  (no-op if outputs/lgbm_thesis was uploaded)"; then
  # The primary LightGBM folds were run on the workstation and copied here, so
  # --resume normally skips all 8 immediately. The step exists so this script is
  # self-sufficient: step 3 compares three families and step 10 needs LightGBM's
  # models, neither of which works if the family is simply absent.
  python run_thesis_folds.py --model lgbm --from 1 --to 8 "${COMMON[@]}" \
    --jobs $JOBS --threads $THREADS --resume
fi

# -----------------------------------------------------------------------------
if step 1 "XGBoost, 8 folds"; then
  python run_thesis_folds.py --model xgb --from 1 --to 8 "${COMMON[@]}" \
    --jobs $JOBS --threads $THREADS --resume
fi

# -----------------------------------------------------------------------------
if step 2 "Random Forest, 8 folds  (900 trees, unlimited depth -- the slow one)"; then
  python run_thesis_folds.py --model rf --from 1 --to 8 "${COMMON[@]}" \
    --jobs $RF_JOBS --threads $RF_THREADS --resume
fi

# -----------------------------------------------------------------------------
if step 3 "Compare the three families on identical out-of-fold rows -> the winner"; then
  python compare_models.py
fi

# -----------------------------------------------------------------------------
if step 4 "CanOSSEM benchmark, per region-year block"; then
  python build_canossem_block_metrics.py
  python make_canossem_block_heatmap.py
fi

# -----------------------------------------------------------------------------
if step 5 "Robustness: feature ablations  (5 experiments x 8 folds = 40 runs)"; then
  for a in ablation_no_aod ablation_no_burned ablation_no_fire ablation_no_merra ablation_met_only; do
    for f in $(seq 1 8); do
      echo "python run_lgbm_thesis_fold.py --fold $f ${COMMON[*]} --threads $THREADS --features-file features_ablation/$a.json --out-root outputs/robustness/$a"
    done
  done | pool $JOBS
fi

# -----------------------------------------------------------------------------
if step 6 "Robustness: alternative fold assignments  (4 seeds x 8 folds)"; then
  # NOTE measured 76-83% of held-out blocks shared with the thesis split, so a
  # null result here is weak evidence. Skip with: ./run_everything.sh 7
  for s in 101 202 303 404; do
    for f in $(seq 1 8); do
      echo "python run_lgbm_thesis_fold.py --fold $f --shard-root Data --case-plans-dir case_plans_foldalt_seed$s --threads $THREADS --out-root outputs/robustness/foldalt_$s"
    done
  done | pool $JOBS
fi

# -----------------------------------------------------------------------------
if step 7 "Robustness: leave-cells-out spatial CV  (7 folds, cell-partitioned shards)"; then
  for f in $(seq 1 7); do
    echo "python run_lgbm_thesis_fold.py --fold $f --shard-root Data_by_cell --case-plans-dir case_plans_spatialcv --threads $THREADS --out-root outputs/robustness/spatial_cv"
  done | pool $JOBS
fi

# -----------------------------------------------------------------------------
if step 8 "Robustness: corrector on out-of-fold residuals  (8 fits per fold)"; then
  for f in $(seq 1 8); do
    echo "python run_lgbm_oof_corrector_fold.py --fold $f --shard-root Data --case-plans-dir case_plans --threads $THREADS --out-root outputs/robustness/oof_corrector"
  done | pool $JOBS
fi

# -----------------------------------------------------------------------------
if step 9 "External validation: train all Ontario, test all Quebec"; then
  python run_lgbm_thesis_fold.py --fold 1 --shard-root Data_on_plus_qc \
    --case-plans-dir case_plans_external_qc --threads $VCPU \
    --out-root outputs/external_qc
fi

# -----------------------------------------------------------------------------
if step 10 "SHAP, LightGBM only, 500 rows x 8 folds"; then
  # LightGBM only, by decision: it is the winning family (block R2 0.7606 against
  # XGBoost 0.7490 and Random Forest 0.7063), and the figures in step 11 read
  # outputs/lgbm_thesis regardless -- they have no family selector.
  #
  # The other two are left out on cost, not oversight. XGBoost would be cheap
  # (native pred_contribs), but Random Forest has no native contribution path in
  # sklearn and goes through shap.TreeExplainer at a measured 47.8 s/row against
  # 73.6M nodes: ~100 hours for 4,000 rows. To add them back:
  #   python make_fold_shap.py --model xgb --rows-per-fold 500
  #   python make_fold_shap.py --model rf  --rows-per-fold 500   # ~100h, resumable
  python make_fold_shap.py --model lgbm --rows-per-fold 500
fi

# -----------------------------------------------------------------------------
if step 11 "SHAP figures, all 8 folds"; then
  # All 8, not just one. The reviewer asked for the corrected script to be run for
  # "any other fold used in the thesis figures", and --fold defaults to 4 -- so a
  # bare call silently covers one fold and leaves the rest carrying the old
  # mislabelled panels. Cheaper to regenerate all of them than to guess which the
  # thesis prints.
  #
  # Only fig4 (beeswarm) and fig5 (dependence) were ever wrong: they selected plot
  # columns by importance rank against data in canonical order. The bar chart,
  # repaired-share and family figures, and shap_feature_importance.csv, index by
  # canonical column throughout and were always correct -- no reported number was
  # affected.
  #
  # Each fold recomputes ~21k rows x 651 features at ~0.104 s/row, about 37 min
  # single-threaded, so they run 4-wide rather than end to end.
  for f in $(seq 1 8); do
    echo "python shap_fold_analysis_fixed.py --fold $f --stage all"
  done | pool $JOBS
fi

# -----------------------------------------------------------------------------
if step 12 "Final model on all 72 blocks, then the province-wide raster"; then
  python train_final_ontario_model.py
  # predict_raster_cells.py defaults --grid-dir to a Windows path, so it must be
  # pointed at the 60 GB disk where the 144 monthly rasters actually live.
  python predict_raster_cells.py --grid-dir "$GRID_DIR" --dry-run
  python predict_raster_cells.py --grid-dir "$GRID_DIR"
fi

# -----------------------------------------------------------------------------
echo; printf '=%.0s' {1..78}; echo
echo "COMPLETE in $(( ($(date +%s) - STARTED) / 60 )) min"
printf '=%.0s' {1..78}; echo
