# ONPM25 — Ontario daily PM2.5, two-stage model

Region-year cross-validation for a two-stage PM2.5 estimator: a Stage-1 regressor over
651 predictors plus a residual "smoke corrector", evaluated over 8 outer folds of 72
region-year blocks (2012–2023, 6 Ontario regions, 169,882 monitored grid-cell-days).

---

## Setup after cloning

```bash
git clone https://github.com/ShaojieDong503/ONPM25.git
cd ONPM25

python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# .venv\Scripts\Activate.ps1         # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
```

Check it worked — the three learner versions are pinned and matter:

```bash
python -c "import lightgbm,xgboost,sklearn;print(lightgbm.__version__,xgboost.__version__,sklearn.__version__)"
# expect: 4.6.0 3.2.0 1.9.0
```

Then confirm the data came down with the clone:

```bash
ls Data_zerofilled/manifest.json && ls Data_zerofilled/pair_blocks | wc -l    # expect 72
ls case_plans/GROUP_01.json
```

All commands below use **relative paths**, so they work from the repo root on any
machine. Run them from the repo root.

| Path | What it is |
|---|---|
| `Data/` | original shards (72 blocks, contains NaNs) |
| `Data_zerofilled/` | shards after zero-fill — **use this for runs** |
| `case_plans/` | `GROUP_01.json` … `GROUP_08.json`, the 8 fold definitions |
| `outputs/` | run results |
| `cloud/` | scripts to run folds on Google Cloud |

---

## 1. Check the data

```bash
python pm25_data_check.py --shard-root Data --out-dir data_check_reports
```

## 2. Zero-fill the NaNs

Writes a **copy**; the input is not modified. Already done — `Data_zerofilled/` is in the
repo. Re-run only if you rebuild `Data/`.

```bash
python pm25_zero_fill.py \
  --shard-root  Data \
  --output-root Data_zerofilled \
  --report-dir  zero_fill_reports
```

Add `--in-place` to overwrite the originals instead.

## 3. Run one fold

```bash
python run_lgbm_thesis_fold.py \
  --fold 1 \
  --shard-root     Data_zerofilled \
  --case-plans-dir case_plans \
  --out-root       outputs/lgbm_thesis_zerofilled
```

Swap in `run_xgb_thesis_fold.py` or `run_rf_thesis_fold.py` for the other learners.

`--shard-root` and `--case-plans-dir` are **required**: the scripts' built-in defaults
point at the original author's machine, and `case_plans/` is not inside
`Data_zerofilled/`.

## 4. Run several folds in a row

```bash
python run_thesis_folds.py --model lgbm --from 1 --to 8 \
  --shard-root     Data_zerofilled \
  --case-plans-dir case_plans \
  --out-root       outputs/lgbm_thesis_zerofilled
```

| Option | Effect |
|---|---|
| `--model lgbm\|xgb\|rf` | which learner |
| `--from 5 --to 8` | a fold range |
| `--folds 2,4,7` | specific folds |
| `--resume` | skip folds that already have `metrics.json` |
| `--dry-run` | print the commands, run nothing |
| `--stop-on-error` | stop at the first failure (default: attempt every fold) |

Per-fold logs go to `<out-root>/_logs/`. Live detail is in
`<out-root>/GROUP_0N/training.log` — check there, not the launcher console, if a run
looks stalled.

## 5. SHAP

```bash
python make_fold_shap.py                    # all 8 folds, lgbm
python make_fold_shap.py --folds 1,2        # a subset
python make_fold_shap.py --raw-sample 20000 # also keep raw SHAP for plots
```

| Option | Effect |
|---|---|
| `--folds 1,2` | which folds |
| `--rows-per-fold N` | subsample rows per fold |
| `--raw-sample N` | keep N raw SHAP rows for plotting |
| `--out-dir` / `--out-root` / `--shard-root` | paths |

## 6. Compare against CanOSSEM

```bash
python build_canossem_block_metrics.py                    # auto-detect the obs source
python build_canossem_block_metrics.py --obs-from shards  # observations from the shard frames
```

Then the heatmap:

```bash
python make_canossem_block_heatmap.py                # R2 (default)
python make_canossem_block_heatmap.py --metric rmse
python make_canossem_block_heatmap.py --metric both
```

## 7. Run on Google Cloud

Needs the `gcloud` CLI, authenticated, with a project and a bucket. Edit `BUCKET` at the
top of the scripts to your own.

```bash
bash cloud/upload_inputs.sh          # uploads Data_zerofilled + case_plans + code
bash cloud/launch_vm.sh xgb 1 8      # a VM that runs folds 1-8 and shuts itself down
bash cloud/fetch_results.sh xgb      # download results
```

Watch it without SSH:

```bash
gcloud storage cat gs://<your-bucket>/post_defense/runs/xgb/_status/progress.log
```

See `cloud/README.md` for machine sizing — it matters for Random Forest.

---

## Notes

**Timing** (12-core laptop): LightGBM ~8m41s per fold, so ~70 min for all 8. XGBoost is
roughly 2.5x that. Random Forest is far slower.

**Memory.** LightGBM and XGBoost fit comfortably in 16 GB. **Random Forest does not** —
at 900 trees with `max_depth=None` each forest is ~9 GB and there are two per fold, so
budget 15–20 GB of RAM. Run RF on the cloud VM (`cloud/launch_vm.sh rf`) or a
large-memory machine.

**Reproducibility.** Seed 2026, fixed hyperparameters, no inner tuning, no early stopping
against the held-out fold. The same fold re-run on the same machine reproduces
bit-identically. Across machines expect agreement to several significant figures, not the
last digit — thread count changes floating-point reduction order inside the tree learners.

**Feature names.** The shards store rolling aggregates with a `_lag1_` infix
(`X_lag1_roll3_mean`) while `manifest.json` records the canonical name (`X_roll3_mean`).
The loaders resolve this: every run logs `direct=213 | translated=438 | absent=0`. If
`absent` is ever non-zero, stop — predictors are reaching the model as fabricated zeros.
