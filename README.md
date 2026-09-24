# Ontario daily PM2.5 — post-defense code

A two-stage model for daily PM2.5 at 1 km over Ontario, 2012–2023.

* **Stage 1** — LightGBM on 651 predictors: MERRA-2 reanalysis, NARR meteorology,
  satellite AOD, VIIRS/HMS fire, burned area, land cover, roads, calendar.
* **Corrector** — a second model fitted to Stage 1's **training residuals** from 40 raw
  inputs, each paired with a missingness indicator (80 columns).
* `pred_final = pred_stage1 + pred_corrector`

Validation is an 8-group outer holdout over 72 region-year blocks (12 years × 6
regions): 169,882 monitored grid-cell-days across 41 cells, seed 2026, fixed
hyperparameters, no inner tuning, no early stopping. Out-of-fold predictions are not
clipped at zero.

---

## Code structure

The design rule is that **one file owns the modelling behaviour** and everything else
borrows it. `run_lgbm_thesis_fold.py` is that file. It defines the loader, the
651-feature resolution, the corrector pool rule, the fill rules, the metrics and the
hyperparameters. No other script reimplements any of them.

```
                      imports third-party only (numpy, pandas, lightgbm, sklearn)
                      -- no project module. Runnable on its own.
  run_lgbm_thesis_fold.py
            │
            │  is imported by
            ▼
  thesis_core.py ......... re-exports the above unchanged, adds fit/score helpers
            │
            │  is imported by
            ▼
  ├─ robustness_runner.py ──imports──> robustness_features.py
  ├─ train_final_ontario_model.py
  ├─ quebec_external_test.py
  ├─ predict_raster_cells.py
  ├─ refit_correctors.py          (also imports run_lgbm_thesis_fold directly)
  └─ audit_corrector_table.py     (also imports run_lgbm_thesis_fold directly)

  make_fold_shap.py           ──imports──> run_lgbm_thesis_fold.py
  shap_fold_analysis_fixed.py ──imports──> run_lgbm_thesis_fold.py

  run_thesis_folds.py         ──subprocess──> run_{lgbm,xgb,rf}_thesis_fold.py
  run_all_robustness.py       ──subprocess──> robustness_runner.py
  run_revision_local.py       ──subprocess──> the stage scripts

  run_xgb_thesis_fold.py      standalone, imports no project module
  run_rf_thesis_fold.py       standalone, imports no project module
```

Read the arrows downward: `run_lgbm_thesis_fold.py` depends on nothing in this project,
`thesis_core.py` depends on it, and the design/application scripts depend on
`thesis_core`. The drivers do not import at all -- they invoke scripts as subprocesses,
which is why a failing fold cannot take the whole run down with it.


`thesis_core.py` is deliberately small. It re-exports `load_blocks`, `metrics`,
`load_fold_plan`, `STAGE1_PARAMS`, `CORRECTOR_PARAMS` unchanged, and adds only what a
second design needs: a `TwoStage` dataclass, `fit_two_stage`, `fit_and_score`,
`prediction_table`, `save_model`/`load_model`. So an ablation and the thesis fold script
fit *the same model by the same code*, differing only in which data goes in.

### Layers

| layer | files | role |
|---|---|---|
| **Reference model** | `run_lgbm_thesis_fold.py`, `run_xgb_thesis_fold.py`, `run_rf_thesis_fold.py` | one fold, end to end. XGB and RF are standalone ports, each with its own copy of the corrector logic |
| **Library** | `thesis_core.py` | re-export + fit/score helpers |
| **Drivers** | `run_thesis_folds.py`, `run_all_robustness.py`, `run_revision_local.py` | sequence folds/experiments/stages, handle resume and logging |
| **Designs** | `robustness_runner.py`, `robustness_features.py` | ablations, alternative folds, spatial CV, leave-one-cell-out |
| **Application** | `train_final_ontario_model.py`, `quebec_external_test.py`, `predict_raster_cells.py` | final fit, external test, 1 km surface |
| **Attribution** | `make_fold_shap.py`, `shap_fold_analysis_fixed.py` | 8-fold SHAP; one fold in depth with figures |
| **Verification** | `audit_corrector_table.py`, `refit_correctors.py`, `compare_models.py` | data gate, cheap corrector refit, three-family comparison |
| **Auxiliary** | `pm25_data_check.py`, `build_canossem_block_metrics.py`, `make_canossem_block_heatmap.py` | shard audit, CanOSSEM comparison |

XGBoost and Random Forest are **not** wired through `thesis_core`. Each carries its own
copy of `build_corrector_pool` / `fit_corrector_fill_values` / `transform_corrector`.
That duplication is real and worth knowing: a change to corrector handling must be made
in all three files, not one.

---

## Inputs

Four read-only sources. Nothing else is required.

| path | contents |
|---|---|
| `Data/` | `manifest.json` (651 canonical features), `pair_blocks/PAIR_<year>_<region>/frame.parquet` ×72, `temporal_feature_selection.txt` (73 bases) |
| `case_plans/` | `GROUP_01..08.json` — 63 training / 9 held-out blocks per fold |
| `Data_external/qc/` | Quebec shards for the external test (181,401 rows, 50 cells) |
| `ontario_surface_build/` | `temporal_tables/temporal_YYYY_MM.parquet` ×144 + `static_features.parquet` |

`Data/` holds **raw** shards with missing values present as NaN. That matters — the
corrector derives its indicators from them.

### Two fills, and why they differ

`load_one_block` returns **two** objects from one parquet: the Stage-1 matrix `X`, which
is zero-filled, and the frame `df`, which keeps its NaN. Stage 1 uses `X`; the corrector
reads `df`.

| fill | where | applies to | on disk |
|---|---|---|---|
| NaN → `0.0` | `load_one_block` | Stage-1 matrix `X` (651 cols) | no — in memory |
| NaN → training median, or a beyond-range sentinel for distances | `transform_corrector` | corrector matrix (40 raw → 80 cols) | no — in memory |

Corrector fills come from the **training portion of each fold only**, and the indicator
is computed **before** any fill, so a genuine measured zero stays zero with its flag off.

An earlier version ran against a copy of the shards with NaN already replaced by 0 on
disk. Stage 1 was unaffected — `nan_to_num` is idempotent — but every corrector indicator
was constant and every median was pulled toward zero. That copy is deleted, and
`fit_corrector_fill_values` now **aborts** if a training pool has no missing values at
all, in all three fold scripts.

---

## Quick start

```bash
pip install -r requirements.txt
python audit_corrector_table.py --shard-root Data   # gate: expect VERDICT OK
python run_revision_local.py                        # everything, resumable
```

`run_revision_local.py` skips any stage whose output exists. `--force` overrides,
`--only`/`--skip` select stages, `--dry-run` prints the plan.

## Stages individually

`--model lgbm | xgb | rf` selects which of the three fold scripts the driver invokes.

```bash
python run_thesis_folds.py --model lgbm --from 1 --to 8 \
  --shard-root Data --case-plans-dir case_plans --out-root outputs/lgbm_thesis

python train_final_ontario_model.py --shard-root Data \
  --out-dir outputs/final_ontario_model

python quebec_external_test.py --province QC --shard-root Data \
  --external-root Data_external/qc --model-dir outputs/final_ontario_model \
  --out-dir outputs/external_qc

python run_all_robustness.py --shard-root Data --case-plans-dir case_plans \
  --out-dir outputs/robustness --jobs 4

python predict_raster_cells.py --grid-dir <ontario_surface_build> \
  --shard-root Data --model-dir outputs/final_ontario_model \
  --out-dir outputs/raster_prediction
```

`--case-plans-dir` is required: `case_plans/` sits beside `Data/`, not inside it.

LightGBM's `n_jobs` overrides `OMP_NUM_THREADS`, so when several experiments run at once
set `PM25_LGBM_THREADS` — otherwise each job grabs every core and they spin against each
other at OpenMP barriers.

## Attribution

```bash
python make_fold_shap.py --folds 1,2,3,4,5,6,7,8 --rows-per-fold 1000 \
  --shard-root Data --out-root outputs/lgbm_thesis --out-dir outputs/shap_all8_n1000

python shap_fold_analysis_fixed.py --fold 1 --stage all \
  --shard-root Data --out-root outputs/lgbm_thesis
```

Stage 1 and the corrector are **never summed**. The corrector takes `pred_stage1` as an
input, so adding the two feature-wise counts the 651 Stage-1 features twice — once
directly, once through that channel. `make_fold_shap.py` records `stages_combined:
false` and keeps them in separate tables.

These are shares of **attribution**, not of predictive improvement: the corrector's
entire contribution to skill is Stage-1 R² 0.7493 → 0.7505.

Per-feature values are the least stable level — 73 base variables each carry 7
correlated derivatives and the neighbourhood radii overlap, so TreeSHAP divides credit
among correlated features by tree structure. Use the family and base-variable rollups
for any claim.

## Verification

```bash
python audit_corrector_table.py --shard-root Data --fold-blocks all
python refit_correctors.py --family lgbm --out-root outputs/lgbm_thesis --verify
python compare_models.py --out-dir outputs/model_comparison
```

`refit_correctors.py` refits a corrector from a saved Stage-1 model without refitting
Stage 1 — ~40 s/fold against minutes to hours — and asserts per fold that the saved
Stage-1 predictions reproduce to 1e-6. That assertion is the licence for the shortcut,
so a failure there should stop the run, not be relaxed.

---

## Outputs

```
outputs/
  lgbm_thesis/ xgb_thesis/ rf_thesis/   GROUP_01..08: predictions, metrics, models
  final_ontario_model/                  trained on all 72 blocks, no holdout
  external_qc/                          Quebec external test
  robustness/                           ablations, fold variants, spatial CV, LOCO
  raster_prediction/                    1 km daily surface
  shap_all8_n500/ shap_all8_n1000/      8-fold attribution, both stages
  model_comparison/                     LightGBM vs XGBoost vs Random Forest
```

XGBoost and Random Forest had **only their correctors refitted**; their Stage-1 models
were reused unchanged — valid because Stage 1 is bit-identical under the corrector fix,
and re-verified per fold at the 1e-14 level. Each `GROUP_0N/PROVENANCE.json` records it.

## Cloud

`cloud/REVISION_RUNBOOK.md` covers the GCP path end to end. The upload script will not
proceed unless `Data/` passes the corrector audit.

## Environment

Pinned in `requirements.txt`. LightGBM in particular must match: `refit_correctors.py`
asserts Stage-1 predictions reproduce to 1e-6, and a version drift breaks that assertion
loudly — which is the intent.
