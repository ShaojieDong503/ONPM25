# Running XGBoost and Random Forest folds on Google Cloud

Four scripts. Upload once, launch a VM per learner, fetch results.

```bash
cd /d/lambda/Post_Defense_Code

bash cloud/upload_inputs.sh            # ~408 MB, once
bash cloud/launch_vm.sh xgb 1 8        # XGBoost, folds 1-8
bash cloud/launch_vm.sh rf  1 8        # Random Forest, folds 1-8
bash cloud/fetch_results.sh xgb        # when done
```

Your environment is already configured: project `my-project-pm25predictionon`,
zone `us-central1-c`, bucket `gs://my-project-pm25predictionon-ontario-out`.

---

## What gets uploaded

Only inputs — **408 MB**, not the 2.3 GB the folder weighs:

| uploaded | size | why |
|---|---|---|
| `Data_zerofilled/` | 407 MB | the shards the folds read |
| `case_plans/` | 1.2 MB | the 8 GROUP_0N.json fold definitions |
| `run_*_thesis_fold.py`, `run_thesis_folds.py` | ~130 KB | the code |

Skipped: `outputs/` (1.5 GB — those are *results*), `Data/` (the pre-zero-fill copy,
redundant once `Data_zerofilled` exists) and `zero_fill_reports/`.

`upload_inputs.sh` refuses to proceed unless it finds 72 pair blocks and 8 case
plans, so a half-synced input tree fails at upload rather than 40 minutes into a run.

---

## Machine sizing — the two learners are very different

**XGBoost** is CPU-bound with a modest footprint: `hist` tree method, 3000 rounds,
depth 8. `n2-standard-16` (16 vCPU, 64 GB, 200 GB disk) is comfortable.

**Random Forest is the expensive one**, and it is worth understanding why before
launching. The thesis configuration is

```python
n_estimators=900, max_depth=None, min_samples_leaf=2, max_features=0.6
```

Unlimited depth with `min_samples_leaf=2` over 148,785 training rows grows roughly
150,000 nodes per tree. At ~65 bytes per sklearn node that is **~9 GB for the
stage-1 forest alone**, held in RAM during the fit and written again to `.pkl`.
The corrector is a second forest with the same `n_estimators`. Budget on the order
of **15–20 GB RAM and ~15 GB of model files per fold**, so ~120 GB of disk across 8
folds. Hence the default `n2-highmem-16` (16 vCPU, **128 GB**) with an 800 GB disk.

`max_features=0.6` also means each split evaluates 391 of 651 features, which makes
RF far slower per tree than the depth-18 variant used for the cloud runs in the
older bundle. The bundle's own README flags this configuration as "dramatically
slower" — that warning applies here.

| | machine | disk | why |
|---|---|---|---|
| `xgb` | `n2-standard-16` | 200 GB | CPU-bound, small models |
| `rf` | `n2-highmem-16` | 800 GB | ~9 GB per forest, two forests per fold |
| `lgbm` | `n2-standard-16` | 200 GB | for completeness; already run locally |

Override either: `MACHINE=n2-highmem-32 DISK=1000 bash cloud/launch_vm.sh rf 1 8`.

### Runtime and cost — measure, don't trust my estimate

LightGBM took **8m41s per fold** locally. XGBoost is historically ~2.5× LightGBM on
this problem, so expect roughly 20–25 min per fold, ~3 h for eight. Random Forest at
this configuration I genuinely cannot estimate within a factor of two — it could be
30 minutes or three hours per fold.

**Run one RF fold first and read the timing before committing to eight:**

```bash
bash cloud/launch_vm.sh rf 1 1
```

Then scale. At us-central1 on-demand list prices, `n2-standard-16` is ~$0.78/h and
`n2-highmem-16` ~$1.05/h, so a 3-hour XGBoost run is a couple of dollars and RF
depends entirely on that first measurement.

### Spot VMs are not available on this project

`PREEMPTIBLE_CPUS` quota in `us-central1` is **0**, so `SPOT=1` will fail. The
launcher checks this and tells you rather than surfacing an opaque quota error.
Request an increase under IAM & Admin → Quotas if you want ~70% off; otherwise the
on-demand default is correct.

Your other quotas are ample: `CPUS` 200, `N2_CPUS` 200, `DISKS_TOTAL_GB` 4096, all
currently unused.

---

## Preemption and crash safety

`vm_startup.sh` syncs results to GCS **after every fold**, not at the end. A VM that
dies at fold 6 loses fold 6 only; relaunching resumes, because the startup script
pulls existing results back down first and `run_thesis_folds.py` skips any fold that
already has `metrics.json`.

The VM shuts itself down when finished (`shutdown_when_done=1`). That stops CPU
billing but **not disk billing** — delete the instance to stop paying for the disk:

```bash
gcloud compute instances delete pm25-rf-<timestamp> --zone=us-central1-c
```

Pass `KEEP_ALIVE=1` to keep the VM up for debugging.

---

## Watching a run

No SSH needed — the startup script writes a breadcrumb log to GCS:

```bash
gcloud storage cat gs://my-project-pm25predictionon-ontario-out/post_defense/runs/rf/_status/progress.log
```

```
2026-09-20T14:02:11Z | installing python environment
2026-09-20T14:04:55Z | inputs ready (72 blocks)
2026-09-20T14:05:02Z | fold 1 START
2026-09-20T15:31:40Z | fold 1 DONE in 5198s
```

A `_status/COMPLETE` object appears at the end. For full detail:

```bash
gcloud compute ssh <name> --zone=us-central1-c --command="sudo tail -f /var/log/pm25_run.log"
```

---

## Fetching results

```bash
bash cloud/fetch_results.sh xgb          # metrics + predictions only
MODELS=1 bash cloud/fetch_results.sh rf  # also the .pkl/.txt model files
```

Model files are excluded by default — for RF they are several GB per fold and you
rarely need them locally to write up results. The script prints a per-fold metrics
table after downloading.

---

## Reproducibility

The VM pins `scikit-learn==1.9.0`, `lightgbm==4.6.0`, `xgboost==3.2.0` — the versions
the thesis runs used and the ones the verification suite asserts. Tree-learner
releases change binning and tie-breaking, so an unpinned install is a different
model, not just a different build.

Seed 2026, fixed hyperparameters, no inner tuning — identical to the local runs, so
cloud and local folds are directly comparable.

One caveat: results will not be **bit-identical** to a local run. The VM has a
different CPU and thread count, which changes floating-point reduction order inside
the tree learners. Metrics agree to several significant figures; they will not match
to the last digit.
