# How this ran on the VM

Post-defence rerun of the Ontario daily PM2.5 pipeline, 2026-09-24.
Everything below is what actually happened, including what went wrong.

---

## 1. The machine

```
instance-20260923-184043      us-central1-a
e2-standard-32                32 vCPU / 125 GB RAM
CPU platform                  Intel Broadwell
external IP                   34.30.154.225
```

Note it is **e2**, not n2. Same core count, but a mixed/older CPU platform —
expect roughly 10-20% lower per-core throughput than an n2 of the same size.

### Disks

The 10 GB boot disk cannot hold this pipeline; Random Forest alone writes ~13 GB
per fold. Two data disks were already attached but **not mounted** (they were
absent from `/etc/fstab`, so `df` showed only `/`):

```
/dev/sda   10 GB  pd-balanced   /          boot, OS only
/dev/sdc  200 GB  pd-balanced   /mnt/work  code, data, all outputs
/dev/sdb   60 GB  pd-balanced   /mnt/grid  raster inputs (pre-existing)
```

Both were added to `/etc/fstab` by UUID with `nofail`, so a reboot remounts them.

`/mnt/grid/grid/` holds 144 monthly raster parquets (`temporal_2012_01` ..
`temporal_2023_12`) plus `static_features.parquet`. **These exist only on the VM** —
they were not in the local project — so they are the one irreplaceable input here.

The 200 GB disk arrived holding 104 GB of outputs from a 2026-09-20 run. Its 42 MB
of metrics and predictions were archived to `/mnt/work/old_run_sep20_results.tgz`
before the 103.6 GB of refittable model pickles were deleted.

---

## 2. Environment

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Versions were verified identical to the workstation that produced the reported
results — this matters because LightGBM tree construction differs subtly between
minor versions:

```
lightgbm 4.6.0   xgboost 3.2.0   scikit-learn 1.9.0   pandas 3.0.3
numpy 2.4.6      pyarrow 24.0.0  joblib 1.5.3
matplotlib 3.10.9  shap 0.52.0   scipy 1.17.1
```

`matplotlib`, `shap` and `scipy` were missing from `requirements.txt` and were added
during this run. They are not needed to fit or score anything, which is why their
absence stayed hidden until a figure step failed.

---

## 3. Getting the data there

**Do not use `scp` for the bulk transfer.** Measured on this link:

```
gcloud compute scp     85 kB/s     ~5 hours for the data
rclone (multi-stream)  ~30 MB/s    321 MB in 11 seconds
```

The bottleneck was SSH's single stream, not the uplink. Transfer used:

```
rclone copy <src> :sftp:/mnt/work/pm25/ \
  --sftp-host=34.30.154.225 --sftp-user=<user> \
  --sftp-key-file=~/.ssh/google_compute_engine \
  --multi-thread-streams=8 --multi-thread-cutoff=10M
```

Note `rclone` does **not** preserve the executable bit — `chmod +x *.sh` after upload.

The payload was cut from 3.1 GB to 604 MB by not shipping what could be derived:

| not uploaded | how it was obtained instead |
|---|---|
| `Data_by_cell` (427 MB) | regenerated on the VM: `python make_spatialcv_shards.py` |
| `Data_on_plus_qc` (754 MB) | its 72 Ontario blocks are **byte-identical** to `Data`; only the 4 QC blocks were sent and the Ontario blocks hard-linked (`cp -al`) |
| `outputs/lgbm_thesis` (1.5 GB) | refit on the VM in step 0 — cheaper than uploading |

---

## 4. The parallel design

This is the part that matters for reuse.

All fold scripts default to `n_jobs=-1`, and **LightGBM's `n_jobs` overrides
`OMP_NUM_THREADS`** — exporting that variable does nothing. Four folds left at the
default would each claim all 32 cores: 128 threads on 32, where the OpenMP barriers
spin instead of working and a 9-minute fit can take hours.

So a `--threads` argument was added to `run_lgbm_thesis_fold.py` (and its generated
xgb/rf twins) and to `run_lgbm_oof_corrector_fold.py`, and `--jobs` to
`run_thesis_folds.py`. **`run_thesis_folds.py` refuses `--jobs > 1` without
`--threads`**, so the bad configuration cannot be produced by accident.

The chosen split, for 32 vCPUs:

```
JOBS=4  THREADS=8        all families
```

LightGBM and XGBoost stop scaling near 8 threads (3.26x at 8, 3.30x at 12), so more
thin jobs beat fewer fat ones. Random Forest scales further (5.00x at 8, 5.91x at 12)
but still sublinearly, so for total throughput over 8 folds `4x8` (~20x aggregate)
beats `2x16` (~14x). Four concurrent RF fits hold ~24 GB against 125 GB of RAM.

Measured: 4 concurrent folds ran at ~660% CPU each — ~26 of 32 cores, no
oversubscription. Verify with `uptime`, not with the Cloud Console graph, which is a
lagging time-average over a window that includes idle time.

`run_everything.sh` drives everything, one command per step:

```
./run_everything.sh          # all steps
./run_everything.sh 5        # start at step 5
./run_everything.sh 5 8      # steps 5 through 8
```

It runs under `tmux` so the pipeline survives the SSH session closing:

```
tmux new-session -d -s pm25 'source .venv/bin/activate && ./run_everything.sh 2>&1 | tee -a pipeline.log'
```

---

## 5. What ran, and how long it took

| step | what | wall-clock |
|---|---|---|
| 0 | LightGBM, 8 folds, 4x8 | 17 min |
| 1 | XGBoost, 8 folds, 4x8 | 38 min |
| 2 | Random Forest, 8 folds, 4x8 | 6 h 12 m |
| 3 | three-family comparison | seconds |
| 4 | CanOSSEM benchmark + heatmap | ~1 min |
| 5 | 5 ablations x 8 folds | 56 min |
| 6 | 4 alternative fold splits x 8 | 73 min |
| 7 | leave-cells-out spatial CV, 7 folds | 17 min |
| 8 | out-of-fold corrector, 8 fits/fold | 1 h 51 m |
| 9 | external validation on Quebec | ~10 min |
| 10 | SHAP, LightGBM, 500 rows x 8 folds | 5 min |
| 11 | SHAP figures, all held-out rows per fold | ~70 min per fold, 4 at a time |
| 12 | final model + 46.7M cell-day raster | 2 h 34 m |

Steps 4-9 together: 261 min.

Step 10 was reduced to LightGBM only. XGBoost would have been cheap, but
scikit-learn has no native contribution path so Random Forest goes through
`shap.TreeExplainer` at a measured 47.8 s/row against 73.6M nodes — about 100 hours.

Step 11 is slow for a legitimate reason, not a threading fault: it explains **every**
held-out row (~21k per fold, not a sample), and exact TreeSHAP is
O(trees x leaves x depth^2) — 2000 trees x 255 leaves is ~1e8 operations per row.
The pickled models carry `n_jobs=8`, so a 4-wide pool is already a correct 32-thread
match. Four concurrent folds take ~70 min each against 47 min for one alone; the
extra 1.47x is memory bandwidth, not thread contention.

---

## 6. Things that went wrong

**Fold outputs landed in a directory named after a Windows path.** The fold scripts
defaulted `--out-root` to `Path(r"D:\lambda\...\thesis_lgbm_grouped_runs")`. On Linux
a backslash is an ordinary filename character, so that is neither absolute nor an
error — it is a *relative* name containing backslashes. Python silently created
`./D:\lambda\...\thesis_rf_grouped_runs` and wrote 104 GB into it, where
`compare_models.py` does not look. Step 3 then died with `StopIteration` on an empty
family list after 7 hours of fitting. Nothing was lost; `relocate_win_outputs.py`
moved all 24 folds into `outputs/{lgbm,xgb,rf}_thesis`. The scripts now fall back to
`outputs/<family>_thesis` when the Windows path is absent.

**A SHAP figure guard fired on correct data.** `shap_values.npy` is stored float32
while the importance table's `mean_abs_shap` was computed in float64 before the cast,
so an honest match still differs by ~1.9e-6 relative — and the guard used
`rtol=1e-6`. The tolerance was re-derived from measurement: the smallest relative gap
between two *distinct* top-20 features is 2.7e-3, ~1400x larger, so `rtol=1e-4` sits
between them. Verified that the guard still rejects a two-column swap and the
original importance-order bug.

**`set -e` halts the whole script on any step failure.** That is deliberate — it
stops a broken dependency propagating — but it means one missing input strands
everything after it. Two steps failed this way and were fixed and resumed:
CanOSSEM data was not on the VM, and `matplotlib` was not installed.

---

## 7. Re-running

```
ssh into the VM, then:
cd /mnt/work/pm25 && source .venv/bin/activate
./run_everything.sh 0 12
```

Fold steps pass `--resume`, so completed folds are skipped rather than refit. The
step 11 SHAP cache under `outputs/shap_lgbm_n500/_cache` also resumes.

Check progress with the step banners, not with CPU load — a dead pipeline and a busy
one look identical if you only watch `uptime`:

```
grep -E '^\[[0-9]+\] ' pipeline.log | tail
tmux attach -t pm25
```
