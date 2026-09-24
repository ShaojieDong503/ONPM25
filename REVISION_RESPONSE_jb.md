# Response to comments — *A Machine Learning Framework for Developing an Ontario-Specific Daily PM2.5 …*

Reviewer: **JeffB** · 16 annotated comments across pp. 7–40 of the marked-up PDF.

Numbers cited below are from the repaired 651-feature run (out-of-fold, 169,882
monitored grid-cell-days, 72 region-year blocks, 8 folds, seed 2026) unless noted.
Items marked **[pending]** depend on experiments still running; I will not write them
into the manuscript until the numbers are in hand.

---

## Summary of the substantive changes

Four of the comments converge on one question — **what does the model add beyond the
MERRA-2 reanalysis?** — and I now treat that as the paper's central analytical claim
rather than a discussion aside. Three more ask for evidence that already exists but
was not reported (external provinces, the CanOSSEM comparison, the extreme-value
sample size). The remainder are structural or framing.

| # | Comment (p.) | Response | Status |
|---|---|---|---|
| 1 | "nice" (7) | — | noted |
| 2 | CanOSSEM/wildfire fairness (7) | reframed, see §2 | revised |
| 3 | "be clear about what the science aspect is" (8) | rewritten objective (§3) | revised |
| 4 | surrounding states (18) | external test added, §4 | **new results** |
| 5 | methods written as results (18) | structural pass, §5 | revised |
| 6 | RMSE "pretty large" (20) | contextualised, §6 | revised |
| 7 | AOD is inside MERRA-2 (30) | accepted + tested, §7 | **new results** |
| 8 | "your explanation for why AOD is not used" (31) | §7 | revised |
| 9 | MERRA-2 and wildfire emissions (34) | §8 | **pending** |
| 10 | "what are the stakes" (35) | §9 | revised |
| 11 | under-prediction due to MERRA-2 (37) | §8 | **pending** |
| 12 | evaluate against the reanalysis (38) | §7, new analysis | **new results** |
| 13 | extremes and the development data (39) | §10, sample sizes | **new results** |
| 14 | time-series studies need temporal not spatial (39) | §11 | revised |
| 15 | errors usable in epidemiology (40) | §12 | revised, scoped |
| 16 | compare to her performance (40) | §13, table added | **new results** |

---

## 2. CanOSSEM comparison and wildfire emphasis (p. 7)

> *"not sure this is fair with CanOSSEM's emphasis on wildfire PM2.5"*

Accepted. The original text implied a like-for-like benchmark when the two products
have different design goals. Revised to state explicitly that CanOSSEM is optimised
for wildfire-smoke surveillance across Canada while this model is fitted to all
Ontario days, so a pooled comparison flatters the Ontario-specific model by
construction. The comparison is now reported **both pooled and stratified by
concentration band and by year**, so the reader can see where each product is
stronger. See §13 for the numbers.

## 3. "Be clear about what the science aspect is" (p. 8)

Accepted. The objectives paragraph described procedure rather than a scientific
question. Rewritten to lead with the question the thesis actually answers: *how much
predictive information about daily Ontario PM2.5 exists beyond a global aerosol
reanalysis, and is it enough to support exposure assignment at 1-km daily
resolution?* The methodological machinery (blocking, correctors, uncertainty) is now
framed as what is needed to answer that honestly, not as the contribution itself.

## 4. "What happened to the plan to use surrounding states?" (p. 18)

Fair — the plan was stated and then dropped without explanation. The external test is
now **run and reported**. The nearest available out-of-province monitors are Canadian
rather than US (the assembled panel covers QC, MB, SK, AB, BC), so the revision uses
Quebec as the primary external province and reports the others as a secondary panel.

**Quebec, never seen in training** — 181,401 cell-days, 50 grid cells, 60 stations,
2010–2024, scored with the final all-Ontario model:

| | RMSE | MAE | R² | bias | within 3 | slope |
|---|---|---|---|---|---|---|
| Stage 1 | 5.050 | 2.558 | 0.364 | −1.111 | 0.734 | 0.445 |
| Final | **5.025** | 2.561 | **0.370** | −0.984 | 0.730 | 0.466 |

spatial R² **−0.122**, temporal R² **0.384**.

The honest reading, which the revision states plainly: transfer is **temporal, not
spatial**. The model tracks day-to-day variation in Quebec (temporal R² 0.38) but a
negative spatial R² means it does *worse than the Quebec mean* at ranking which cells
are more polluted. It also under-predicts by ~1 µg/m³. This is a considerably weaker
result than the Ontario block CV (R² 0.75) and is now reported as a limit on
transferability rather than buried.

## 5. Methods written as results (p. 18)

> *"show data/results for this statement since you wrote it here in results. If it is
> a methodological detail then it belongs in methods. Check this issue more broadly."*

Accepted, and treated as a structural pass over the whole Results chapter rather than
a single edit. Every paragraph in Results that asserts a design property without a
number has been either (a) moved to Methods, or (b) kept with the supporting figure
or table. The specific passage flagged — blocking by region-year reducing overlap —
moved to Methods, with the leakage check it implies now reported as a verification
result: every held-out block is scored only by a model that did not train on it,
checked programmatically for all 8 folds (0 overlapping blocks, 0 duplicated
cell-days across the assembled 169,882-row out-of-fold table).

## 6. "Pretty large given the low levels in Ontario" (p. 20)

Accepted and now contextualised rather than left as a bare number. Observed mean is
**6.99 µg/m³** and the median **5.83**, so an RMSE near 2.6 is ~37% of the mean. The
revision reports RMSE alongside the observed distribution and adds the
within-3 µg/m³ fraction (**0.895**) and the concentration-stratified errors, which
make the scale interpretable:

| band (µg/m³) | n | RMSE |
|---|---|---|
| < 12 | 149,692 | 1.78 |
| 12–25 | 18,926 | 3.72 |
| 25–50 | 1,124 | 9.43 |
| ≥ 50 | 140 | 48.06 |

## 7. AOD, MERRA-2, and what the model adds beyond the reanalysis (pp. 30, 31, 38)

> *"this is because AOD is included in the MERRA-2 data"*
> *"so your model should be evaluated based upon what it adds beyond the reanalysis
> estimates … you fail to physically think and articulate about why reanalysis
> contains most of the useful information"*

**Accepted — this is the most important comment in the set, and the mechanism given
is correct.** MERRA-2's aerosol analysis assimilates bias-corrected AOD (MODIS, MISR,
AVHRR, AERONET), so the satellite AOD columns are largely redundant once the
reanalysis is present. The suggested reference is incorporated.

The revision replaces the previous hedged "possible substitution" language with a
physical argument and a direct test.

**Attribution evidence.** TreeSHAP over one fold's 21,133 held-out cell-days
(exact TreeSHAP, additivity verified to 3.4e-06):

| family | predictors | share of \|SHAP\| |
|---|---|---|
| MERRA-2 aerosol | 204 | **42.9%** |
| MERRA-2 winds/levels | 200 | 14.9% |
| MERRA-2 surface flux | 128 | 14.8% |
| land cover / roads | 10 | 10.9% |
| local meteorology | 14 | 7.8% |
| VIIRS active fire | 68 | 5.4% |
| **satellite AOD** | **12** | **1.3%** |
| HMS smoke | 8 | 1.2% |
| burned area | 6 | 0.2% |

The single strongest predictor is `BCSMASS_wmean_1000km` (MERRA-2 black-carbon
surface mass, 1000 km neighbourhood) at 10.3% on its own; the four BCSMASS variants
together account for ~23%.

**Ablation evidence [pending].** Two experiments now running answer the comment
directly by construction — drop AOD entirely (12 predictors), and drop MERRA-2
entirely (532 predictors) — each re-fitting all 8 folds:

- `ablation_no_aod` → tests whether AOD carries anything the reanalysis lacks
- `ablation_no_merra` → quantifies how much the model depends on the reanalysis

The revised Discussion will state the ΔR² for both. The framing the comment asks for
is adopted regardless of the numbers: **the model's contribution is what it adds on
top of a reanalysis that already assimilates the satellite signal**, namely local
meteorology, land cover/roads, fire proximity, and a learned mapping from
coarse-resolution reanalysis mass to monitor-level concentration.

## 8. MERRA-2 and wildfire under-prediction (pp. 34, 37) **[pending]**

> *"how well does MERRA-2 include wildfire emissions?"*
> *"it'd be good to know how much the under-prediction is due to MERRA-2. Should look
> at its error for these and other cases."*

This requires evaluating MERRA-2's own surface PM2.5 proxy against the monitors on
the same cell-days — an analysis not yet run. It is the single most valuable addition
the comments identify, because it separates *"the model is bad at extremes"* from
*"the model's dominant input is bad at extremes."* MERRA-2 uses QFED biomass-burning
emissions, which are known to underestimate boreal smoke plumes.

Planned: regress observed PM2.5 on the MERRA-2 aerosol mass columns alone for the
smoke-affected blocks (2023, North), report its error, and compare to the model's, so
the under-prediction can be attributed. I have not run this and will not write a
conclusion about it until I have.

## 9. "What are the stakes" (p. 35)

Accepted. The uncertainty paragraph described the band widths without saying what
follows from them. Revised to state the consequence: the intervals are widest in the
smoke-affected North and in 2023, i.e. exactly where exposure misclassification would
most affect a health analysis, so studies of wildfire-smoke health effects using this
surface should propagate the intervals rather than treat the point estimate as known.

## 10. Extremes and the development data (p. 39)

> *"what does the inability to do the high cases say about the development data and
> lack of cases to learn from?"*

This has a direct quantitative answer, now added. Of 169,882 monitored cell-days:

| | n | share |
|---|---|---|
| ≥ 25 µg/m³ | 1,264 | 0.74% |
| ≥ 50 µg/m³ | **140** | **0.08%** |
| ≥ 100 µg/m³ | **32** | **0.02%** |

Observed maximum 400.3 µg/m³; predictions extend to ~115. So the model is asked to
extrapolate to a range represented by **32 training rows in twelve years**. The
revision states that the compression of extremes is a property of the *sample*, not
only of the algorithm: no learner can reliably fit a region of the response surface
with 32 examples. This also reframes the limitation constructively — it is an
argument for either more monitoring during smoke episodes, or a physically-based
(not purely statistical) treatment of the extreme tail.

## 11. Temporal vs spatial resolution for time-series studies (p. 39)

> *"in purely time series studies we don't need the spatial resolution, just a
> reliable temporal signal across a population"*

Accepted; this sharpens the use-case claim. Revised to distinguish the two
applications explicitly: for **time-series** designs the relevant quantity is a
reliable population-weighted daily series, which the model supports well (temporal
R² is the strong component, and it transfers to Quebec at 0.38 even where spatial
skill does not); for **cohort/spatial-contrast** designs the between-cell ranking
matters, and the negative spatial R² in Quebec plus the modest within-Ontario spatial
component are a genuine caution. The surface should not be presented as equally fit
for both.

## 12. Errors usable in epidemiology (p. 40)

> *"so will you work on errors that can be used in epi? how are they used to advance
> knowledge?"*

The honest answer is that the current intervals are **not** in a form an
epidemiological analysis can consume, and the revision says so rather than implying
otherwise. Per-prediction bands describe each cell-day in isolation; exposure
measurement error in a health model is spatially and temporally correlated, and
propagating it requires either a joint posterior or a bootstrap ensemble of
surfaces. I scope this as future work with a concrete route (an ensemble of surfaces
from the out-of-fold models, preserving correlation structure) rather than claiming
the present bands suffice.

## 13. Comparison with CanOSSEM (p. 40)

> *"I'd like to see you compare your performance to her performance."*

Added as a table. Both products evaluated on the **identical 169,648 matched
cell-days**:

| | RMSE | MAE | R² | bias | within 3 | slope |
|---|---|---|---|---|---|---|
| CanOSSEM | 3.325 | 2.017 | 0.596 | +0.156 | 0.787 | 0.620 |
| This model (Stage 1) | 2.618 | 1.459 | 0.749 | −0.012 | 0.896 | 0.723 |
| **This model (final)** | **2.613** | **1.460** | **0.750** | −0.004 | **0.895** | **0.746** |

A 21% RMSE reduction and +0.15 R². Per the comment at p. 7, this is reported with the
caveat that CanOSSEM is a national product optimised for wildfire surveillance and
this model is Ontario-specific — so the comparison establishes value *for Ontario
exposure assignment*, not general superiority. Block-level and year-level breakdowns
are included so the reader can see whether the advantage holds in smoke years.

---

## 14. Missingness handling, matched ablation, SHAP indexing, and how the corrector is described

Five points, each verified against the code and re-run rather than argued.

### 14.1 Missingness indicators and fill values (point 1)

Two of the three requirements were already met and one was not.

*Already correct.* `transform_corrector` computes `missing = ~np.isfinite(x)` **before**
substituting the fill, and `fit_corrector_fill_values(pool_df)` is called on
`build_corrector_pool(train_df)` — the training portion of the fold only. No zero is
ever converted back to missing; the flag is derived from the value that arrived, so a
genuine measured 0 stays a 0 with `__isna = 0`.

*Not correct.* Every run read `Data_zerofilled/`, a copy of the shards in which an
upstream step had already replaced NaN with 0.0. `missing` was therefore computed on
data with nothing missing left in it: **every `__isna` flag was 0**, and every median
was pulled toward 0 by the substituted zeros. The rule was right; the input defeated it.

The reason this survived review is worth stating, because it also bounds the damage.
Stage 1 is *unaffected*: `load_one_block` applies `nan_to_num` to the design matrix
whether or not the stored frame was pre-filled, so the two roots give a bit-identical
Stage-1 matrix (verified: `max|dX| = 0.0` over 651 x 2,903). Only the corrector reads
the frame directly, so only the corrector was contaminated — 18,471 missing values
across 22 of the 39 raw inputs in a single block, all invisible.

Everything has been re-run against the raw shards. All 8 folds, pooled over the
169,882 out-of-fold cell-days:

| | pre-filled input | raw input | delta |
|---|---|---|---|
| Stage-1 RMSE | 2.617685 | 2.617685 | **0.000000** |
| Stage-1 MAE | 1.459585 | 1.459585 | **0.000000** |
| Stage-1 R^2 | 0.749314 | 0.749314 | **0.000000** |
| Final RMSE | 2.612691 | 2.611589 | -0.001101 |
| Final MAE | 1.460643 | 1.460536 | -0.000107 |
| Final R^2 | 0.750270 | 0.750480 | +0.000211 |

Stage 1 is identical to every printed digit in all 8 folds individually as well, which
is the predicted consequence of `nan_to_num` being applied to the design matrix on both
paths. Per-fold final-prediction deltas scatter in both directions (worst: -0.0071 RMSE
on fold 5, +0.0037 on fold 6) and average to roughly zero.

The correction is a genuine methodological fix that changes no conclusion. Pooled, it
makes the model negligibly *better* (+0.0002 R^2); on individual folds it goes either
way. Both facts are reported as they are: the case for the fix is that the flags and
medians are now what the method says they are, not that the numbers improved.

**What the contamination actually did to the model.** Comparing the two fitted
correctors makes the defect concrete rather than hypothetical:

| | pre-filled input | raw input |
|---|---|---|
| `__isna` features with non-zero split gain | **0 of 40** | 21 of 40 |
| share of corrector split gain carried by them | 0.00% | 0.64% |

Because every flag was constant zero in training, no tree could split on one: the
missingness indicators were present in the design matrix and entirely inert. Seven of
the 40 fill values were also wrong, one of them seriously:

| input | from pre-filled data | correct |
|---|---|---|
| `burned_nearest_km_500km` | **10.0** | **9999.0** |
| `AOD_047` | 0.0 | 0.1787 |
| `AOD_055` | 0.05 | 0.1220 |

The distance rule is `max(finite) + 10`, intended to place "no burned area in range"
beyond every observed distance. With zeros substituted for the missing values the
column's maximum became 0, so the sentinel evaluated to 10 km — telling the model that
a cell with no fire anywhere within 500 km was 10 km from one. The AOD medians were
likewise pulled toward zero by the substituted values.

**What the fix did NOT restore.** The AOD imputation signal was never lost. AOD is
represented in the corrector five times: raw `AOD_055` (32.6% missing) and `AOD_047`
(89.3% missing), the gap-filled `AOD_055_filled`, and the two explicit indicators
`aod_obs_flag` / `aod_imputed_flag`. The last three have no missing values at all, so
zero-filling was a no-op on them -- even the contaminated corrector could always tell
an imputed AOD retrieval from an observed one. What the fix restored is missingness on
the other 22 inputs.

Those indicators are also exactly redundant with each other:

    AOD_055__isna == aod_imputed_flag == 1 - aod_obs_flag      (identical on every row)
    aod_obs_flag + aod_imputed_flag == 1                       (on every row)

TreeSHAP divides credit among perfectly collinear features according to which one a
tree happened to split on, so the split of AOD's attribution BETWEEN these three
indicators is arbitrary. Only the AOD family rollup carries meaning, which is how 14.4
reports it.

This is also why the external test moved more than the Ontario CV did. Quebec carries
far more missingness than the Ontario training domain, so a corrector that can respond
to missingness behaves differently there:

| Quebec, 181,401 cell-days, 50 cells | pre-filled | raw |
|---|---|---|
| Stage-1 R^2 | 0.3637 | 0.3637 |
| final R^2 | 0.3699 | 0.3630 |
| final RMSE | 5.0251 | 5.0525 |
| **spatial R^2** | **-0.1223** | **-0.2747** |

The conclusion the external test supports is unchanged and if anything strengthened:
the model does not transfer across provinces at the cell level. The revised text reports
the raw-input figures.

To prevent recurrence, `fit_corrector_fill_values` now aborts when the training pool
contains zero missing values across all 40 raw inputs — the signature of a pre-filled
input, which is otherwise undetectable downstream because the model still trains
happily.

### 14.2 Matched AOD ablation (point 2)

Agreed: the earlier no-AOD figure came from a separately-configured run and was not a
like-for-like contrast. That reference is withdrawn.

A matched pair already exists and is what the revision cites. `baseline` and
`ablation_no_aod` were produced by the same robustness runner in the same session, and
the recorded provenance shows they differ in exactly one respect:

| | baseline | ablation_no_aod |
|---|---|---|
| predictors | 651 | **639** (the 12 AOD columns removed) |
| held-out cell-days | 169,882 | 169,882 |
| monitored cells | 41 | 41 |
| shard root | same | same |
| preprocessing, hyperparameters, seed, fold plan | same | same |

**Delta R^2 = -0.0013** (pooled out-of-fold, 0.750270 -> 0.748968). Removing satellite
AOD costs about one thousandth of R^2. That is consistent with the 1.69% attribution
share in 14.4 and with the comment's own argument: the reanalysis already assimilates
the satellite signal, so AOD adds little on top of it.

These rows were fitted before the missingness fix in 14.1. They are still the right
numbers to quote, for a reason that is checkable rather than assumed: Stage 1 is
bit-identical across that fix, so every experiment's Stage-1 component is unchanged,
and the corrector moves the pooled figure by +0.00021 (0.750270 -> 0.750480). Because
`delta_r2` is a difference between an experiment and the baseline, both shift together
and the delta moves by less than that -- an order of magnitude below the precision at
which these deltas are reported. The suite was therefore not re-fitted; this paragraph
records that decision and its basis.

### 14.3 SHAP figure indexing (point 3)

Confirmed and fixed. `sv` and `X` carry all 651 columns in canonical order, but the
plotting code selected columns by each feature's position in the *importance-sorted*
table. Rank *k*'s data was therefore drawn under rank *k*'s label — the top feature's
panel was showing `dayofyear`. Columns are now taken from the canonical list, and each
selected column's `mean|SHAP|` is recomputed and asserted equal to the table value for
that name, so a recurrence fails loudly instead of plotting quietly.

Scope: the beeswarm and the dependence panels only. The bar chart, the repaired-share
and family figures, and `shap_feature_importance.csv` were computed per canonical
column index and were unaffected — so no reported *number* was wrong for this reason.

A matching class of defect was found and closed in the 8-fold script: it verified that
the contribution matrix had the same feature *count* as the model, which two different
orderings also satisfy. It now compares the booster's own stored feature names against
the matrix, position by position.

### 14.4 SHAP across all eight folds, reported separately (point 4)

Agreed on all three counts.

*These are shares of attribution, not of skill.* Mean |SHAP| measures how far a
predictor moves this model's output; it is not a share of prediction improvement, and a
family with a small share may still be the only source of some of the signal. The
ablation deltas in 14.2 are the quantity that speaks to improvement, and the two are
now reported as separate things.

*Stage 1 and the corrector are not additive.* The corrector takes `pred_stage1` as one
of its inputs, so part of the correction is attributed to a quantity that is itself a
function of all 651 Stage-1 features. Summing the two tables feature-wise double-counts
through that path. They are reported as two tables and never added; the previously
quoted Stage-1 and corrector percentages must not be combined.

*One fold is not enough.* The reported 1.67% (fold 1) and the previously circulated
1.3% describe single folds — and the 1.3% additionally came from a 400-row preview that
was never refreshed, while being presented as the 21,133-row result. Both are withdrawn.

The replacement is the mean across all 8 folds, on held-out rows only, with each fold's
sampled rows joined back to that fold's saved `holdout_predictions.parquet` on
(cell, date) and the predictions reconciled, so the explained matrix is provably the one
behind the reported metrics. Additivity is asserted per fold against the booster's own
output (worst residual 9.4e-07 against a 1e-04 limit).

| family | Stage 1 | corrector |
|---|---|---|
| MERRA-2 aerosol | 41.14% | — |
| MERRA-2 meteorology | 34.03% | — |
| NARR meteorology | 7.98% | — |
| wildfire smoke | 7.48% | 60.89% |
| land cover / roads | 6.81% | — |
| **satellite AOD** | **1.69%** | **12.13%** |
| seasonality | 0.88% | — |
| `pred_stage1` | — | 26.98% |

(8 folds, 1,000 held-out rows per fold, 8,002 rows explained. Additivity residual
1.4e-06 against a 1e-04 limit; sampled rows reconciled against each fold's saved
`holdout_predictions.parquet` with a worst absolute difference of 0.00e+00 for both
Stage 1 and the final prediction; feature order compared position-by-position against
each booster's own stored names.)

The two columns are separate quantities and are not to be added: `pred_stage1` is itself
a function of all 651 Stage-1 features, so summing across them double-counts. Note also
how little rides on the corrector column — the corrector's entire contribution to skill
is Stage-1 R^2 0.749314 -> final 0.750480, i.e. **+0.0012 R^2**, and it makes MAE
slightly worse (1.459585 -> 1.460536). A 12% share of a stage that adds ~0.001 R^2 is
not evidence that AOD contributes 12% of anything.

*On sample size.* The comment asks that the sample be increased if conclusions change
with it. Three independent runs -- 500, 1,000 and 1,200 rows per fold -- move no family
share by more than 0.31 percentage points, and move AOD by 0.004. Per-feature
Monte-Carlo standard errors are reported in the `sampling_se` column and are an order
of magnitude smaller than the across-fold standard deviations (e.g. 0.0095 vs 0.1026 on
the leading feature), so fold-to-fold disagreement, not sample size, is what limits
precision here. The conclusions are not sample-size sensitive.

### 14.6 Do the missingness indicators mean what they claim?

Establishing that the flags are no longer constant is not the same as establishing
that they are correct, so each was tested against an independent quantity.

**VIIRS.** `src_viirs_dist_nearest_*` is missing exactly when `src_viirs_count_*` is
zero -- identical row sets at all four radii (83,910 / 50,139 / 29,498 / 19,466). So
the indicator means "no fire detected in this radius", a real state rather than a data
defect, and the zeros in the count columns are genuine measurements that correctly stay
zero.

**AOD.** Over all 169,882 cell-days:

| where the flag claims | check | result |
|---|---|---|
| observed (89,891 rows) | raw `AOD_055` present | 89,891 / 89,891 |
| observed | `AOD_055_filled` == raw exactly | max diff 0.000e+00 |
| imputed (79,991 rows) | raw `AOD_055` missing | 79,991 / 79,991 |
| imputed | filled value present | 79,991 / 79,991, **79,737 distinct** |

The imputed rows carry a genuine per-row estimate, not one constant substituted
everywhere. No mislabelling in either direction.

**One consequence worth stating in the Discussion.** Imputed AOD has a higher mean
(0.1761) than observed AOD (0.1399), about 26% higher. That is expected -- MODIS
retrievals fail preferentially under cloud, snow and heavy aerosol loading -- but it
means `aod_imputed_flag` is not a neutral nuisance indicator: it is informative about
conditions. Part of what the AOD block contributes is therefore the PATTERN OF
RETRIEVAL FAILURE rather than the aerosol measurement itself, which is a further
reason not to read the AOD attribution share as "what the satellite measurement adds".

**Two defects found while checking, neither material.**

1. `aod_obs_flag` is exactly "`AOD_055` present", not "either band observed". It
   disagrees with the union definition on **2 rows out of 169,882** (0.0012%), because
   `AOD_047` is almost a strict subset of `AOD_055`.

   **Resolution: the name is corrected, the variable is not.** The thesis, codebook and
   any figure legend describe it as an *AOD_055 observation flag*. Redefining it as the
   union would change two rows and invalidate every fitted corrector, the SHAP tables
   and the three-family comparison, to buy a better name -- and could not move a
   reported figure. Nothing is lost by leaving it: whether `AOD_047` was retrieved is
   already available to the corrector as `AOD_047__isna`, a separate live indicator
   (85.8% missing). The definition and this reasoning are recorded at the point of
   definition in `build_corrector_raw_cols()` in all three fold scripts, so the next
   reader meets it where the columns are declared rather than only here.
2. Four burned-area inputs are degenerate in Ontario: `burned_nearest_km_500km` and
   `burned_weighted_frac_500km` are missing on every row, while `burned_frac_500km`
   (constant 0) and `burned_weighted_frac_1000km` (constant 1) vary only through their
   missingness. No burned area ever falls within 500 km of an Ontario monitor in this
   record. This is consistent with `ablation_no_burned` costing nothing
   (delta R^2 = +0.0016) and should be stated rather than left implicit.

Also note that `aod_obs_flag`, `aod_imputed_flag` and `AOD_055__isna` are the same bit
(see 14.1), so three of the corrector's 80 inputs are perfectly collinear.

### 14.5 How the corrector is described (point 5)

Agreed, and the code is more decisive than the comment assumed.

*"Trained only on smoke or high-error days" is false.* The pool rule is
`smoke/fire signal OR pm25 >= 15 OR |training residual| >= 3`. Its first clause tests
whether any of 34 fire/smoke columns is positive, and two of those —
`src_hms_density_weight_wmean_500km` and `_1000km` — are positive on **every row**,
because over a 500-1000 km neighbourhood there is always some HMS smoke contribution.
The selector is therefore vacuous: `pool_rows = 148,785` of `training_rows = 148,785`,
`pool_fraction = 1.0`. The corrector trains on **all** training rows. It is a plain
second-stage residual model, not a smoke or extreme-value specialist, and the thesis
will describe it that way.

*"Out-of-fold residuals" is also wrong.* The target is `resid_stage1` computed on
`train_df` from the Stage-1 model fitted on those same rows — in-sample **training
residuals**. (A genuine out-of-fold variant is run separately as the `oof_corrector`
robustness experiment, where it is labelled as such.) The wording is corrected
wherever it appears.

---

## Still outstanding

0. **Two SHAP figures must be regenerated.** `fig4_beeswarm.png` and
   `fig5_dependence.png` under `GROUP_04/shap/` were produced by the buggy indexing in
   14.3 and are mislabelled: the five most prominent panels show `dayofyear`,
   `air_sfc`, `hpbl`, `windspeed_10m` and `vis` under MERRA-2 aerosol labels. The code
   is fixed; the figures have not yet been re-plotted. Do not reuse the existing PNGs.
1. **MERRA-2 as a standalone benchmark** (comments 9, 11) — not yet run; the most
   valuable remaining analysis.
2. **Ablation ΔR² values** (comments 7, 8, 12) — experiments running.
3. Whether to report the alternative fold assignments and leave-one-cell-out results
   in the main text or an appendix, once available.
