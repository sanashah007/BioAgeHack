# Methodology and decision log

Every weighting, stratification and exclusion decision, with its rationale.
These are the choices that get questioned; each one is recorded here and
commented at its point of use in the code.

---

## 0. The governing constraint

Biological age has no ground truth label. The only directly measurable label is
chronological age, and that creates the central trap: **a model that predicts
chronological age perfectly is useless**, because a perfect predictor has zero
residual and the residual is the entire product.

Two rules follow, and they are enforced structurally rather than by convention:

1. **Never combine raw predicted ages.** Averaging several good chronological-age
   predictors just re-derives the birthdate. Every combination step in
   `bioage/scoring.py` operates on *gaps*.
2. **Never fit a combination step against chronological age.** Both combination
   stages are fitted against linked mortality. `scoring.fit_combiner` raises if
   the mortality outcome is absent rather than silently substituting an age
   target.

---

## 1. Cohort

**NHANES cycle D (2005–2006) for all three NHANES-derived arms.**

| Reason | Detail |
|---|---|
| Follow-up length | The Public-Use Linked Mortality File tracks deaths to 2019-12-31. An older cycle buys more follow-up and more events — the thing the combiner is actually fitted against. Cycle D yields **15.0 years** max follow-up and **810 deaths** in the analytic sample. |
| Single-cycle join | Cycle D carries the accelerometer sub-study *and* a full biochemistry panel, so blood, wearable and mortality all key on the same `SEQN` with no cross-cycle weight rescaling. |
| No pooling | Pooling cycles would require dividing weights by the number of cycles per NHANES analytic guidelines. Not needed at this sample size, so not done. |

**Analytic sample:** 4,623 participants aged 20–84, mortality-eligible, with
810 deaths. 4,329 have blood, 2,990 have wearable, **2,861 have both**.

### Age top-code
`RIDAGEYR` is top-coded at **85** in cycle D — 170 participants are recorded as
exactly 85 regardless of true age. That bin is a mixture of 85- and
100-year-olds, which flattens any real trend and corrupts inversion at the upper
tail. The analysis window is therefore **20–84**, strictly below the top-code.
Verified empirically, not assumed (`nhanes.load_demographics` warns on mismatch).

---

## 2. Survey weights

NHANES oversamples older adults, several race/ethnicity groups and low-income
households **by design**. Raw sample means are biased toward whoever was
oversampled, so every reference curve is survey-weighted.

The correct weight is determined by the **narrowest sub-sample a variable was
measured in**, not by the file it ships in:

| Weight | Applies to | Verified how |
|---|---|---|
| `WTMEC2YR` | BIOPRO, CBC, CRP, GHB, TCHOL, HDL, BMX, BPX | no sub-sample weight column present in file |
| `WTSAF2YR` | **GLU_D, TRIGLY_D** | `WTSAF2YR` is a column *in those two files only* — confirmed empirically |
| `WTINT2YR` | interview-only (questionnaires, used for exclusions) | — |

Using `WTMEC2YR` on the fasting labs would weight them as though the whole exam
sample had been measured, when only the morning fasting sub-sample (n=3,352 vs
9,950) was.

**Simplification (`# SCOPE`)**: `SDMVSTRA`/`SDMVPSU` are carried through the
analytic table but the curve fits use weights only. Strata and PSU matter for
*variance estimation*; for the point-estimate curves the weights carry the
bias correction. Confidence bands on the curves themselves would need the full
design, and are not claimed.

---

## 3. Sex stratification

Every curve is fitted separately by sex. This is not a formality — the data
shows the trajectories genuinely diverge:

| Marker | Male | Female |
|---|---|---|
| **Total cholesterol** | **decreasing**, 46–84y | **increasing**, 20–69.5y |
| Albumin | monotone decreasing, full range | decreasing only after 49.5y |
| GGT | decreasing, 40.5–84y | increasing, full range |
| Hemoglobin | monotone decreasing | decreasing only after 56y |

Total cholesterol runs in **opposite directions by sex** across most of adult
life (men peak in mid-life and decline; women rise through menopause). A pooled
curve there represents nobody, and inverting it would be meaningless.

---

## 4. Reference population

**Default: healthy reference** (`--population-reference` switches to population
average).

A population-average curve bakes prevalent disease into the baseline, so a
genuinely abnormal marker is scored against a sick population and looks *less*
abnormal than it is. A healthy reference matches clinical reference-range
practice and published bioage work. The cost, stated plainly: the curve no
longer describes "typical" adults.

Exclusions are **per-feature, not global**. Excluding every diagnosed condition
from every curve would shrink the reference to an unrepresentative
super-healthy remnant, and the exclusion that matters for glucose (diabetes) is
not the one that matters for creatinine (kidney disease).

| Rule | Definition | n flagged |
|---|---|---|
| `diabetes` | `DIQ010==1` (told diabetic) or `DIQ050==1` (on insulin) | 523 |
| `kidney_disease` | `KIQ022==1` | 136 |
| `liver_disease` | `MCQ160L==1` | 163 |
| `bp_meds` | `BPQ040A==1` | 1,328 |
| `lipid_meds` | `BPQ100D==1` | 704 |
| `cvd` | `MCQ160B/C/E/F==1` (CHF, CHD, MI, stroke) | 515 |
| `cancer` | `MCQ220==1` | 414 |
| `current_smoker` | `SMQ040 in {1,2}` | 1,089 |
| `acute_inflammation` | `LBXCRP > 1.0 mg/dL` (= 10 mg/L) | 708 |
| `anemia` | WHO: Hgb < 12 (F) / < 13 (M) g/dL | 1,177 |

Applied per feature via `Feature.exclude_if` in `config.py`.

**Coding convention:** NHANES uses 1=Yes, 2=No, 7=Refused, 9=Don't know,
NaN=not asked. Refused/DK/not-asked are treated as **not excluded**. Treating
unknown as diseased would drop every participant a question was not asked of
(most `MCQ160*` items are asked only of those 20+), silently gutting the
reference population.

### Unavailable exclusions (`# SCOPE`)
Two rules referenced by the feature registry have no source in the acquired
files. They are recorded as never-triggering and **logged as warnings** rather
than dropped silently:
- `heavy_alcohol` — needs `ALQ_D`; affects the GGT curve.
- `gout_meds` — needs prescription data (`RXQ_RX_D`) to find urate-lowering
  therapy; affects the uric acid curve.

---

## 5. Curve fitting

**Survey-weighted local linear regression** (`curves.weighted_loess`).
`statsmodels`' lowess does not accept observation weights, so the local linear
fit is done directly, with survey weights entering as multiplicative case
weights alongside a tricube kernel.

The bandwidth is **nearest-neighbour** (`LOESS_FRAC = 0.45`): at each grid point
the bandwidth covers that fraction of observations, so it widens automatically
where data thins — the upper age tail, exactly where a fixed bandwidth goes
jagged. This is the alternative to pooling adjacent cycles, and avoids the
weight-division that pooling would require.

`MIN_N_PER_SEX = 200` — below that the curve is refused, not fitted badly.

---

## 6. Monotonicity screening — mandatory before inversion

Nearest-match inversion assumes the curve is one-to-one. Against a curve that
rises then falls, a single marker value matches **two ages**, and the inverter
returns whichever the search hits first. The result is not noisy, it is
*arbitrary*, and it looks perfectly reasonable in the output.

### The metric matters
The obvious implementation — count the sign of each grid step — is wrong. On a
0.5-year grid, a genuinely smooth curve with a flat stretch flips sign
constantly on numerically negligible wiggle. Measured that way, 40% of curves
"reversed" and real markers (WBC, hemoglobin, lymphocyte %) were excluded for
noise.

The screen therefore uses a **magnitude-weighted** measure: the share of the
curve's *total variation* that travels against the dominant direction,
equivalently `1 − |net change| / total variation`. Scale-free, and insensitive
to plateaus. Tolerance `MONOTONIC_TOL = 0.05`.

Non-monotonic curves are not discarded outright — the screen finds the **widest
sub-interval** monotone to within tolerance (all-pairs scan, ~8k evaluations)
and restricts inversion to it, provided the span is ≥ `MIN_MONOTONIC_SPAN = 25`
years.

### Outcome
| | count |
|---|---|
| Curves fitted (feature × sex) | **64** |
| Fully monotonic over 20–84 | **26** |
| Restricted to a monotonic segment | **37** |
| **Excluded entirely** | **1** — `total_protein` (female) |

Full per-curve log with spans and reasons: `outputs/tables/curve_screening.csv`.
Every curve is plotted in `outputs/figures/curves_*.png`, with the usable span
shown as a thick band and excluded panels tinted — so the screen is auditable,
not just a number in a log.

---

## 7. Inversion and the four corrections

Raw curve inversion (`curves.invert`) selects the age minimising
`|curve(a) − value| / sd(a)` over the usable segment. The variance
normalisation is what makes the match meaningful: marker spread is strongly
heteroscedastic, so a raw nearest-match is drawn toward whichever end of the
curve is steepest rather than toward the age the value resembles.

Four corrections are then applied, in order. Each fixes a specific, observed
failure:

| # | Correction | Problem it fixes | Effect |
|---|---|---|---|
| 1 | **Inverse-variance (precision) weighting** — `curves.curve_precision` | A marker that barely changes with age but varies a lot *within* age cannot locate anyone. Its implied age swings to whichever segment endpoint the person falls past. | Blood gap SD **18.7 → 11.5y** |
| 2 | **Outcome weighting** — `scoring.feature_mortality_weights` | A feature that barely predicts death should not get equal say. Univariate Cox on `gap + age + sex`; weight = z-statistic, clipped at 0. | Top blood features: RDW, albumin, CRP, lymphocyte % — the classic PhenoAge markers, recovered from the data |
| 3 | **Reliability shrinkage** — `scoring.shrink_to_reliability` | An unshrunk gap is reported at full spread as if measured exactly. Split-half + Spearman-Brown gives blood ρ=**0.75**, wearable ρ=**0.95**. | Blood gap SD **11.5 → 8.6y** |
| 4 | **Age-equivalent calibration** — `scoring.calibrate_to_age_scale` | Each modality's implied age is on an arbitrary scale. Raw inversion reported a fit 68-year-old as wearable-age 30. | Blood SD **4.3y**, wearable SD **4.9y** |

**Corrections 3 and 4 are positive linear rescalings, so they change no
validation metric** — HR per SD, C-index and AUC are all invariant. They change
only what the reported number *means*.

Correction 4 is PhenoAge's own construction: fit `Cox(death) ~ b_gap·gap +
b_age·age + sex`; the ratio `b_gap / b_age` is how many years of chronological
age carry the same hazard as one unit of gap. A gap of "+5 years" then means
"the mortality risk of someone five years older" — the only definition
comparable across arms. Measured: 1 raw blood unit = **0.507** age-equivalent
years; 1 raw wearable unit = **0.319**.

### Regression-to-the-mean (essential, and easy to miss)
Any predictor imperfectly correlated with age regresses toward the mean, so the
raw gap is systematically **positive in the young and negative in the old**.
Measured here: `corr(gap, age)` = **−0.76** (blood), **−0.52** (wearable). Left
uncorrected the gap is partly a proxy for age itself — and since age is the
strongest mortality predictor there is, validation would look impressive while
measuring almost nothing.

Two defences, both applied:
1. `scoring.deattenuate` — regress gap on age, keep the residual (the standard
   "age acceleration residual"). Post-correction `corr(gap, age)` = **0.000**.
2. Chronological age is a covariate in **every** outcome model in the pipeline.

**These are not two independent corrections, and it matters which one is doing
the work.** With chronological age already in the Cox model, residualising the
gap is an *exact algebraic no-op* — the residual is a linear function of the gap
and age, both already in the model, so the fit is identical. Observed directly
in this pipeline: raw and deattenuated gaps returned **identical p-values**
(1.44e-23 blood, 1.14e-16 wearable) and identical concordance, differing only in
the HR because the SD changed.

So the division of labour is:
- **Covariate adjustment** is what makes the *outcome inference* honest. It is
  load-bearing and non-negotiable.
- **Residualisation** is what makes the *reported number* honest — the gap
  shown to a person, the tertiles in the Kaplan-Meier plot, the correlation in
  the redundancy table. Without it those descriptive outputs would be partly
  restating chronological age.

The failure mode is dropping age from the outcome model and trusting
residualisation alone to have handled it. Without age as a covariate the raw
gap's apparent performance is inflated because it smuggles chronological age in
through the back door.

---

## 8. Combiner

Fitted against **NHANES linked mortality**, never chronological age. Elastic-net
penalised Cox (`penalizer=0.1, l1_ratio=0.5`) on time-to-death, with age and sex
as covariates. Trained on n=**2,956** with **546 deaths** — the blood ∩ wearable
overlap.

Cox is preferred over logistic because it uses the follow-up time: a death at 6
months and one at 14 years are not the same event, and participants examined in
different years carry different censoring under a fixed 2019 cut-off.
`--combiner logistic` implements the elastic-net logistic alternative.

**Cross-fitting.** Both stages learn weights from the same outcome used for
validation. Scoring a participant with weights fitted on data including them
leaks the outcome and inflates every metric. All reported scores are
**out-of-fold** (5-fold stratified).

**Missing modalities — the norm, not the exception.** Missing arms are
**dropped and the remaining weights renormalised**, never imputed as zero.
Imputing zero would assert "this person is exactly average on the channel we did
not measure", pulling every partially-measured person toward the mean and making
a single-modality reading falsely reassuring. Verified in the CLI with `--drop`.

---

## 9. Methylation arm — two hard scope limits

**Clocks**: GrimAgeV2 primary, PhenoAge secondary, via Biolearn's published
coefficients. Horvath v1 and Hannum are computed **as a contrast only**, to
quantify (not assert) the first- vs second-generation argument:
first-generation clocks were trained to predict chronological age and are very
good at it, which leaves little residual to combine.
See `outputs/tables/methylation_clock_comparison.csv`.

> **`# SCOPE`: different cohort.** NHANES has never collected DNA methylation in
> any public-use cycle. There is no `SEQN` to join on. This arm runs on GEO
> **GSE40279** and is reported as a *demonstration arm*. **No cross-cohort join
> is fabricated anywhere in this codebase.**

> **`# SCOPE`: no mortality linkage in GSE40279.** The cohort carries
> chronological age only — no vital status, no follow-up. Per the project rule
> that a missing mortality target must be **flagged, not replaced**, this arm
> receives **no fitted combiner weight** and is excluded from the combined score.
> It appears in the per-modality breakdown labelled as such.

---

## 10. Wearable arm — honest limitation

> **`# SCOPE`:** The 2005–2006 ActiGraph AM-7164 is a **uniaxial, waist-worn,
> counts-only** device worn during waking hours. It yields **no HRV, no resting
> heart rate, no sleep staging, no SpO2** — precisely the channels carrying most
> of the aging signal in a modern consumer wearable. What survives is activity
> volume, intensity distribution and circadian fragmentation. Treat this arm as
> a **lower bound** on what a real wearable arm would contribute.

A second, subtler limit: the device is removed for sleep, so the "rest" period
is largely non-wear. L5 and interdaily stability describe **daytime** rest, not
sleep.

### Processing
75M minute-level rows / 7,455 participants (3.0 GB), streamed in chunks with
participants buffered across chunk boundaries so nobody is split and half-analysed.

- **Device QC**: `PAXCAL==1` and `PAXSTAT==1` across the whole record — NCHS
  flags the entire wear period, so a failing minute condemns the participant.
  **592 failed.**
- **Non-wear**: Troiano 2008 — ≥60 consecutive zero-count minutes, tolerating ≤2
  interrupting minutes below 100 counts.
- **Valid day** ≥600 wear-minutes; **valid participant** ≥4 valid days →
  **4,991 usable**.
- **Cut-points**: Troiano 2008 adult — sedentary <100, moderate ≥2020,
  vigorous ≥5999 cpm.

### Two bugs worth recording
1. **`pandas.read_sas` decodes SAS transport zeros as the denormal `5.397e-79`,
   not `0.0`.** Every non-wear and sedentary rule keys off "counts == 0". Left
   unnormalised, non-wear detection silently disables entirely: wear time
   becomes 1440 min/day and the features become garbage that still looks
   plausible. `wearable._denorm` is load-bearing.
2. **M10/L5 must be computed over wear minutes only.** Treating non-wear as zero
   drove L5 to exactly 0 for most participants and pinned relative amplitude at
   1.0 cohort-wide. Windows now require ≥80% wear or are rejected. Fixed values:
   L5 median 0 → **168**; RA mean 1.00 → **0.42**.
   Interdaily stability also exceeded its [0,1] bound under the textbook formula
   with gappy data, and is computed in variance-ratio form instead.

---

## 11. Validation

All models adjust for chronological age and sex. Metrics are computed on the
**common sample** (n=2,956, 546 deaths) where every score exists — a combined
score evaluated on a different, better-measured subset would beat the singles
for reasons having nothing to do with fusion.

Results: see `docs/RESULTS.md` and `outputs/tables/validation.csv`.

---

## 12. Everything deliberately not done

- Strata/PSU variance estimation (weights only for point estimates) — §2.
- `heavy_alcohol` and `gout_meds` exclusions — no data source — §4.
- Cycle pooling — unnecessary at this sample size — §1.
- Methylation combiner weight — no mortality linkage in GSE40279 — §9.
- HRV / sleep / resting HR wearable features — not collected in 2005–2006 — §10.
- Survey weights are **not** applied inside the Cox models. NCHS guidance for
  design-based mortality analysis would use them with the full design;
  here the mortality models are used to *rank and weight* rather than to
  estimate population hazards, and all reported metrics are internal
  comparisons on one sample. Population-generalisable hazard estimates are
  not claimed.
