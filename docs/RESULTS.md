# Results

NHANES 2005–2006 (cycle D), mortality follow-up through 2019-12-31.
All models adjust for chronological age and sex. All scores are **out-of-fold**
(5-fold cross-fitted). Regenerate with `python scripts/run_pipeline.py`.

---

## Sample

| | n |
|---|---|
| Analytic sample (age 20–84, mortality-eligible) | **4,623** |
| Deaths | **810** |
| Max follow-up | **15.0 years** |
| With blood | 4,329 |
| With wearable | 2,990 |
| **With both** (combiner training set) | **2,861** |
| Validation common sample / deaths | **2,956 / 546** |

---

## 1. Does each modality earn its seat?

Both do. Each gap predicts death independently of chronological age and sex.

| Score | gap SD | **HR per year** | HR per SD | 95% CI (per SD) | C-index | gain over age+sex | AUC (gap alone) |
|---|---|---|---|---|---|---|---|
| blood | 4.19 y | 1.086 | 1.412 | 1.287 – 1.550 | 0.8530 | +0.0077 | 0.616 |
| wearable | 4.86 y | 1.091 | 1.526 | 1.376 – 1.692 | 0.8542 | +0.0089 | 0.651 |
| **combined** | 3.62 y | **1.153** | **1.677** | **1.514 – 1.857** | **0.8587** | **+0.0134** | **0.665** |

All p < 1e-12. Age+sex alone gives C = 0.8453.

**HR per year is the field's reporting convention** for age-acceleration
measures (Chen 2016, Liu 2018, Lu 2019); HR per SD is included because it is
comparable across modalities whose gaps have different spreads.

### Benchmarking against published clocks

| Measure | HR/year | HR/SD | C-index gain over age+sex |
|---|---|---|---|
| PhenoAge, NHANES 2005–06 + 2019 LMF | 1.023 | 1.230 | +0.0097 |
| Chen 2016 EEAA (13-cohort meta) | 1.040 | ~1.28 | — |
| GrimAge (Lu 2019), strongest published | 1.10 | — | — |
| **This pipeline, combined** | **1.153** | **1.677** | **+0.0134** |

The combined gap's HR of 1.153 per year is at or slightly above the strongest
published single clock, and its C-index increment of +0.0134 exceeds PhenoAge's
+0.0097 on this exact cohort and linkage. That is the expected direction for a
multi-modal score against a single-modality one, and it lands in a plausible
range rather than an implausibly good one — a sanity check that the pipeline is
measuring what it claims to, on the scale it claims to.

## 2. Does fusion beat the best single modality?

**Yes, on every metric.**

| | best single | combined | gain |
|---|---|---|---|
| HR per year | 1.091 (wearable) | **1.153** | +0.062 |
| HR per SD | 1.526 (wearable) | **1.677** | +0.151 |
| C-index | 0.8542 (wearable) | **0.8587** | **+0.0045** |
| C-index gain over age+sex | +0.0089 | **+0.0134** | **+51% more** |
| AUC, gap alone | 0.651 (wearable) | **0.665** | +0.014 |

The absolute C-index numbers look close together because **chronological age
alone already gives C = 0.8453** — age is overwhelmingly the strongest mortality
predictor and everything else competes for the remaining margin. The meaningful
column is the *increment over age+sex*: the combined gap adds **51% more
discrimination** than the best single modality does.

For scale, **a +0.01 C-index increment over age+sex is the published-consistent
result for a bioage measure, not a shortfall** — PhenoAge achieves +0.0097 on
this same cohort. The combined score's +0.0134 is above that.

### One tuning decision that mattered
The combiner was initially fitted with `penalizer=0.1` (mirroring GrimAge's
elastic net). That was wrong for this problem: GrimAge was regularising 12
protein surrogates, whereas this combiner has **two gaps and 546 events**. At
0.1 the penalty collapsed the blood weight from 0.46 to 0.06 and **cost 0.008 of
C-index** (0.8587 → 0.8502) — it was discarding a real predictor, not
controlling overfitting. Reduced to 0.01, which leaves the fit stable if two
modalities are near-collinear without throwing one away.

**Final combiner weights: blood 0.461, wearable 0.539** — near-balanced, which
is itself consistent with the low redundancy in §3.

## 3. Redundancy — the interesting result

**corr(blood gap, wearable gap) = 0.254** (n = 2,956)

Low-to-moderate. The two arms are **not** noisy proxies for one underlying
signal — they carry substantially complementary information. Had this come back
at 0.8, fusion would be cosmetic and the honest recommendation would be to ship
one arm. At 0.25, roughly 94% of the variance in one gap is unexplained by the
other, which is what makes a per-modality breakdown worth showing and what
gives the combiner something real to do.

## 4. Where the modalities disagree

These are the cases a single-source product cannot see. Full table:
`outputs/tables/disagreement_cases.csv`.

| SEQN | age | blood gap | wearable gap | died | reading |
|---|---|---|---|---|---|
| 36216 | 75 | −15.1y | +5.7y | yes (1.3y) | good labs, failing physically |
| 40827 | 29 | **−10.9y** | **+10.3y** | no | clean bloodwork, sedentary body |
| 36352 | 44 | +11.1y | −8.6y | no | fit, but metabolically strained |
| 35206 | 68 | +7.7y | −12.1y | yes (4.9y) | genuinely fit, ailing metabolics |

Example CLI output for SEQN 40827:

```
  Chronological age : 29.0   (male)
  Combined bioage   : 29.6   (0.6y older than chronological)

  PER-MODALITY BREAKDOWN
    modality           gap     younger  |  older       weight
    blood           -10.9y      =========               0.461
    wearable        +10.3y              #########       0.539

  DRIVER: wearable (+5.6y of the +0.6y combined gap)
  DIVERGENCE: wearable is aging 21.2y faster than blood
```

This is the clearest argument in the whole project for never shipping the scalar
alone. The combined number is **+0.6 years — indistinguishable from normal** —
and it is averaging over two systems that are **21 years apart**. Everything
actionable about this person is in the breakdown and nothing is in the total.

---

## 5. What the model learned

### Blood — top features by final weight
| feature | mortality z | precision | final weight |
|---|---|---|---|
| RDW (red cell distribution width) | 6.36 | 0.032 | **0.304** |
| Lymphocyte % | 2.00 | 0.058 | **0.176** |
| Albumin | 4.32 | 0.018 | **0.117** |
| Creatinine | 1.20 | 0.045 | 0.083 |
| C-reactive protein | 2.41 | 0.020 | 0.074 |

RDW, albumin, CRP and lymphocyte percent are exactly the markers the PhenoAge
biomarker set is built from. They were **recovered from the mortality data**,
not supplied — an independent check that the outcome weighting is finding real
signal rather than fitting noise.

### Wearable — top features
| feature | mortality z | final weight |
|---|---|---|
| Mean counts per wear-minute | 6.66 | **0.270** |
| MVPA minutes/day | 7.59 | **0.234** |
| M10 (most-active 10h) | 6.91 | **0.225** |
| L5 (least-active 5h) | 3.81 | 0.145 |

Activity **volume** dominates; circadian-rhythm features contribute little once
volume is accounted for.

### The sign safeguard firing
Two wearable features have gaps that run **backwards** against mortality:
`relative_amplitude` (z = **−5.73**) and `intradaily_variability` (z = −2.90).
A higher implied age predicting *lower* mortality is a sign of an inverted or
noisy curve, not a protective effect, so both are clipped to **zero weight**
rather than allowed to subtract. (Mechanism: the waist-worn device is removed
for sleep, so "L5" is daytime rest — older, more sedentary participants show a
*larger* daytime rest/activity contrast, inverting the expected direction.)

### Reliability and scale
| modality | split-half reliability | 1 raw unit = age-equivalent years | final gap SD |
|---|---|---|---|
| blood | 0.747 | 0.507 | **4.33 y** |
| wearable | 0.954 | 0.319 | **4.86 y** |

The wearable arm is *more internally consistent* (0.95) but on a *more
compressed* age scale (0.32) — its ten features largely measure one thing
(activity volume), so they agree with each other strongly, while activity's
between-person variance at a fixed age far exceeds its between-age variance.

---

## 6. Curve screening

| | count |
|---|---|
| Curves fitted (feature × sex) | 64 |
| Fully monotonic over 20–84 | 26 |
| Restricted to a monotonic segment | 37 |
| Excluded entirely | **1** (`total_protein`, female) |

The fitted curves reproduce known physiology without being told to: creatinine,
BUN, MCV, RDW, HbA1c, glucose and systolic BP rise; albumin, hemoglobin and
lymphocyte % fall; uric acid and alkaline phosphatase rise sharply in women
after ~50 and stay flat in men; waist circumference peaks near 65 and declines.

**Total cholesterol runs in opposite directions by sex** — decreasing in men
from 46y, increasing in women to 69y. A pooled curve there would represent
nobody, and inverting it would be meaningless. This is the single clearest
justification for sex-stratifying every curve.

See `outputs/figures/curves_blood.png` and `curves_wearable.png` (usable
monotonic span drawn as a thick band; excluded panels tinted).

---

## 6b. Methylation arm (separate cohort, GSE40279, n=656)

Five published clocks applied via Biolearn, ages 19–101:

| clock | generation | r with chronological age | MAE | gap SD | residual gap SD |
|---|---|---|---|---|---|
| Horvathv1 | 1st | 0.918 | 4.77 | 5.88 | 5.06 |
| Hannum | 1st | 0.946 | 5.51 | 4.85 | 4.25 |
| GrimAgeV1 | 2nd | 0.941 | 4.41 | 5.44 | **3.93** |
| GrimAgeV2 | 2nd | 0.917 | 6.19 | 6.37 | 4.44 |
| PhenoAge | 2nd | 0.852 | 9.04 | 7.75 | **6.96** |

**This table does not support the first- vs second-generation argument, and it
is reported that way.** The rationale for preferring second-generation clocks is
that first-generation ones were trained on chronological age and leave little
residual to combine. Here Hannum (1st gen) leaves residual SD 4.25 while
GrimAgeV1 (2nd gen) leaves 3.93 — the opposite ordering.

The reason the test cannot settle it: **residual spread conflates signal with
noise.** PhenoAge's 6.96 is the widest of the five and cannot be read as "the
most information". Separating signal from noise requires regressing the residual
on an outcome, and **GSE40279 carries chronological age only — no vital
status**. The clock choice (GrimAge primary, PhenoAge secondary) therefore rests
on the published mortality evidence, not on this cohort.

Two implementation notes worth recording:
- GrimAge's `predict()` returns a **wide** DataFrame whose first column is
  `DNAmADM`, a plasma-protein sub-clock — *not* an age. Taking column 0 (the
  natural thing to write) yields a plausible-looking number that is not a
  biological age. The age estimate is `DNAmGrimAge`.
- Biolearn codes sex **0=female, 1=male**. Passing NHANES `RIAGENDR` (1=Male,
  2=Female) raw scores **every female as male with no exception raised**.

## 7. Caveats

- The combined score's C-index advantage over the best single arm is **+0.0045**
  (a 51% larger increment over age+sex). Real and consistent, but the absolute
  numbers are close because age dominates. Do not oversell it.
- The wearable arm is a **weak proxy** for a consumer wearable — no HRV, sleep
  or resting HR. See METHODOLOGY §10.
- The methylation arm runs on a **different cohort with no mortality linkage**,
  so it carries no fitted combiner weight and is excluded from the combined
  score. See METHODOLOGY §9.
- Survey weights are applied to the reference curves but **not** inside the Cox
  models; population-generalisable hazard estimates are not claimed.
- 810 deaths over 15 years is adequate but not large. The 95% CIs above are the
  honest measure of precision.
