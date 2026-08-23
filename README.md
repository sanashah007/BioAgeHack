# Multi-Modal Biological Age Pipeline

Biological age from three independent channels — blood chemistry, wearable
activity, and DNA methylation — fused against **linked mortality**, with a
per-modality breakdown as the primary output.

```
Blood (NHANES)      → weighted, sex-stratified age curves per analyte
                    → invert curve → per-analyte implied age
                    → outcome-weighted combine → blood_bioage

Wearable (NHANES)   → same inversion on engineered accelerometry features
                    → outcome-weighted combine → wearable_bioage

Methylation (GEO)   → EXISTING published clock coefficients via Biolearn
                    → methylation_bioage

All three           → gap_m = modality_bioage − chronological_age
                    → combiner fit against NHANES linked mortality
                      (NOT chronological age)
                    → combined bioage + per-modality gap breakdown
```

## The one rule this codebase is built around

**Never combine raw predicted ages, and never fit a combination step against
chronological age.** A model that predicts chronological age perfectly is
useless — a perfect predictor has zero residual, and the residual is the entire
product. Averaging three good age predictors just re-derives the birthdate.

Enforced structurally, not by convention: every combination operates on gaps,
and `scoring.fit_combiner` **raises** if the mortality outcome is missing rather
than silently substituting an age target.

## Headline results

NHANES 2005–2006, 4,623 adults, 810 deaths, 15 years of follow-up.
All adjusted for chronological age and sex; all scores out-of-fold.

| Score | HR per year of gap | HR per SD | C-index | gain over age+sex | AUC (gap alone) |
|---|---|---|---|---|---|
| blood | 1.086 | 1.412 | 0.8530 | +0.0077 | 0.616 |
| wearable | 1.091 | 1.526 | 0.8542 | +0.0089 | 0.651 |
| **combined** | **1.153** | **1.677** | **0.8587** | **+0.0134** | **0.665** |

Age+sex alone gives C = 0.8453, so the meaningful column is the *increment*: the
combined gap adds **51% more discrimination** than the best single modality. For
scale, PhenoAge achieves +0.0097 on this same cohort and linkage.

**corr(blood gap, wearable gap) = 0.251** — the arms are complementary, not
redundant. That, more than the C-index margin, is what justifies multi-modality.

Full numbers and caveats: **[docs/RESULTS.md](docs/RESULTS.md)**.
Every design decision: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# full pipeline (first run downloads ~3.5 GB: NHANES + accelerometer + LMF)
./.venv/bin/python scripts/run_pipeline.py

# skip the 2.5 GB GEO methylation download
./.venv/bin/python scripts/run_pipeline.py --skip-methylation

# population-average reference instead of healthy reference
./.venv/bin/python scripts/run_pipeline.py --population-reference
```

## Output interface

```bash
./.venv/bin/python scripts/bioage_report.py --demo
```

```
==================================================================
  BIOLOGICAL AGE REPORT   —   NHANES SEQN 40827
==================================================================
  Chronological age : 29.0   (male)
  Combined bioage   : 29.6   (0.6y older than chronological)

  PER-MODALITY BREAKDOWN
    modality           gap     younger  |  older       weight
    blood           -10.9y      =========               0.461
    wearable        +10.3y              #########       0.539
    methylation          —        not supplied

  DRIVER: wearable (+5.6y of the +0.6y combined gap)
  DIVERGENCE: wearable is aging 21.2y faster than blood — the modalities disagree.

  Modalities used   : blood, wearable
==================================================================
```

This case is the whole argument for the breakdown. The combined number is
**+0.6 years — indistinguishable from normal** — and it is hiding two systems
that are *21 years apart*: excellent bloodwork alongside the activity profile of
someone far older. A scalar "your bioage is 29.6" tells this person nothing.
"Your wearable signal is aging much faster than your blood chemistry" tells them
exactly what to work on.

### Interactive console

```bash
python scripts/build_console.py && open web/bioage_console.html
```

A self-contained HTML console — no server, no build step, no network. Browse all
4,623 participants, click the blood-vs-wearable scatter to inspect anyone, and
**toggle modalities on and off to watch the combiner renormalise live**. That
toggle runs the same arithmetic as `Combiner.predict_gap`, so it demonstrates
graceful degradation rather than illustrating it.

The data is inlined (~140 KB) because the page is published under a strict CSP
with no fetch available — the upside is the built file works from disk, over
email, or on any static host. `web/console.template.html` is the source; the
built file is generated, so edit the template and re-run the script.

Graceful degradation — missing arms are dropped and remaining weights
renormalised, never imputed as average:

```bash
./.venv/bin/python scripts/bioage_report.py --seqn 31311 --drop wearable
./.venv/bin/python scripts/bioage_report.py --input person.json --json
```

## Layout

```
bioage/
  config.py       cohort, file registry, feature registry, curve parameters
  acquire.py      cached downloads; XPT / fixed-width readers
  nhanes.py       analytic table: demographics, labs, exclusions, mortality
  wearable.py     75M-row accelerometry → per-participant features
  curves.py       weighted LOESS, monotonicity screening, inversion
  scoring.py      outcome weights, reliability, calibration, combiner
  methylation.py  published clocks via Biolearn
  validate.py     mortality validation, redundancy, figures
scripts/
  run_pipeline.py     Phases 1–5 end to end
  bioage_report.py    Phase 6 output interface
outputs/
  figures/  tables/  logs/  manifest.json
```

## Data sources

| Source | What | Access |
|---|---|---|
| NHANES 2005–2006 | demographics, biochemistry, CBC, CRP, HbA1c, lipids, BP, body measures | public, no login |
| NHANES PAXRAW_D | minute-level accelerometry, 75M rows | public, 471 MB zipped |
| NCHS Public-Use LMF | vital status + follow-up through 2019-12-31 | public, fixed-width |
| GEO GSE40279 | 656-sample blood methylation array with ages | public |

All join on `SEQN` **except methylation**, which is a different cohort — see below.

## Known limitations (each marked `# SCOPE` in the source)

- **Methylation is a separate cohort with no mortality linkage.** NHANES has
  never collected DNA methylation. GSE40279 has chronological age only. This arm
  therefore gets **no fitted combiner weight** and is excluded from the combined
  score — reported in the breakdown, flagged as such. No cross-cohort join is
  fabricated.
- **The wearable arm is a weak proxy for a consumer wearable.** The 2005–2006
  waist-worn uniaxial ActiGraph gives no HRV, sleep staging or resting-HR trend —
  the channels carrying most of the aging signal in an Apple Watch or Oura ring.
  Treat this arm as a lower bound.
- **Combined beats the best single arm by +0.0045 C-index** (+51% on the
  increment over age+sex). Real and consistent, but the absolute numbers sit
  close together because chronological age alone already gives 0.845.
- Two healthy-reference exclusions (`heavy_alcohol`, `gout_meds`) have no source
  in the acquired files and are logged as unapplied.
- Survey weights are applied to the reference curves but not inside the Cox
  models; population-generalisable hazard estimates are not claimed.
