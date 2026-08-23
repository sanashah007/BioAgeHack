# Methylation clocks and the intervention protocol

Two features layered onto the console: client-side DNAm PhenoAge/GrimAge
computation, and a per-marker, evidence-graded intervention protocol
(diet/exercise -> supplements -> drugs -> peptides). Both originated in a
teammate's separate build, `web/bioage_console_refit.html` (kept unmodified,
byte-identical to what they handed off -- see that file and
`web/bioage_console_refit_README.txt`).

## What was brought in, and what wasn't

That teammate's file also included its own refit of the blood/wearable
scoring model -- different feature set, different combiner, and a materially
different validation result (their combined `c_gain` was ~0 to slightly
negative vs. this repo's committed +0.0134; see their file's own validation
table). **That refit was deliberately left out.** This repo's own fitted
curves, combiner, and validation numbers (`data/processed/scoring_bundle.*`,
`outputs/tables/validation.csv`) are unchanged. Nobody's work was discarded --
their full original build is preserved at `web/bioage_console_refit.html` for
anyone who wants to run or compare it -- but the console's headline numbers
still come from this repo's own, stronger-validated model.

What *was* brought in, as pure additions on top of that unchanged model:

- **Methylation clocks** (`data/processed/methylation_clocks.json`): DNAm
  PhenoAge (513 CpGs, runs entirely client-side) and GrimAge coefficients
  (8 sub-models; needs the CpGs *and* age/sex). Self-contained -- doesn't
  touch blood/wearable scoring at all, reported as an independent arm exactly
  like the existing methylation discussion in the main README.
- **The intervention protocol** (`data/processed/interventions.json`,
  `data/processed/feature_elite_ranges.json`): per-marker evidence-graded
  interventions (A/B/C/experimental) across four tiers, plus the "elite
  range" each marker is compared against.

## The coverage gap

The intervention library was built against the teammate's own 20-marker
feature set, which only partially overlaps this repo's 32. **12 markers have
real intervention entries; 20 do not** -- including several of the biggest
drivers this model actually surfaces (hs-CRP, RDW, GGT, ALT all show up
repeatedly as top contributors in real scored panels).

Covered (12): `albumin`, `creatinine`, `glucose_fasting`, `hba1c`,
`hemoglobin`, `lymph_pct`, `mcv`, `sedentary_frac`, `total_chol`,
`urea_nitrogen`, `uric_acid`, `waist` -- plus `grimage` / `phenoage` for the
methylation arm.

**Not covered (20) -- needs intervention entries from whoever built the
original library:**

Blood: `alk_phos`, `alt`, `crp`, `ggt`, `hdl`, `pulse_pressure`, `rdw`,
`systolic_bp`, `total_protein`, `triglycerides`, `wbc`

Wearable: `activity_bout_frag`, `activity_cv`, `interdaily_stability`,
`intradaily_variability`, `l5`, `m10`, `mean_cpm`, `mvpa_min`,
`relative_amplitude`

The console degrades honestly for these: a flagged marker with no
intervention entry shows no protocol card, rather than a fabricated one --
`renderProtocol()` filters `driverList()` on `d.iv` being present.

Some of these have no home at all in the teammate's original file either --
their wearable feature engineering (`mean_steps_per_day`, `mean_intensity`,
`total_counts_per_day`, `mvpa_frac`, `cv_daily_counts`) doesn't correspond to
any of this repo's wearable metrics, and their blood panel had no `hematocrit`
/ `rbc` / `neutrophil_pct` equivalents to bioage/config.py's `crp` / `rdw` /
etc. Those entries could not be remapped and were dropped rather than guessed
at -- see `data/processed/interventions.json` for exactly what's there.

## Folding methylation into the combined score (scoped, not built)

The combined score currently excludes methylation entirely -- reported in the
per-arm breakdown, never weighted into the number the user sees as their
"biological age." The reason isn't that methylation age lacks mortality
evidence -- **GrimAge specifically has some of the strongest published
mortality validation of any biological age measure**, which is exactly why
`bioage/methylation.py` picks it as primary over first-generation clocks. The
reason is narrower and mechanical: `Combiner` weights are fit by Cox
regression against *this repo's own* linked mortality (`fit_combiner` in
`bioage/scoring.py`), and GSE40279 -- the cohort the GrimAge/PhenoAge
coefficients were computed from -- has chronological age only, no vital
status. There is no cohort inside this codebase to fit a methylation weight
against.

`bioage/methylation.py` already anticipates this being solved with an
externally-sourced weight: "excluded from the combined score unless a caller
explicitly opts in to a clearly-labelled provisional weight." What follows is
that path, scoped but not implemented -- flagged in code review as too large
to build in the same pass as the UI merge above, but real enough to be worth
recording precisely rather than losing the research.

**The published number.** Lu AT, Quach A, Wilson JG, et al. "DNA methylation
GrimAge strongly predicts lifespan and healthspan." *Aging* (Albany NY).
2019;11(2):303-327. doi:10.18632/aging.101684. PMID 30669119. The paper's own
meta-analysis across 9 strata from 5 cohorts (not a single convenience
cohort): **HR = 1.10 per year of AgeAccelGrim, p = 2.0×10⁻⁷⁵.**

**Why that number is usable, mechanically.** `fit_combiner` computes,
per modality:

```python
raw[mod] = cox_coefficient[mod] / gap_std[mod]   # = ln(HR) per year of gap
weights[mod] = raw[mod] / sum(|raw| across modalities)
```

`cox_coefficient / std` is exactly *ln(hazard ratio) per year of gap* -- the
same statistical quantity Lu et al. report as 1.10. So
`raw['methylation'] = ln(1.10) ≈ 0.0953` slots into that formula on the same
footing as the internally-fit `raw['blood']` / `raw['wearable']`, and the
whole set gets renormalized together.

**The wrinkle that blocked building this now.** Blood and wearable gaps are
not raw `implied_age - chronological_age` by the time they reach
`fit_combiner` -- `bioage/scorer.py` first reliability-shrinks them (split-half
across features) and age-detrends them (regress out the mechanical
age-dependence of an imperfect predictor). The methylation gap
(`GrimAge - age`) has been through neither step. Plugging an external HR
against an uncalibrated gap while the other two weights were fit against
calibrated gaps biases the methylation weight relative to the other two, in an
unknown direction. GrimAge/PhenoAge gaps likely don't need
`calibrate_to_age_scale`'s rescaling (they're already in years by
construction, unlike wearable's arbitrary units), but reliability and
detrending still apply.

**Why that wasn't built today.** Both of those calibration steps need the
real per-sample GSE40279 data (656 donors, full CpG-level values) to compute
against. That table isn't cached anywhere in this repo --
`bioage/methylation.py` fetches it live via Biolearn on first use, and per
`.gitignore`, that's a ~2.7 GB download, the same class of blocker as the
NHANES raw-data fetch `run_pipeline.py` needs. Decided to scope this
precisely and stop rather than either hand-wave the calibration or trigger an
unplanned multi-GB download mid-session.

**To pick this back up:**

1. Run the Biolearn fetch (`bioage/methylation.py`'s `build_methylation_table`
   or equivalent) to get per-sample GrimAge/PhenoAge and age for all 656
   GSE40279 donors.
2. Compute split-half reliability for the clock the same way
   `split_half_reliability` does for blood/wearable -- split the CpG
   coefficient set in half, score each half, correlate. (PhenoAge, being a
   513-CpG linear model, is a direct fit for this. GrimAge's 8-submodel
   structure needs a decision on what "half" means -- split submodels, or
   split CpGs within each submodel.)
3. Run `deattenuate()` on the (reliability-shrunk) gap against GSE40279's own
   `age` column to get `(intercept, slope)`.
4. Add `raw['methylation'] = ln(1.10)` (or a freshly-checked HR, if a more
   current/appropriate meta-analysis has since superseded Lu 2019) into
   `fit_combiner`'s weight computation, sourced from a constant clearly
   labeled as externally-sourced -- not fit, and cite the paper in the code
   comment the way this doc does.
5. Re-verify `test_invariants.py` and the console's browser-side reproduction
   of the Python scores still agree, the same invariant this repo already
   checks for blood/wearable.

## Extending it

`interventions.json` is keyed by `bioage/config.py` feature name (not the
teammate's original naming -- see the rename table in the extraction script
this was built from, no longer present in the repo since it was a one-off).
Each entry:

```json
{
  "direction": "raise | lower | optimise",
  "tiers": {
    "diet_exercise": [{"name": "...", "how": "...", "evidence": "A|B|C|experimental",
                        "effect": "...", "caution": "..."}],
    "supplements": [...], "drugs": [...], "peptides": [...]
  }
}
```

`feature_elite_ranges.json` is `{marker_name: {"elite": {"male": [lo, hi],
"female": [lo, hi]}, "direction": "..."}}`. Add an entry to both files (same
key, the `bioage/config.py` feature name) and it picks up automatically on
the next `python scripts/build_console.py` -- no code change needed, per
`renderProtocol()`/`driverList()` in `web/console.template.html`.
