"""Adapter between the BioAgeHack scoring pipeline and this module.

BioAgeHack emits a report in two slightly different shapes depending on which
entry point produced it, and this module normalises both:

- `scripts/bioage_report.py` (cohort replay) → `modality_contribution_years`,
  `combiner_weights`, `driver`.
- `bioage.scorer.ScoringBundle.score()` (an individual's raw markers) →
  `contribution`, `detail` with per-feature implied ages, and no `driver`.

Use `normalize_scorer_report()` on the latter before passing it in.

Nothing here re-implements any scoring. Individual scoring belongs to
`bioage.scorer`, which applies three calibration steps — shrink by split-half
reliability, subtract the cohort gap-vs-age trend, multiply by the
age-equivalent scale — that a naive invert-and-average would silently skip,
producing a number that does not mean what their validation says it means.

The marker-name mapping between the two projects lives here rather than in
either codebase, so neither has to know about the other's vocabulary.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .schemas import (
    BiologicalAgeResult,
    BloodMarker,
    MarkerAttribution,
    Demographics,
    HealthProfile,
    Lifestyle,
    ModalityGap,
    Sex,
    WearableMetrics,
)

logger = logging.getLogger(__name__)

#: Their feature name -> (our lab-report name, our unit).
#:
#: Their names are NHANES variable slugs; ours are what appears on a lab
#: report. Units differ in two places and both are real conversions, not
#: cosmetic: their CRP is mg/dL where a modern hs-CRP assay reports mg/L, and
#: their enzymes are IU/L where ours are U/L (numerically identical, but the
#: string has to match for the reference table to resolve).
FEATURE_TO_MARKER: Dict[str, Tuple[str, str]] = {
    "albumin": ("Albumin", "g/dL"),
    "creatinine": ("Creatinine", "mg/dL"),
    "alk_phos": ("Alkaline phosphatase", "U/L"),
    "uric_acid": ("Uric acid", "mg/dL"),
    "ggt": ("GGT", "U/L"),
    "alt": ("ALT", "U/L"),
    "crp": ("hs-CRP", "mg/dL"),  # converted to mg/L by the reference table
    "wbc": ("WBC", "10^3/uL"),
    "lymph_pct": ("Lymphocyte %", "%"),
    "rdw": ("RDW", "%"),
    "mcv": ("MCV", "fL"),
    "hba1c": ("HbA1c", "%"),
    "glucose_fasting": ("Fasting glucose", "mg/dL"),
    "hdl": ("HDL cholesterol", "mg/dL"),
    "triglycerides": ("Triglycerides", "mg/dL"),
}

#: Their features with no reference range on our side. Forwarded to the model
#: as ungraded context rather than dropped — several are clinically meaningful
#: (systolic BP, waist) even though we do not grade them.
UNGRADED_FEATURES: Dict[str, Tuple[str, str]] = {
    "urea_nitrogen": ("Blood urea nitrogen", "mg/dL"),
    "total_protein": ("Total protein", "g/dL"),
    "hemoglobin": ("Hemoglobin", "g/dL"),
    "total_chol": ("Total cholesterol", "mg/dL"),
    "systolic_bp": ("Systolic blood pressure", "mmHg"),
    "pulse_pressure": ("Pulse pressure", "mmHg"),
    "waist": ("Waist circumference", "cm"),
}

MODEL_NAME = "BioAgeHack multi-modal combiner (NHANES mortality-fitted)"


# --------------------------------------------------------------------------
# Their report -> our contract
# --------------------------------------------------------------------------


def report_to_biological_age(report: dict) -> BiologicalAgeResult:
    """Convert a `bioage_report.py --json` payload into our result object."""
    per_modality_gap: Dict[str, float] = report.get("per_modality_gap") or {}
    contributions: Dict[str, float] = report.get("modality_contribution_years") or {}
    weights: Dict[str, float] = report.get("combiner_weights") or {}

    per_modality = [
        ModalityGap(
            modality=name,
            gap_years=gap,
            contribution_years=contributions.get(name),
            weight=weights.get(name),
        )
        for name, gap in sorted(per_modality_gap.items())
    ]

    divergence = None
    if len(per_modality_gap) >= 2:
        divergence = round(
            max(per_modality_gap.values()) - min(per_modality_gap.values()), 1
        )

    notes: List[str] = []
    if report.get("modalities_missing"):
        notes.append(
            "Missing channels were dropped and remaining weights renormalised, "
            "not imputed as average."
        )
    if "methylation" in per_modality_gap:
        # Their methylation arm is a different cohort with no mortality
        # linkage, so it carries no fitted weight and sits outside the
        # combined score. The model must not read it as equally grounded.
        notes.append(
            "The methylation channel is reported but excluded from the combined "
            "score — it comes from a separate cohort with no mortality linkage "
            "and therefore has no fitted weight."
        )

    combined_bioage = report.get("combined_bioage")
    if combined_bioage is None:
        raise ValueError(
            "Report has no combined_bioage — no usable modality was supplied."
        )

    attribution = [MarkerAttribution(**entry) for entry in feature_attribution(report)]
    if any(a.extrapolated for a in attribution):
        notes.append(
            "Some markers fell outside their reference curve's range; their "
            "implied ages are bounds pinned to a curve endpoint, not estimates."
        )

    return BiologicalAgeResult(
        predicted_biological_age=combined_bioage,
        age_acceleration=report.get("combined_gap"),
        model_name=MODEL_NAME,
        confidence_note=" ".join(notes) or None,
        per_modality=per_modality,
        driver=report.get("driver"),
        divergence_years=divergence,
        modalities_missing=list(report.get("modalities_missing") or []),
        marker_attribution=attribution,
    )


def normalize_scorer_report(scored: dict) -> dict:
    """Bring `bioage.scorer.ScoringBundle.score()` output to the report shape.

    The scorer names its contributions `contribution`, omits `combiner_weights`
    and never computes a `driver`. Derive the driver the same way
    `bioage_report.build_report` does — largest absolute contribution — rather
    than inventing a different rule, so both paths agree.
    """
    contributions: Dict[str, float] = scored.get("contribution") or {}
    driver = (
        max(contributions, key=lambda k: abs(contributions[k]))
        if contributions
        else None
    )
    return {
        "subject": scored.get("subject", "uploaded panel"),
        "chronological_age": scored["chronological_age"],
        "sex": scored.get("sex", "unspecified"),
        "combined_bioage": scored.get("combined_bioage"),
        "combined_gap": scored.get("combined_gap"),
        "per_modality_gap": scored.get("per_modality_gap") or {},
        "modality_contribution_years": contributions,
        "combiner_weights": {},
        "modalities_used": scored.get("modalities_used") or [],
        "modalities_missing": scored.get("modalities_missing") or [],
        "driver": driver,
        "detail": scored.get("detail") or {},
    }


def feature_attribution(scored_or_report: dict) -> List[dict]:
    """Per-marker attribution from the scorer's `detail` block.

    This is the pipeline's *own* view of which markers drive the score:
    `implied_age` is what that value looks like age-wise against the sex-matched
    reference curve, and `weight` is its mortality-fitted contribution. Both are
    far better prioritisation signal than a reference-range severity alone, so
    they are worth putting in front of the model.

    Markers with `weight == 0` did not contribute to the gap at all. Markers
    with `extrapolated: true` fell outside the curve's range, making the implied
    age a bound rather than an estimate — flagged so nothing over-reads them.
    """
    detail = scored_or_report.get("detail") or {}
    out: List[dict] = []
    for modality, block in detail.items():
        for feature, info in (block.get("per_feature") or {}).items():
            label = (
                FEATURE_TO_MARKER.get(feature)
                or UNGRADED_FEATURES.get(feature)
                or (feature, "")
            )[0]
            out.append(
                {
                    "marker": label,
                    "modality": modality,
                    "implied_age_years": round(float(info["implied_age"]), 1),
                    "mortality_weight": round(float(info.get("weight", 0.0)), 4),
                    "extrapolated": bool(info.get("extrapolated", False)),
                }
            )
    # Heaviest contributors first — that is the order worth reading.
    out.sort(key=lambda d: d["mortality_weight"], reverse=True)
    return out


_scoring_bundle = None  # cached; unpickling is ~218 KB, not a per-call cost


def score_markers(
    *,
    age: float,
    sex: str,
    markers: Dict[str, Dict[str, float]],
    lifestyle: Optional[Lifestyle] = None,
) -> Tuple[dict, HealthProfile]:
    """Score raw markers and build a profile, in one call.

    `markers` is `{"blood": {"crp": 0.34, ...}, "wearable": {...}}`, keyed by
    `bioage.config` feature names — the shape both the CLI's `--markers` and
    the HTTP `/api/analyze` endpoint accept from the outside.

    Shared by both callers so the scoring bundle is loaded once per process
    (module-level cache here) rather than once per request, and so the
    normalize -> profile chain has one implementation instead of two that can
    drift apart.
    """
    global _scoring_bundle
    from bioage import scorer  # deferred: keeps this module importable without

    # bioage's heavier deps (numpy/scipy/lifelines/scikit-learn) when nothing
    # calls score_markers.
    if _scoring_bundle is None:
        _scoring_bundle = scorer.load()

    scored = _scoring_bundle.score(age=age, sex=sex, values=markers)
    report = normalize_scorer_report(scored)
    profile = report_to_profile(
        report, raw_features=markers.get("blood"), lifestyle=lifestyle
    )
    return report, profile


def _sex(value: Optional[str]) -> Sex:
    try:
        return Sex(str(value).strip().lower())
    except ValueError:
        return Sex.unspecified


def report_to_profile(
    report: dict,
    *,
    raw_features: Optional[Dict[str, float]] = None,
    wearables: Optional[WearableMetrics] = None,
    lifestyle: Optional[Lifestyle] = None,
) -> HealthProfile:
    """Build a full `HealthProfile` from their report plus the raw panel.

    `raw_features` is keyed by *their* feature names (`crp`, `hba1c`, ...) —
    i.e. what went into the pipeline. Without it you still get a valid profile,
    but the recommendations degrade sharply: nothing downstream knows any
    marker value, so it can only speak to which channel is aging fastest.
    """
    markers: List[BloodMarker] = []
    for feature, value in (raw_features or {}).items():
        mapping = FEATURE_TO_MARKER.get(feature) or UNGRADED_FEATURES.get(feature)
        if mapping is None:
            logger.debug("No marker mapping for feature %r; skipping.", feature)
            continue
        if value is None:
            continue
        name, unit = mapping
        markers.append(BloodMarker(name=name, value=float(value), unit=unit))

    return HealthProfile(
        demographics=Demographics(
            chronological_age=float(report["chronological_age"]),
            sex=_sex(report.get("sex")),
        ),
        biological_age=report_to_biological_age(report),
        blood_markers=markers,
        wearables=wearables,
        lifestyle=lifestyle,
    )
