"""Input and output contracts for the recommendations module.

`HealthProfile` is the hand-off point from the upload/analysis half of the app:
whatever that pipeline produces, it needs to be shaped into this before calling
`generate_recommendations`. Everything except `demographics` is optional, so a
partial profile (bloods only, no wearable, no methylation) still works.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------


class Sex(str, Enum):
    male = "male"
    female = "female"
    unspecified = "unspecified"


class Demographics(BaseModel):
    chronological_age: float = Field(..., description="Age in years.")
    sex: Sex = Sex.unspecified
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None


class BloodMarker(BaseModel):
    """One result off a blood panel.

    `name` is free text as it appeared on the lab report; it is normalized
    against `reference_ranges.MARKER_ALIASES` before evaluation.
    """

    name: str
    value: float
    unit: str
    lab_ref_low: Optional[float] = Field(
        None, description="Lower bound printed on the lab report, if any."
    )
    lab_ref_high: Optional[float] = Field(
        None, description="Upper bound printed on the lab report, if any."
    )
    collected_on: Optional[str] = Field(None, description="ISO date, YYYY-MM-DD.")


class MethylationClock(str, Enum):
    grimage = "GrimAge"
    phenoage = "PhenoAge"
    dunedinpace = "DunedinPACE"
    horvath = "Horvath"


class MethylationResult(BaseModel):
    clock: MethylationClock
    predicted_age: float = Field(
        ..., description="Clock output in years. For DunedinPACE, pass pace in "
                         "years-per-year and leave age_acceleration null."
    )
    age_acceleration: Optional[float] = Field(
        None, description="predicted_age minus chronological_age, in years."
    )
    # GrimAge sub-components are highly actionable, so accept them when present.
    dnam_pack_years: Optional[float] = None
    dnam_adm: Optional[float] = Field(None, description="DNAm adrenomedullin.")
    dnam_pai1: Optional[float] = Field(None, description="DNAm PAI-1.")


class WearableMetrics(BaseModel):
    """Aggregates, not raw time series — keep the LLM payload small."""

    resting_heart_rate_bpm: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    vo2max_estimate: Optional[float] = None
    avg_sleep_hours: Optional[float] = None
    sleep_efficiency_pct: Optional[float] = None
    avg_sleep_onset_time: Optional[str] = Field(None, description="e.g. '00:45'.")
    sleep_consistency_pct: Optional[float] = None
    avg_daily_steps: Optional[float] = None
    weekly_zone2_minutes: Optional[float] = None
    weekly_vigorous_minutes: Optional[float] = None
    avg_daily_active_calories: Optional[float] = None
    days_of_data: Optional[int] = None


class Lifestyle(BaseModel):
    """Self-reported context. Optional, but it is what makes the advice land."""

    smoker: Optional[bool] = None
    alcohol_drinks_per_week: Optional[float] = None
    diet_pattern: Optional[str] = Field(None, description="e.g. 'mostly vegetarian'.")
    known_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    supplements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(
        default_factory=list,
        description="Things the plan must respect: 'no gym access', "
                    "'shift worker', 'vegetarian', 'bad knees'.",
    )
    goals: List[str] = Field(default_factory=list)


class ModalityGap(BaseModel):
    """One channel's gap against chronological age, in years.

    The upstream pipeline scores blood, wearable and methylation separately and
    combines on *gaps* rather than on predicted ages. The per-channel breakdown
    is the actionable part: a combined gap near zero can hide two channels
    twenty years apart, and that divergence is what a person can act on.
    """

    modality: str = Field(..., description="blood | wearable | methylation")
    gap_years: float = Field(
        ..., description="modality_bioage minus chronological age."
    )
    contribution_years: Optional[float] = Field(
        None, description="This channel's share of the combined gap (weight x gap)."
    )
    weight: Optional[float] = Field(
        None, description="Combiner weight, where the channel has a fitted one."
    )


class MarkerAttribution(BaseModel):
    """The scoring model's own view of one marker.

    Distinct from our reference-range severity, and often disagrees with it:
    a marker can sit only slightly outside the optimal range yet carry the
    largest mortality weight in the model. Where they disagree, the fitted
    weight is the better guide to what to prioritize.
    """

    marker: str
    modality: str
    implied_age_years: float = Field(
        ..., description="Age this value resembles on the sex-matched "
                         "reference curve."
    )
    mortality_weight: float = Field(
        ..., description="Fitted contribution to the modality score. 0 means "
                         "it did not contribute at all."
    )
    extrapolated: bool = Field(
        ..., description="True if the value fell outside the curve's range, "
                         "making implied age a bound rather than an estimate."
    )


class BiologicalAgeResult(BaseModel):
    """Whatever the analysis half of the app computed."""

    predicted_biological_age: float
    age_acceleration: Optional[float] = None
    model_name: Optional[str] = Field(
        None, description="Which model produced this, for display."
    )
    confidence_note: Optional[str] = None

    per_modality: List[ModalityGap] = Field(
        default_factory=list, description="Per-channel breakdown, where available."
    )
    driver: Optional[str] = Field(
        None, description="Modality contributing most to the combined gap."
    )
    divergence_years: Optional[float] = Field(
        None,
        description="Spread between the fastest- and slowest-aging channel. "
                    "Large values mean the channels disagree, which matters "
                    "more than the combined number.",
    )
    modalities_missing: List[str] = Field(
        default_factory=list,
        description="Channels not supplied. Dropped and remaining weights "
                    "renormalised — never imputed as average.",
    )
    marker_attribution: List[MarkerAttribution] = Field(
        default_factory=list,
        description="Per-marker attribution from the scoring model, heaviest "
                    "weight first, where the pipeline supplied it.",
    )


class HealthProfile(BaseModel):
    demographics: Demographics
    biological_age: Optional[BiologicalAgeResult] = None
    blood_markers: List[BloodMarker] = Field(default_factory=list)
    methylation: List[MethylationResult] = Field(default_factory=list)
    wearables: Optional[WearableMetrics] = None
    lifestyle: Optional[Lifestyle] = None


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


class Severity(str, Enum):
    optimal = "optimal"
    suboptimal = "suboptimal"
    out_of_range = "out_of_range"
    high_concern = "high_concern"


SEVERITY_ORDER = {
    Severity.high_concern: 0,
    Severity.out_of_range: 1,
    Severity.suboptimal: 2,
    Severity.optimal: 3,
}


class Effort(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Action(BaseModel):
    action: str = Field(..., description="One concrete, specific thing to do.")
    rationale: str = Field(..., description="Why this moves this marker.")
    effort: Effort
    timeframe: str = Field(..., description="e.g. 'daily', 'next 8 weeks'.")


class MarkerRecommendation(BaseModel):
    marker: str
    category: str = Field(..., description="blood | methylation | wearable | lifestyle")
    current_value: str = Field(..., description="Value with unit, for display.")
    healthy_range: str = Field(..., description="Target range with unit, for display.")
    severity: Severity
    why_it_matters: str = Field(
        ..., description="Plain-language link to aging/biological age. 1-3 sentences."
    )
    expected_impact: str = Field(
        ..., description="What plausibly changes if the actions are followed, "
                         "and over what horizon. Hedged, not promised."
    )
    # Count limits are stated in the prompt rather than as min_length /
    # max_length: those emit minItems/maxItems, which the structured-output
    # schema validator does not accept.
    actions: List[Action] = Field(..., description="1-4 actions.")
    clinician_flag: bool = Field(
        ..., description="True if this warrants a conversation with a doctor."
    )


class Priority(BaseModel):
    rank: int
    focus: str = Field(..., description="Short label, e.g. 'Lower fasting insulin'.")
    why_now: str
    markers_addressed: List[str]


class RecommendationReport(BaseModel):
    summary: str = Field(
        ..., description="2-4 sentences framing the overall picture."
    )
    # Every field here is required with no default: the structured-output
    # schema marks defaulted fields optional, and we want the model to
    # explicitly emit an empty list rather than silently omit the key.
    biological_age_drivers: List[str] = Field(
        ..., description="Which inputs appear to be pushing biological age up or down."
    )
    top_priorities: List[Priority] = Field(
        ..., description="At most 3, ranked 1 first."
    )
    recommendations: List[MarkerRecommendation]
    caveats: List[str] = Field(
        ..., description="Data gaps and limits — missing panels, short wearable history."
    )


class RecommendationResponse(BaseModel):
    """What the API returns. `disclaimer` is set server-side, never by the LLM."""

    report: RecommendationReport
    flagged_marker_count: int
    model: str
    disclaimer: str
