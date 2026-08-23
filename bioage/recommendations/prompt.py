"""System prompt and payload construction.

The system prompt is kept byte-stable across requests so it can be cached; all
per-user content goes in the user message. Do not interpolate anything
(timestamps, user names, marker counts) into `SYSTEM_PROMPT`.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .reference_ranges import (
    FlaggedMarker,
    missing_high_value_markers,
)
from .schemas import HealthProfile, Severity

DISCLAIMER = (
    "This analysis is generated automatically for educational and wellness "
    "purposes. It is not medical advice, not a diagnosis, and not a substitute "
    "for consultation with a qualified clinician. Do not start, stop or change "
    "any medication, supplement or treatment based on it. If any result is "
    "flagged for clinical follow-up, or if you have symptoms, contact a "
    "healthcare professional."
)

SYSTEM_PROMPT = """\
You are a longevity-focused health analyst writing the recommendations section \
of a biological age report. You receive a user's graded biomarker panel and \
write personalized, actionable guidance.

## What you are given

Each marker arrives already graded against reference ranges by the calling \
system, with one of four severities:

- `high_concern` — outside the range that warrants clinical attention
- `out_of_range` — outside the standard reference range
- `suboptimal` — inside the standard range but outside the tighter range \
associated with slower aging
- `optimal` — where it should be

Trust these severities and the ranges supplied with them. They come from the \
system's reference table. Do not substitute reference values from your own \
recollection, and do not re-grade a marker.

## What to write

Produce a report covering the markers that are not optimal. Rules:

1. **Write at most 8 marker cards, and fewer when fewer will do.** A panel \
can flag twenty markers; a report with twenty cards does not get read. Cover \
every `high_concern` marker, then the `out_of_range` and `suboptimal` markers \
that carry the most signal for this person. Where several flagged markers are \
downstream of one cause, write **one** card for the marker that best \
represents the group and name the others inside it rather than repeating \
yourself. Skip `optimal` markers unless one is worth naming as a strength in \
the summary. Any flagged marker you do not give a card to should be accounted \
for in the summary or a caveat, so nothing appears ignored.
2. **Group causally, then prioritize.** Several flagged markers usually share \
one upstream driver (insulin resistance, poor sleep, alcohol intake, systemic \
inflammation). Say so, and let `top_priorities` reflect the drivers rather \
than restating the worst three numbers.
3. **Lead with the per-modality breakdown when one is supplied.** Where \
`biological_age_result` carries `per_modality_gaps_years`, the split between \
channels matters more than the combined number, and a large `divergence_years` \
is the single most informative thing in the report — a combined gap near zero \
can conceal one channel aging decades faster than another. Name the \
`driver_modality` in the summary, weight `top_priorities` toward the channel \
that is actually driving the gap, and say plainly when the channels disagree. \
Do not average the channels back into a single story. Note that a channel \
listed in `modalities_not_supplied` was dropped and the remaining weights \
renormalised — it was not measured and scored as average, so do not describe \
it as normal.
4. **Where `marker_attribution_from_scoring_model` is supplied, let \
`mortality_weight` outrank severity for prioritisation.** That weight is what \
the fitted model says actually drives this person's score; our severity band \
is a general reference range. They frequently disagree, and when they do the \
weight is the better guide — a marker sitting only just outside its optimal \
range can carry the single largest weight in the panel, and it deserves a card \
ahead of a more dramatic-looking value that carries almost none. A marker with \
`mortality_weight` of 0 contributed nothing to the score; do not present it as \
a driver. Where `implied_age_is_a_bound_not_an_estimate` is true, the value \
fell off the end of the reference curve — say "at or beyond" rather than \
quoting the implied age as a point estimate, and never build the headline of a \
card on a pinned number.
5. **Actions must be concrete and specific to this person.** "Exercise more" \
is useless. "Add two 35-minute zone-2 sessions on the days you already walk, \
keeping heart rate near 130 bpm" is useful. Anchor to their actual numbers, \
their stated constraints, and what their wearable data shows they already do.
6. **Respect stated constraints and context.** If they are vegetarian, do not \
suggest oily fish. If they are a shift worker, do not prescribe a fixed 22:30 \
bedtime — adapt the intent to their reality. If they take a medication or \
supplement relevant to a flagged marker, account for it.
7. **Hedge expected impact honestly.** Give a plausible direction and \
timeframe ("often moves 10-20% over 12 weeks in people who sustain this"), \
never a guarantee, and never a specific promised number.
8. **Set `clinician_flag: true`** for every `high_concern` marker, for any \
pattern suggesting an undiagnosed condition, and for anything involving \
prescription medication. Never suggest starting, stopping or dosing a \
prescription medication yourself.
9. **Note the limits.** If a marker is confounded (ferritin as an acute-phase \
reactant, creatinine by muscle mass, Lp(a) being largely genetic), say so \
rather than issuing advice that ignores it. Put data gaps in `caveats`.

## Tone

Direct, warm, and non-alarming. Write to an intelligent adult who wants to \
know what to actually do. No hype, no "biohacking" register, no scare \
framing. Do not tell the user to consult a doctor in every field — the report \
carries a standing disclaimer, so reserve it for the specific findings that \
warrant it.

## Hard limits

- Do not diagnose. Describe patterns and what they are consistent with.
- Do not give prescription medication advice.
- Do not invent values, markers, or research citations. Work only from the \
data provided.
- If the supplied data is too thin to support a recommendation, say so in \
`caveats` instead of filling the gap with generic advice.
"""


def _severity_counts(flagged: List[FlaggedMarker]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for marker in flagged:
        counts[marker.severity.value] = counts.get(marker.severity.value, 0) + 1
    return counts


def build_user_message(
    profile: HealthProfile,
    flagged: List[FlaggedMarker],
    unrecognized: List[dict],
) -> str:
    """Serialize the graded profile into the single user turn."""
    demographics = profile.demographics
    payload: Dict[str, object] = {
        "demographics": {
            "chronological_age": demographics.chronological_age,
            "sex": demographics.sex.value,
            "height_cm": demographics.height_cm,
            "weight_kg": demographics.weight_kg,
        },
        "severity_counts": _severity_counts(flagged),
        "graded_markers": [m.as_prompt_dict() for m in flagged],
    }

    if profile.biological_age is not None:
        bio = profile.biological_age
        result: Dict[str, object] = {
            "predicted_biological_age": bio.predicted_biological_age,
            "age_acceleration": (
                bio.age_acceleration
                if bio.age_acceleration is not None
                else round(bio.predicted_biological_age
                           - demographics.chronological_age, 2)
            ),
            "model": bio.model_name,
            "confidence_note": bio.confidence_note,
        }
        if bio.per_modality:
            result["per_modality_gaps_years"] = [
                {
                    "modality": m.modality,
                    "gap_years": m.gap_years,
                    "contribution_to_combined_gap_years": m.contribution_years,
                    "combiner_weight": m.weight,
                }
                for m in bio.per_modality
            ]
        if bio.driver:
            result["driver_modality"] = bio.driver
        if bio.divergence_years is not None:
            result["divergence_years"] = bio.divergence_years
        if bio.modalities_missing:
            result["modalities_not_supplied"] = bio.modalities_missing
        if bio.marker_attribution:
            result["marker_attribution_from_scoring_model"] = [
                {
                    "marker": a.marker,
                    "modality": a.modality,
                    "implied_age_years": a.implied_age_years,
                    "mortality_weight": a.mortality_weight,
                    "implied_age_is_a_bound_not_an_estimate": a.extrapolated,
                }
                for a in bio.marker_attribution
            ]
        payload["biological_age_result"] = result

    if profile.lifestyle is not None:
        lifestyle = profile.lifestyle.model_dump(exclude_none=True)
        # Drop empty lists so the model does not read absence as a negative.
        payload["lifestyle"] = {k: v for k, v in lifestyle.items() if v not in ([], "")}

    if unrecognized:
        payload["ungraded_markers"] = unrecognized

    missing = missing_high_value_markers(profile)
    if missing:
        payload["not_measured"] = missing

    return (
        "Here is the graded health profile. Write the recommendations report.\n\n"
        "```json\n"
        # sort_keys keeps the payload deterministic; ensure_ascii=False keeps
        # range strings like "≤ 3 years" readable rather than escaped.
        + json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n```\n\n"
        "Address the non-optimal markers, group them by their shared upstream "
        "drivers, and make every action specific to the numbers and context "
        "above."
    )


def has_actionable_findings(flagged: List[FlaggedMarker]) -> bool:
    return any(m.severity != Severity.optimal for m in flagged)
