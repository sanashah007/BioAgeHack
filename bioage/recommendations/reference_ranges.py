"""Deterministic reference ranges and marker flagging.

Severity is decided here in Python, not by the model. The LLM is good at
personalizing and prioritizing advice; it is not a reliable place to store
reference values. So we compute `optimal / suboptimal / out_of_range /
high_concern` from a table, and hand the model an already-graded list.

The ranges below are general adult wellness targets aimed at longevity, which
are deliberately tighter than the "not currently sick" ranges most labs print.
They are demo values and are not clinical guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schemas import (
    BloodMarker,
    HealthProfile,
    MethylationClock,
    Severity,
    Sex,
)

Bound = Tuple[Optional[float], Optional[float]]  # (low, high); None = unbounded


@dataclass(frozen=True)
class MarkerSpec:
    canonical: str
    unit: str
    optimal: Bound
    clinical: Bound
    #: Outside this, treat as high_concern and flag for a clinician.
    critical: Bound = (None, None)
    aging_relevance: str = ""
    #: Per-sex replacements for (optimal, clinical, critical).
    sex_overrides: Dict[str, Tuple[Bound, Bound, Bound]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Blood panel
# --------------------------------------------------------------------------

BLOOD_SPECS: Dict[str, MarkerSpec] = {
    "hs_crp": MarkerSpec(
        "hs-CRP", "mg/L", (None, 1.0), (None, 3.0), (None, 10.0),
        "Systemic inflammation; a PhenoAge input and one of the strongest "
        "modifiable drivers of accelerated aging.",
    ),
    "hba1c": MarkerSpec(
        "HbA1c", "%", (4.8, 5.4), (4.0, 5.6), (None, 6.5),
        "Three-month average glycemia. Glycation damage accumulates in "
        "collagen and vasculature.",
    ),
    "fasting_glucose": MarkerSpec(
        "Fasting glucose", "mg/dL", (75, 90), (70, 99), (None, 126),
        "PhenoAge input; early marker of insulin resistance.",
    ),
    "fasting_insulin": MarkerSpec(
        "Fasting insulin", "uIU/mL", (2.0, 5.0), (2.0, 8.0), (None, 15.0),
        "Rises years before glucose does; central to metabolic aging.",
    ),
    "triglycerides": MarkerSpec(
        "Triglycerides", "mg/dL", (None, 80), (None, 150), (None, 500),
        "Tracks insulin resistance and hepatic fat more tightly than LDL.",
    ),
    "hdl": MarkerSpec(
        "HDL-C", "mg/dL", (60, None), (40, None), (30, None),
        "Reverse cholesterol transport; low HDL tracks cardiometabolic risk.",
        sex_overrides={
            Sex.female.value: ((65, None), (50, None), (35, None)),
        },
    ),
    "ldl": MarkerSpec(
        "LDL-C", "mg/dL", (None, 100), (None, 130), (None, 190),
        "Cumulative LDL exposure is the primary driver of atherosclerosis.",
    ),
    "apob": MarkerSpec(
        "ApoB", "mg/dL", (None, 80), (None, 100), (None, 130),
        "Counts atherogenic particles directly; better risk signal than LDL-C.",
    ),
    "lpa": MarkerSpec(
        "Lp(a)", "nmol/L", (None, 75), (None, 125), (None, 250),
        "Largely genetic and not lifestyle-modifiable, but it changes how "
        "aggressively the other lipids should be managed.",
    ),
    "homocysteine": MarkerSpec(
        "Homocysteine", "umol/L", (None, 9.0), (None, 15.0), (None, 30.0),
        "Elevated levels track vascular and cognitive aging; often responds "
        "to B12/folate status.",
    ),
    "albumin": MarkerSpec(
        "Albumin", "g/dL", (4.3, 5.0), (3.5, 5.2), (3.0, None),
        "PhenoAge input. Low albumin reflects inflammation and poor "
        "nutritional/hepatic status.",
    ),
    "creatinine": MarkerSpec(
        "Creatinine", "mg/dL", (0.7, 1.1), (0.6, 1.3), (None, 2.0),
        "PhenoAge input; kidney filtration, though confounded by muscle mass.",
        sex_overrides={
            Sex.female.value: ((0.6, 0.9), (0.5, 1.1), (None, 1.8)),
        },
    ),
    "egfr": MarkerSpec(
        "eGFR", "mL/min/1.73m2", (90, None), (60, None), (45, None),
        "Kidney function declines predictably with age; a low value for age "
        "is a strong aging signal.",
    ),
    "alt": MarkerSpec(
        "ALT", "U/L", (None, 25), (None, 40), (None, 100),
        "Elevated ALT usually means hepatic fat, which tracks metabolic aging.",
    ),
    "ggt": MarkerSpec(
        "GGT", "U/L", (None, 25), (None, 50), (None, 150),
        "Sensitive to alcohol intake and oxidative stress.",
    ),
    "alkaline_phosphatase": MarkerSpec(
        "Alkaline phosphatase", "U/L", (45, 90), (35, 120), (None, 200),
        "PhenoAge input; liver and bone turnover.",
    ),
    "wbc": MarkerSpec(
        "White blood cell count", "10^3/uL", (4.5, 6.5), (3.5, 10.5), (2.5, 15.0),
        "PhenoAge input; chronically high counts indicate low-grade inflammation.",
    ),
    "lymphocyte_pct": MarkerSpec(
        "Lymphocyte %", "%", (30, 45), (20, 50), (10, None),
        "PhenoAge input; falls with immune aging (immunosenescence).",
    ),
    "rdw": MarkerSpec(
        "RDW", "%", (None, 13.0), (11.5, 14.5), (None, 16.0),
        "PhenoAge input and a surprisingly strong standalone mortality "
        "predictor.",
    ),
    "mcv": MarkerSpec(
        "MCV", "fL", (85, 92), (80, 100), (None, 110),
        "PhenoAge input; high values can indicate B12/folate deficit or "
        "alcohol intake.",
    ),
    "ferritin": MarkerSpec(
        "Ferritin", "ng/mL", (50, 150), (30, 300), (None, 700),
        "Iron stores; also an acute-phase reactant, so high values may mean "
        "inflammation rather than iron overload.",
        sex_overrides={
            Sex.female.value: ((40, 120), (15, 200), (None, 500)),
        },
    ),
    "vitamin_d": MarkerSpec(
        "Vitamin D (25-OH)", "ng/mL", (40, 60), (30, 80), (20, 100),
        "Low status is common and associates with immune and bone aging.",
    ),
    "tsh": MarkerSpec(
        "TSH", "uIU/mL", (0.5, 2.5), (0.4, 4.5), (None, 10.0),
        "Thyroid drives resting metabolic rate; subtle shifts affect energy, "
        "lipids and body composition.",
    ),
    "uric_acid": MarkerSpec(
        "Uric acid", "mg/dL", (3.5, 5.5), (2.5, 7.0), (None, 9.0),
        "Tracks fructose intake and metabolic load; high levels associate "
        "with hypertension.",
    ),
    "igf1": MarkerSpec(
        "IGF-1", "ng/mL", (100, 180), (80, 250), (None, 350),
        "Growth signalling; both very high and very low values associate "
        "with worse long-term outcomes.",
    ),
}

#: Lab reports name the same analyte many ways. Map lowercased report text to a key.
MARKER_ALIASES: Dict[str, str] = {
    "hs-crp": "hs_crp", "hscrp": "hs_crp", "crp": "hs_crp",
    "c-reactive protein": "hs_crp", "high sensitivity crp": "hs_crp",
    "hba1c": "hba1c", "a1c": "hba1c", "hemoglobin a1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glucose": "fasting_glucose", "fasting glucose": "fasting_glucose",
    "blood glucose": "fasting_glucose", "fasting blood sugar": "fasting_glucose",
    "insulin": "fasting_insulin", "fasting insulin": "fasting_insulin",
    "triglycerides": "triglycerides", "trigs": "triglycerides", "tg": "triglycerides",
    "hdl": "hdl", "hdl-c": "hdl", "hdl cholesterol": "hdl",
    "ldl": "ldl", "ldl-c": "ldl", "ldl cholesterol": "ldl",
    "ldl-c (calc)": "ldl", "ldl calculated": "ldl",
    "apob": "apob", "apo b": "apob", "apolipoprotein b": "apob",
    "lp(a)": "lpa", "lpa": "lpa", "lipoprotein(a)": "lpa",
    "homocysteine": "homocysteine",
    "albumin": "albumin", "serum albumin": "albumin",
    "creatinine": "creatinine", "serum creatinine": "creatinine",
    "egfr": "egfr", "gfr": "egfr", "estimated gfr": "egfr",
    "alt": "alt", "sgpt": "alt", "alanine aminotransferase": "alt",
    "ggt": "ggt", "gamma gt": "ggt", "gamma-glutamyl transferase": "ggt",
    "alp": "alkaline_phosphatase", "alkaline phosphatase": "alkaline_phosphatase",
    "wbc": "wbc", "white blood cells": "wbc", "leukocytes": "wbc",
    "white blood cell count": "wbc",
    "lymphocytes %": "lymphocyte_pct", "lymphocyte %": "lymphocyte_pct",
    "lymphocyte percent": "lymphocyte_pct", "lymphs %": "lymphocyte_pct",
    "rdw": "rdw", "red cell distribution width": "rdw",
    "mcv": "mcv", "mean corpuscular volume": "mcv",
    "ferritin": "ferritin",
    "vitamin d": "vitamin_d", "25-oh vitamin d": "vitamin_d",
    "vitamin d 25-hydroxy": "vitamin_d", "25(oh)d": "vitamin_d",
    "tsh": "tsh", "thyroid stimulating hormone": "tsh",
    "uric acid": "uric_acid", "urate": "uric_acid",
    "igf-1": "igf1", "igf1": "igf1", "insulin-like growth factor 1": "igf1",
}

#: Unit conversions into each spec's unit. Keyed by (marker key, lowercased unit).
UNIT_CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("hs_crp", "mg/dl"): 10.0,
    ("fasting_glucose", "mmol/l"): 18.016,
    ("triglycerides", "mmol/l"): 88.57,
    ("hdl", "mmol/l"): 38.67,
    ("ldl", "mmol/l"): 38.67,
    ("apob", "g/l"): 100.0,
    ("albumin", "g/l"): 0.1,
    ("creatinine", "umol/l"): 0.0113,
    ("vitamin_d", "nmol/l"): 0.4006,
    ("ferritin", "ug/l"): 1.0,
    ("wbc", "10^9/l"): 1.0,
    ("wbc", "k/ul"): 1.0,
    ("lpa", "mg/dl"): 2.5,  # approximate; assay-dependent
}


# --------------------------------------------------------------------------
# Wearable targets — same grading idea, different source
# --------------------------------------------------------------------------

WEARABLE_SPECS: Dict[str, MarkerSpec] = {
    "resting_heart_rate_bpm": MarkerSpec(
        "Resting heart rate", "bpm", (48, 60), (40, 70), (None, 85),
        "Tracks cardiovascular fitness and autonomic load; a rising baseline "
        "usually precedes other changes.",
    ),
    "hrv_rmssd_ms": MarkerSpec(
        "HRV (RMSSD)", "ms", (50, None), (30, None), (20, None),
        "Parasympathetic tone. Declines with age, stress load and poor sleep.",
    ),
    "vo2max_estimate": MarkerSpec(
        "VO2 max (estimated)", "mL/kg/min", (45, None), (35, None), (25, None),
        "The single strongest fitness predictor of all-cause mortality.",
        sex_overrides={
            Sex.female.value: ((38, None), (30, None), (21, None)),
        },
    ),
    "avg_sleep_hours": MarkerSpec(
        "Average sleep duration", "hours", (7.0, 8.5), (6.5, 9.0), (5.5, None),
        "Chronic short sleep drives inflammation, insulin resistance and "
        "impaired glymphatic clearance.",
    ),
    "sleep_efficiency_pct": MarkerSpec(
        "Sleep efficiency", "%", (88, None), (80, None), (70, None),
        "Fragmented sleep blunts overnight recovery even when time in bed "
        "looks adequate.",
    ),
    "sleep_consistency_pct": MarkerSpec(
        "Sleep timing consistency", "%", (85, None), (70, None), (55, None),
        "Irregular sleep timing associates with cardiometabolic risk "
        "independently of duration.",
    ),
    "avg_daily_steps": MarkerSpec(
        "Average daily steps", "steps", (8000, None), (6000, None), (3000, None),
        "Non-exercise movement volume; benefits accrue steeply up to ~8k.",
    ),
    "weekly_zone2_minutes": MarkerSpec(
        "Weekly zone 2 minutes", "min/week", (150, None), (90, None), (30, None),
        "Builds mitochondrial density and aerobic base.",
    ),
    "weekly_vigorous_minutes": MarkerSpec(
        "Weekly vigorous minutes", "min/week", (75, None), (40, None), (0, None),
        "High-intensity work is the main driver of VO2 max improvement.",
    ),
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class FlaggedMarker:
    """One graded marker, ready to be serialized into the prompt."""

    marker: str
    category: str
    value: float
    unit: str
    severity: Severity
    optimal_range: str
    acceptable_range: str
    aging_relevance: str
    direction: str  # "high" | "low" | "in_range"
    notes: List[str] = field(default_factory=list)

    def as_prompt_dict(self) -> dict:
        d = {
            "marker": self.marker,
            "category": self.category,
            "value": f"{_fmt(self.value)} {self.unit}".strip(),
            "severity": self.severity.value,
            "direction": self.direction,
            "optimal_range": self.optimal_range,
            "acceptable_range": self.acceptable_range,
            "why_this_marker_relates_to_aging": self.aging_relevance,
        }
        if self.notes:
            d["notes"] = self.notes
        return d


def _fmt(value: float) -> str:
    return f"{value:g}"


def _describe(bound: Bound, unit: str) -> str:
    low, high = bound
    if low is not None and high is not None:
        return f"{_fmt(low)}–{_fmt(high)} {unit}"
    if low is not None:
        return f"≥ {_fmt(low)} {unit}"
    if high is not None:
        return f"≤ {_fmt(high)} {unit}"
    return "not defined"


def _within(value: float, bound: Bound) -> bool:
    low, high = bound
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _direction(value: float, bound: Bound) -> str:
    low, high = bound
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    return "in_range"


def _resolve(spec: MarkerSpec, sex: Sex) -> Tuple[Bound, Bound, Bound]:
    override = spec.sex_overrides.get(sex.value)
    if override:
        return override
    return spec.optimal, spec.clinical, spec.critical


def _grade(value: float, spec: MarkerSpec, sex: Sex) -> Tuple[Severity, str, Bound, Bound]:
    optimal, clinical, critical = _resolve(spec, sex)
    if not _within(value, critical) and critical != (None, None):
        severity = Severity.high_concern
    elif not _within(value, clinical):
        severity = Severity.out_of_range
    elif not _within(value, optimal):
        severity = Severity.suboptimal
    else:
        severity = Severity.optimal
    reference = optimal if severity == Severity.suboptimal else clinical
    return severity, _direction(value, reference), optimal, clinical


def normalize_marker_name(name: str) -> Optional[str]:
    key = name.strip().lower()
    if key in MARKER_ALIASES:
        return MARKER_ALIASES[key]
    if key in BLOOD_SPECS:
        return key
    # Tolerate suffixes the aliases don't list, e.g. "HDL cholesterol, serum".
    for alias, canonical in MARKER_ALIASES.items():
        if key.startswith(alias):
            return canonical
    return None


def _convert(key: str, value: float, unit: str, spec: MarkerSpec) -> Tuple[float, bool]:
    """Return (value in the spec's unit, converted_ok)."""
    given = unit.strip().lower().replace(" ", "")
    expected = spec.unit.strip().lower().replace(" ", "")
    if given == expected or not given:
        return value, True
    factor = UNIT_CONVERSIONS.get((key, given))
    if factor is not None:
        return value * factor, True
    return value, False


def evaluate_blood_markers(
    markers: List[BloodMarker], sex: Sex
) -> Tuple[List[FlaggedMarker], List[dict]]:
    """Grade every recognized marker. Returns (graded, unrecognized)."""
    graded: List[FlaggedMarker] = []
    unrecognized: List[dict] = []

    for marker in markers:
        key = normalize_marker_name(marker.name)
        if key is None:
            unrecognized.append(
                {
                    "marker": marker.name,
                    "value": f"{_fmt(marker.value)} {marker.unit}".strip(),
                    "lab_reference": _lab_ref(marker),
                }
            )
            continue

        spec = BLOOD_SPECS[key]
        value, converted = _convert(key, marker.value, marker.unit, spec)
        if not converted:
            unrecognized.append(
                {
                    "marker": marker.name,
                    "value": f"{_fmt(marker.value)} {marker.unit}".strip(),
                    "lab_reference": _lab_ref(marker),
                    "note": f"Unit not convertible to our reference unit "
                            f"({spec.unit}); not graded.",
                }
            )
            continue

        severity, direction, optimal, clinical = _grade(value, spec, sex)
        notes: List[str] = []
        if marker.unit.strip().lower() != spec.unit.strip().lower() and marker.unit:
            notes.append(
                f"Converted from {_fmt(marker.value)} {marker.unit}."
            )
        lab_ref = _lab_ref(marker)
        if lab_ref:
            notes.append(f"Lab's own reference range: {lab_ref}.")
        if marker.collected_on:
            notes.append(f"Collected {marker.collected_on}.")

        graded.append(
            FlaggedMarker(
                marker=spec.canonical,
                category="blood",
                value=round(value, 4),
                unit=spec.unit,
                severity=severity,
                optimal_range=_describe(optimal, spec.unit),
                acceptable_range=_describe(clinical, spec.unit),
                aging_relevance=spec.aging_relevance,
                direction=direction,
                notes=notes,
            )
        )

    return graded, unrecognized


def _lab_ref(marker: BloodMarker) -> str:
    if marker.lab_ref_low is not None and marker.lab_ref_high is not None:
        return f"{_fmt(marker.lab_ref_low)}–{_fmt(marker.lab_ref_high)} {marker.unit}"
    if marker.lab_ref_high is not None:
        return f"≤ {_fmt(marker.lab_ref_high)} {marker.unit}"
    if marker.lab_ref_low is not None:
        return f"≥ {_fmt(marker.lab_ref_low)} {marker.unit}"
    return ""


def evaluate_wearables(profile: HealthProfile) -> List[FlaggedMarker]:
    wearables = profile.wearables
    if wearables is None:
        return []

    sex = profile.demographics.sex
    graded: List[FlaggedMarker] = []
    for field_name, spec in WEARABLE_SPECS.items():
        value = getattr(wearables, field_name, None)
        if value is None:
            continue
        severity, direction, optimal, clinical = _grade(value, spec, sex)
        notes: List[str] = []
        if wearables.days_of_data is not None:
            notes.append(f"Averaged over {wearables.days_of_data} days of data.")
            if wearables.days_of_data < 14:
                notes.append(
                    "Short history — treat this value as provisional."
                )
        graded.append(
            FlaggedMarker(
                marker=spec.canonical,
                category="wearable",
                value=round(float(value), 4),
                unit=spec.unit,
                severity=severity,
                optimal_range=_describe(optimal, spec.unit),
                acceptable_range=_describe(clinical, spec.unit),
                aging_relevance=spec.aging_relevance,
                direction=direction,
                notes=notes,
            )
        )
    return graded


def evaluate_methylation(profile: HealthProfile) -> List[FlaggedMarker]:
    """Grade epigenetic age acceleration, in years, against chronological age."""
    chrono = profile.demographics.chronological_age
    graded: List[FlaggedMarker] = []

    for result in profile.methylation:
        if result.clock == MethylationClock.dunedinpace:
            pace = result.predicted_age
            if pace > 1.20:
                severity, direction = Severity.high_concern, "high"
            elif pace > 1.05:
                severity, direction = Severity.out_of_range, "high"
            elif pace > 0.95:
                severity, direction = Severity.suboptimal, "high"
            else:
                severity, direction = Severity.optimal, "in_range"
            graded.append(
                FlaggedMarker(
                    marker="DunedinPACE",
                    category="methylation",
                    value=round(pace, 3),
                    unit="years of aging per calendar year",
                    severity=severity,
                    optimal_range="≤ 0.95",
                    acceptable_range="≤ 1.05",
                    aging_relevance="Rate of aging right now, rather than "
                                    "damage accumulated so far. Most responsive "
                                    "of the clocks to recent lifestyle change.",
                    direction=direction,
                )
            )
            continue

        acceleration = result.age_acceleration
        if acceleration is None:
            acceleration = result.predicted_age - chrono

        if acceleration > 7:
            severity, direction = Severity.high_concern, "high"
        elif acceleration > 3:
            severity, direction = Severity.out_of_range, "high"
        elif acceleration > 0:
            severity, direction = Severity.suboptimal, "high"
        else:
            severity, direction = Severity.optimal, "in_range"

        notes = [
            f"Clock predicts {_fmt(result.predicted_age)} years against a "
            f"chronological age of {_fmt(chrono)}."
        ]
        for label, value in (
            ("DNAm pack-years", result.dnam_pack_years),
            ("DNAm adrenomedullin", result.dnam_adm),
            ("DNAm PAI-1", result.dnam_pai1),
        ):
            if value is not None:
                notes.append(f"{label}: {_fmt(value)}.")

        graded.append(
            FlaggedMarker(
                marker=f"{result.clock.value} age acceleration",
                category="methylation",
                value=round(acceleration, 2),
                unit="years",
                severity=severity,
                optimal_range="≤ 0 years",
                acceptable_range="≤ 3 years",
                aging_relevance="Epigenetic age relative to chronological age. "
                                "GrimAge in particular is trained on mortality "
                                "and smoking/inflammation surrogates.",
                direction=direction,
                notes=notes,
            )
        )

    return graded


def evaluate_profile(
    profile: HealthProfile,
) -> Tuple[List[FlaggedMarker], List[dict]]:
    """Grade the whole profile, most severe first."""
    blood, unrecognized = evaluate_blood_markers(
        profile.blood_markers, profile.demographics.sex
    )
    graded = blood + evaluate_methylation(profile) + evaluate_wearables(profile)

    from .schemas import SEVERITY_ORDER

    graded.sort(key=lambda m: SEVERITY_ORDER[m.severity])
    return graded, unrecognized


def missing_high_value_markers(profile: HealthProfile) -> List[str]:
    """Markers we would want but did not receive — surfaced as caveats."""
    present = {
        normalize_marker_name(m.name) for m in profile.blood_markers
    }
    wanted = {
        "hs_crp": "hs-CRP",
        "hba1c": "HbA1c",
        "fasting_insulin": "fasting insulin",
        "apob": "ApoB",
        "albumin": "albumin",
        "vitamin_d": "vitamin D",
    }
    return [label for key, label in wanted.items() if key not in present]
