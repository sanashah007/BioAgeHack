"""Central configuration: paths, cohort choice, file registry, feature registry.

COHORT CHOICE
-------------
NHANES cycle D (2005-2006) is used for ALL THREE NHANES-derived arms.

Rationale (this is the single most consequential acquisition decision):
  * The Public-Use Linked Mortality File follows participants through 2019-12-31.
    An older cycle therefore buys more follow-up years and more deaths, which is
    what the combiner in Phase 4 is actually fit against. Cycle D gives ~14 years.
  * Cycle D is one of only two cycles (C=2003-2004, D=2005-2006) carrying the
    ActiGraph AM-7164 accelerometer sub-study AND a full biochemistry panel.
    Using one cycle means blood, wearable and mortality all key on the same SEQN
    with no cross-cycle weight rescaling. Cycle C would work too; D is preferred
    because its lab panel is more complete.
  * Methylation cannot join here at all -- see METHYLATION_COHORT below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"

for _p in (RAW, INTERIM, PROCESSED, OUTPUTS, FIGURES, TABLES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Cohort / source URLs
# --------------------------------------------------------------------------
CYCLE = "D"
CYCLE_YEARS = "2005-2006"
CYCLE_START_YEAR = 2005

# Confirmed-working base path (CDC reorganised NHANES URLs; the older
# /Nchs/Nhanes/<years>/<FILE>.XPT form now serves an HTML landing page).
NHANES_BASE = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{CYCLE_START_YEAR}/DataFiles"
NHANES_DOC_BASE = f"https://wwwn.cdc.gov/Nchs/Nhanes/{CYCLE_YEARS}"

LMF_BASE = (
    "https://ftp.cdc.gov/pub/HEALTH_STATISTICS/NCHS/datalinkage/linked_mortality"
)
LMF_FILE = f"NHANES_{CYCLE_YEARS.replace('-', '_')}_MORT_2019_PUBLIC.dat"

# Methylation runs on a DIFFERENT cohort. NHANES has never collected DNA
# methylation in any public-use cycle, so there is no join to make. See
# docs/METHODOLOGY.md; Phase 4 treats this arm as a separate analysis.
METHYLATION_COHORT = "GSE40279"


@dataclass(frozen=True)
class NhanesFile:
    """One NHANES public-use data file."""

    name: str  # e.g. "BIOPRO_D"
    kind: str  # demographics | lab | exam | questionnaire | activity
    weight: str | None  # the survey weight variable to use WITH this file
    note: str = ""

    @property
    def url(self) -> str:
        ext = "ZIP" if self.kind == "activity" else "xpt"
        return f"{NHANES_BASE}/{self.name}.{ext}"

    @property
    def doc_url(self) -> str:
        return f"{NHANES_DOC_BASE}/{self.name}.htm"

    @property
    def local(self) -> Path:
        ext = "zip" if self.kind == "activity" else "xpt"
        return RAW / "nhanes" / f"{self.name}.{ext}"


# WEIGHT ASSIGNMENT
# -----------------
# NHANES oversamples by design (older adults, several race/ethnicity groups,
# low-income households). Unweighted means are biased toward the oversampled
# strata, so every population curve in Phase 2 must be weighted.
#
# Which weight is correct depends on the *narrowest* sub-sample a variable was
# measured in, not on the file it happens to ship in:
#   WTINT2YR  - interview-only variables
#   WTMEC2YR  - anything requiring the mobile exam centre (most labs)
#   WTSAF2YR  - the morning FASTING sub-sample (glucose, insulin, triglycerides,
#               LDL). Using WTMEC2YR on these over-counts non-fasting people who
#               have no value, biasing the curve.
# Verified per-file against each component's "Analytic Notes" documentation.
NHANES_FILES: dict[str, NhanesFile] = {
    f.name: f
    for f in [
        NhanesFile("DEMO_D", "demographics", "WTMEC2YR",
                   "age, sex, race, WTINT2YR/WTMEC2YR, SDMVSTRA, SDMVPSU"),
        # --- blood chemistry -------------------------------------------------
        NhanesFile("BIOPRO_D", "lab", "WTMEC2YR", "standard biochemistry profile"),
        NhanesFile("CBC_D", "lab", "WTMEC2YR", "complete blood count"),
        NhanesFile("CRP_D", "lab", "WTMEC2YR", "C-reactive protein"),
        NhanesFile("GHB_D", "lab", "WTMEC2YR", "glycohemoglobin (HbA1c)"),
        NhanesFile("TCHOL_D", "lab", "WTMEC2YR", "total cholesterol"),
        NhanesFile("HDL_D", "lab", "WTMEC2YR", "HDL cholesterol"),
        NhanesFile("GLU_D", "lab", "WTSAF2YR", "FASTING sub-sample -- fasting weight"),
        NhanesFile("TRIGLY_D", "lab", "WTSAF2YR", "FASTING sub-sample -- fasting weight"),
        # --- exam ------------------------------------------------------------
        NhanesFile("BMX_D", "exam", "WTMEC2YR", "body measures (BMI, waist)"),
        NhanesFile("BPX_D", "exam", "WTMEC2YR", "blood pressure"),
        # --- questionnaire (used only to define the healthy reference) --------
        NhanesFile("DIQ_D", "questionnaire", "WTINT2YR", "diabetes diagnosis"),
        NhanesFile("BPQ_D", "questionnaire", "WTINT2YR", "hypertension / chol meds"),
        NhanesFile("MCQ_D", "questionnaire", "WTINT2YR", "cancer, MI, stroke, CHF"),
        NhanesFile("SMQ_D", "questionnaire", "WTINT2YR", "smoking status"),
        NhanesFile("KIQ_U_D", "questionnaire", "WTINT2YR", "kidney conditions"),
        # --- accelerometer ---------------------------------------------------
        NhanesFile("PAXRAW_D", "activity", "WTMEC2YR",
                   "minute-level ActiGraph counts, ~471 MB zipped"),
    ]
}

JOIN_KEY = "SEQN"

# --------------------------------------------------------------------------
# Demographics
# --------------------------------------------------------------------------
AGE_VAR = "RIDAGEYR"
SEX_VAR = "RIAGENDR"  # 1 = Male, 2 = Female
SEX_LABELS = {1: "male", 2: "female"}
STRATA_VAR = "SDMVSTRA"
PSU_VAR = "SDMVPSU"

# RIDAGEYR in cycle D is TOP-CODED at 85: every participant aged 85+ is recorded
# as exactly 85. Age curves are therefore meaningless at/above the top-code --
# the "85" bin is a mixture of 85- and 100-year-olds, which flattens any real
# trend and corrupts inversion at the upper tail. Verified empirically in
# scripts/01_download.py, which asserts the observed max equals this value.
AGE_TOPCODE = 85

# Analysis age window. Lower bound: biomarker-age relationships are dominated by
# growth/puberty below ~20 and the adult aging trend does not extend there.
# Upper bound: strictly below the top-code.
AGE_MIN = 20
AGE_MAX = 84


@dataclass(frozen=True)
class Feature:
    """One biomarker used to build an age curve.

    exclude_if: healthy-reference exclusions applied ONLY when building this
        feature's reference curve. Per-feature rather than global -- excluding
        every diagnosed condition from every curve would shrink the reference
        population to the point of unrepresentativeness, and the exclusion that
        matters for glucose (diabetes) is not the one that matters for
        creatinine (kidney disease).
    expect: prior on the direction of the adult age trend, from the literature.
        Used only to flag surprises during monotonicity screening -- the screen
        itself is empirical and does not trust this field.
    """

    name: str
    var: str
    file: str
    label: str
    units: str
    expect: str  # "increase" | "decrease" | "nonmonotonic"
    exclude_if: tuple[str, ...] = ()
    log_transform: bool = False
    valid_range: tuple[float, float] | None = None


# Blood panel. Chosen for (a) documented adult age trends, (b) overlap with the
# PhenoAge biomarker set, (c) full availability in cycle D.
BLOOD_FEATURES: list[Feature] = [
    Feature("albumin", "LBXSAL", "BIOPRO_D", "Albumin", "g/dL", "decrease",
            exclude_if=("liver_disease",), valid_range=(1.5, 6.0)),
    Feature("creatinine", "LBXSCR", "BIOPRO_D", "Creatinine", "mg/dL", "increase",
            exclude_if=("kidney_disease",), log_transform=True, valid_range=(0.2, 15.0)),
    Feature("alk_phos", "LBXSAPSI", "BIOPRO_D", "Alkaline phosphatase", "IU/L", "increase",
            exclude_if=("liver_disease",), log_transform=True, valid_range=(10, 500)),
    Feature("urea_nitrogen", "LBXSBU", "BIOPRO_D", "Blood urea nitrogen", "mg/dL", "increase",
            exclude_if=("kidney_disease",), log_transform=True, valid_range=(2, 100)),
    Feature("uric_acid", "LBXSUA", "BIOPRO_D", "Uric acid", "mg/dL", "increase",
            exclude_if=("kidney_disease", "gout_meds"), valid_range=(1.0, 15.0)),
    Feature("total_protein", "LBXSTP", "BIOPRO_D", "Total protein", "g/dL", "decrease",
            valid_range=(4.0, 10.0)),
    Feature("ggt", "LBXSGTSI", "BIOPRO_D", "Gamma glutamyl transferase", "IU/L", "nonmonotonic",
            exclude_if=("liver_disease", "heavy_alcohol"), log_transform=True,
            valid_range=(3, 800)),
    Feature("alt", "LBXSATSI", "BIOPRO_D", "Alanine aminotransferase", "IU/L", "nonmonotonic",
            exclude_if=("liver_disease",), log_transform=True, valid_range=(3, 400)),
    Feature("crp", "LBXCRP", "CRP_D", "C-reactive protein", "mg/dL", "increase",
            exclude_if=("acute_inflammation",), log_transform=True, valid_range=(0.01, 20.0)),
    Feature("wbc", "LBXWBCSI", "CBC_D", "White blood cell count", "1000 cells/uL", "nonmonotonic",
            exclude_if=("acute_inflammation",), valid_range=(1.5, 30.0)),
    Feature("lymph_pct", "LBXLYPCT", "CBC_D", "Lymphocyte percent", "%", "decrease",
            exclude_if=("acute_inflammation",), valid_range=(2.0, 80.0)),
    Feature("rdw", "LBXRDW", "CBC_D", "Red cell distribution width", "%", "increase",
            exclude_if=("anemia",), valid_range=(9.0, 30.0)),
    Feature("mcv", "LBXMCVSI", "CBC_D", "Mean cell volume", "fL", "increase",
            exclude_if=("anemia",), valid_range=(55, 125)),
    Feature("hemoglobin", "LBXHGB", "CBC_D", "Hemoglobin", "g/dL", "decrease",
            exclude_if=("anemia",), valid_range=(6.0, 20.0)),
    Feature("hba1c", "LBXGH", "GHB_D", "Glycohemoglobin", "%", "increase",
            exclude_if=("diabetes",), log_transform=True, valid_range=(3.0, 18.0)),
    Feature("glucose_fasting", "LBXGLU", "GLU_D", "Fasting glucose", "mg/dL", "increase",
            exclude_if=("diabetes",), log_transform=True, valid_range=(40, 500)),
    Feature("total_chol", "LBXTC", "TCHOL_D", "Total cholesterol", "mg/dL", "nonmonotonic",
            exclude_if=("lipid_meds",), valid_range=(70, 500)),
    Feature("hdl", "LBDHDD", "HDL_D", "HDL cholesterol", "mg/dL", "nonmonotonic",
            exclude_if=("lipid_meds",), valid_range=(10, 150)),
    Feature("triglycerides", "LBXTR", "TRIGLY_D", "Triglycerides", "mg/dL", "nonmonotonic",
            exclude_if=("lipid_meds",), log_transform=True, valid_range=(10, 1500)),
    Feature("systolic_bp", "SYSTOLIC", "BPX_D", "Systolic blood pressure", "mmHg", "increase",
            exclude_if=("bp_meds",), valid_range=(70, 260)),
    Feature("pulse_pressure", "PULSE_PRESSURE", "BPX_D", "Pulse pressure", "mmHg", "increase",
            exclude_if=("bp_meds",), valid_range=(10, 150)),
    Feature("waist", "BMXWAIST", "BMX_D", "Waist circumference", "cm", "nonmonotonic",
            valid_range=(50, 200)),
]

# Wearable features engineered from PAXRAW_D minute-level counts.
#
# SCOPE: NHANES accelerometry is a WEAK proxy for Apple Health / consumer-wearable
# input. The 2005-2006 ActiGraph AM-7164 is a uniaxial, counts-only device worn at
# the waist during waking hours for 7 days. It yields NO heart-rate variability,
# NO resting heart rate trend, NO sleep staging, NO SpO2 -- precisely the channels
# that carry the most aging signal in modern consumer wearables. What remains is
# volume, intensity distribution and rhythm/fragmentation. Expect this arm to be
# materially weaker than a real wearable arm would be; that is a limitation of
# public data, not of the method. Documented in docs/METHODOLOGY.md.
WEARABLE_FEATURES: list[Feature] = [
    Feature("mean_cpm", "mean_cpm", "PAX", "Mean counts per wear-minute", "counts/min",
            "decrease", log_transform=True),
    Feature("mvpa_min", "mvpa_min_per_day", "PAX", "Moderate-vigorous activity", "min/day",
            "decrease", log_transform=True),
    Feature("sedentary_frac", "sedentary_frac", "PAX", "Sedentary fraction of wear time",
            "fraction", "increase"),
    Feature("activity_cv", "activity_cv", "PAX", "Within-day activity variability", "CV",
            "nonmonotonic"),
    Feature("m10", "m10", "PAX", "Most-active 10h mean", "counts/min", "decrease",
            log_transform=True),
    Feature("l5", "l5", "PAX", "Least-active 5h mean", "counts/min", "nonmonotonic",
            log_transform=True),
    Feature("relative_amplitude", "relative_amplitude", "PAX", "Relative amplitude (M10 vs L5)",
            "ratio", "decrease"),
    Feature("intradaily_variability", "intradaily_variability", "PAX",
            "Intradaily variability (fragmentation)", "IV", "increase"),
    Feature("interdaily_stability", "interdaily_stability", "PAX",
            "Interdaily stability (rhythm regularity)", "IS", "nonmonotonic"),
    Feature("activity_bout_frag", "activity_bout_frag", "PAX",
            "Active-bout fragmentation rate", "1/min", "increase"),
]

FEATURES_BY_MODALITY: dict[str, list[Feature]] = {
    "blood": BLOOD_FEATURES,
    "wearable": WEARABLE_FEATURES,
}

MODALITIES = ("blood", "wearable", "methylation")

# --------------------------------------------------------------------------
# Curve-fitting parameters
# --------------------------------------------------------------------------
LOESS_FRAC = 0.45          # LOESS bandwidth; wide because tails are sparse
CURVE_AGE_GRID_STEP = 0.5  # years, resolution of the fitted reference curve
MIN_N_PER_SEX = 200        # refuse to fit a curve on fewer observations
# Share of a curve's TOTAL VARIATION that may travel against the dominant
# direction before it is judged non-monotonic. Magnitude-weighted, not step-
# counted -- see curves.screen_monotonic for why. 5% tolerates the small dip a
# LOESS fit puts in an otherwise monotone marker without waving through a curve
# that genuinely turns over.
MONOTONIC_TOL = 0.05
MIN_MONOTONIC_SPAN = 25    # years; a usable monotonic segment must be >= this

RANDOM_SEED = 20260823
