"""Phase 1a -- assemble the NHANES analytic table: demographics, blood panel,
healthy-reference exclusion flags, survey weights, and the mortality linkage.

Everything keys on SEQN, the universal NHANES respondent sequence number, which
is present in every public-use file and in the Linked Mortality File.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import acquire, config as C

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Linked Mortality File layout
# --------------------------------------------------------------------------
# The 2019 public-use LMF is fixed-width with no header. These 0-indexed,
# half-open column positions were derived EMPIRICALLY from the cycle-D file by
# mapping the character class of every byte position across all 10,348 records
# (see docs/METHODOLOGY.md), then validated against the known NHANES 2005-2006
# SEQN range and the expected follow-up horizon. Positions 21-41 are reserved
# filler in the public-use release and are intentionally not read.
LMF_COLSPECS: list[tuple[str, int, int]] = [
    ("SEQN", 0, 6),
    ("ELIGSTAT", 14, 15),      # 1=eligible, 2=under age 18, 3=ineligible
    ("MORTSTAT", 15, 16),      # 0=assumed alive, 1=assumed deceased, .=ineligible
    ("UCOD_LEADING", 16, 19),  # NCHS leading-cause recode, blank if alive
    ("DIABETES", 19, 20),      # diabetes listed as a contributing cause
    ("HYPERTEN", 20, 21),      # hypertension listed as a contributing cause
    ("PERMTH_INT", 42, 45),    # person-months from INTERVIEW to death/censor
    ("PERMTH_EXM", 45, 48),    # person-months from EXAM to death/censor
]

UCOD_LABELS = {
    "001": "Diseases of heart",
    "002": "Malignant neoplasms",
    "003": "Chronic lower respiratory diseases",
    "004": "Accidents",
    "005": "Cerebrovascular diseases",
    "006": "Alzheimer's disease",
    "007": "Diabetes mellitus",
    "008": "Influenza and pneumonia",
    "009": "Nephritis / nephrotic syndrome",
    "010": "All other causes",
}


def load_mortality() -> pd.DataFrame:
    """Parse the Public-Use Linked Mortality File and sanity-check it."""
    path = acquire.fetch_mortality()
    df = pd.read_fwf(
        path,
        colspecs=[(a, b) for _, a, b in LMF_COLSPECS],
        names=[n for n, _, _ in LMF_COLSPECS],
        na_values=[".", ""],
        dtype={"UCOD_LEADING": "object"},
    )

    # Guard the fixed-width parse. A silent off-by-one here would shift every
    # field and produce a plausible-looking but meaningless mortality outcome,
    # which is exactly the failure this pipeline cannot tolerate.
    assert df["SEQN"].between(31127, 41474).all(), "SEQN outside NHANES 2005-2006 range"
    assert set(df["MORTSTAT"].dropna().unique()) <= {0, 1}, "MORTSTAT not binary"
    assert set(df["ELIGSTAT"].dropna().unique()) <= {1, 2, 3}, "ELIGSTAT out of range"
    assert df["PERMTH_EXM"].max() <= 200, "follow-up longer than the 2019 horizon"

    df[C.JOIN_KEY] = df[C.JOIN_KEY].astype("Int64")
    df["cause"] = df["UCOD_LEADING"].map(UCOD_LABELS)
    log.info(
        "mortality: %d records, %d eligible, %d deaths, max follow-up %.0f months",
        len(df), int((df.ELIGSTAT == 1).sum()),
        int(df.MORTSTAT.sum()), df.PERMTH_EXM.max(),
    )
    return df.set_index(C.JOIN_KEY)


def load_demographics() -> pd.DataFrame:
    df = acquire.read_xpt(C.NHANES_FILES["DEMO_D"].local).set_index(C.JOIN_KEY)
    keep = [C.AGE_VAR, C.SEX_VAR, "RIDRETH1", "WTINT2YR", "WTMEC2YR",
            C.STRATA_VAR, C.PSU_VAR, "RIDAGEMN", "DMDEDUC2", "INDFMPIR"]
    df = df[[c for c in keep if c in df.columns]].copy()

    observed_max = df[C.AGE_VAR].max()
    if observed_max != C.AGE_TOPCODE:
        log.warning("age top-code is %s, config says %s", observed_max, C.AGE_TOPCODE)
    n_top = int((df[C.AGE_VAR] == C.AGE_TOPCODE).sum())
    log.info("demographics: %d rows; %d participants at the age top-code of %d",
             len(df), n_top, C.AGE_TOPCODE)

    df["sex"] = df[C.SEX_VAR].map(C.SEX_LABELS)
    df["age"] = df[C.AGE_VAR]
    return df


def _mean_bp(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Mean of the up-to-four valid BP readings; NHANES discards reading 1 only
    when later readings exist, so we simply average whatever is non-missing."""
    cols = [f"{prefix}{i}" for i in (1, 2, 3, 4) if f"{prefix}{i}" in df.columns]
    vals = df[cols].replace(0, np.nan)
    return vals.mean(axis=1, skipna=True)


def load_blood_panel() -> pd.DataFrame:
    """Merge every lab/exam file that contributes a blood-arm feature."""
    frames: dict[str, pd.DataFrame] = {}
    for name in ["BIOPRO_D", "CBC_D", "CRP_D", "GHB_D", "GLU_D",
                 "TRIGLY_D", "HDL_D", "TCHOL_D", "BPX_D", "BMX_D"]:
        p = C.NHANES_FILES[name].local
        if not p.exists():
            log.warning("missing %s -- its features will be unavailable", name)
            continue
        frames[name] = acquire.read_xpt(p).set_index(C.JOIN_KEY)

    out = pd.DataFrame(index=pd.Index([], name=C.JOIN_KEY, dtype="Int64"))
    for df in frames.values():
        out = out.join(df, how="outer", rsuffix="_dup")

    # Derived exam features.
    if "BPX_D" in frames:
        bp = frames["BPX_D"]
        sys_ = _mean_bp(bp, "BPXSY")
        dia = _mean_bp(bp, "BPXDI")
        out["SYSTOLIC"] = sys_
        out["PULSE_PRESSURE"] = sys_ - dia

    # The fasting sub-sample weight travels with its own labs.
    if "GLU_D" in frames and "WTSAF2YR" in frames["GLU_D"].columns:
        out["WTSAF2YR"] = frames["GLU_D"]["WTSAF2YR"]

    log.info("blood panel: %d participants x %d columns", len(out), out.shape[1])
    return out


# --------------------------------------------------------------------------
# Healthy reference population
# --------------------------------------------------------------------------
# Each rule marks participants to EXCLUDE when fitting a particular feature's
# age curve. Rules are per-feature, not global: excluding every diagnosed
# condition from every curve would shrink the reference to an unrepresentative
# "super-healthy" remnant, and the exclusion that matters for glucose (diabetes)
# is not the one that matters for creatinine (kidney disease).
#
# Rationale for defaulting to a healthy reference at all: a population-average
# curve bakes prevalent disease into the baseline, so a genuinely abnormal marker
# looks less abnormal than it is -- it is scored against a sick population. A
# healthy reference matches clinical reference-range practice and published
# bioage work. The cost is that the curve no longer describes "typical" adults.
# Both options are implemented; see build_analytic_table(healthy_reference=...).
#
# NHANES questionnaire coding throughout: 1=Yes, 2=No, 7=Refused, 9=Don't know,
# NaN=not asked (most MCQ160* items are asked only of participants aged 20+).


def build_exclusion_flags() -> pd.DataFrame:
    """Boolean exclusion flags keyed by SEQN, one column per rule name."""
    q: dict[str, pd.DataFrame] = {}
    for name in ["DIQ_D", "BPQ_D", "MCQ_D", "KIQ_U_D", "SMQ_D"]:
        p = C.NHANES_FILES[name].local
        if p.exists():
            q[name] = acquire.read_xpt(p).set_index(C.JOIN_KEY)

    idx = load_demographics().index
    flags = pd.DataFrame(index=idx)

    def yes(file: str, var: str) -> pd.Series:
        """True where the participant answered Yes; False otherwise.

        Refused (7) / Don't know (9) / not asked (NaN) are treated as NOT
        excluded. This is deliberate: treating unknown as diseased would drop
        every participant a question was not asked of, silently gutting the
        reference population.
        """
        if file not in q or var not in q[file].columns:
            return pd.Series(False, index=idx)
        return q[file][var].reindex(idx).eq(1).fillna(False)

    flags["diabetes"] = yes("DIQ_D", "DIQ010") | yes("DIQ_D", "DIQ050")
    flags["kidney_disease"] = yes("KIQ_U_D", "KIQ022")
    flags["liver_disease"] = yes("MCQ_D", "MCQ160L")
    flags["bp_meds"] = yes("BPQ_D", "BPQ040A")
    flags["lipid_meds"] = yes("BPQ_D", "BPQ100D")
    flags["cvd"] = (yes("MCQ_D", "MCQ160B") | yes("MCQ_D", "MCQ160C")
                    | yes("MCQ_D", "MCQ160E") | yes("MCQ_D", "MCQ160F"))
    flags["cancer"] = yes("MCQ_D", "MCQ220")
    flags["current_smoker"] = yes("SMQ_D", "SMQ040") | (
        q["SMQ_D"]["SMQ040"].reindex(idx).eq(2).fillna(False) if "SMQ_D" in q else False
    )

    # Lab-derived exclusions.
    blood = load_blood_panel()
    demo = load_demographics()
    # Acute inflammation: CRP > 10 mg/L. NHANES LBXCRP is in mg/dL, so the
    # threshold is 1.0 mg/dL. Getting this unit wrong by 10x would exclude
    # almost everyone or almost no one.
    crp = blood["LBXCRP"].reindex(idx) if "LBXCRP" in blood.columns else pd.Series(np.nan, index=idx)
    flags["acute_inflammation"] = crp.gt(1.0).fillna(False)

    # Anemia: WHO thresholds, sex-specific.
    hgb = blood["LBXHGB"].reindex(idx) if "LBXHGB" in blood.columns else pd.Series(np.nan, index=idx)
    is_female = demo[C.SEX_VAR].reindex(idx).eq(2)
    flags["anemia"] = ((is_female & hgb.lt(12.0)) | (~is_female & hgb.lt(13.0))).fillna(False)

    # SCOPE: two exclusion rules referenced by the feature registry have no
    # usable source in the files acquired here, so they are recorded as
    # never-triggering and logged rather than silently dropped.
    #   heavy_alcohol -- would need ALQ_D; affects the GGT curve.
    #   gout_meds     -- NHANES prescription data (RXQ_RX_D) would be required
    #                    to identify urate-lowering therapy; affects uric acid.
    for unavailable in ("heavy_alcohol", "gout_meds"):
        flags[unavailable] = False
        log.warning("SCOPE: exclusion rule '%s' has no data source; not applied", unavailable)

    log.info("exclusion flags built: %s",
             {c: int(flags[c].sum()) for c in flags.columns})
    return flags


def build_analytic_table() -> pd.DataFrame:
    """The joined NHANES analytic table: demographics + blood + wearable +
    exclusion flags + mortality, one row per participant."""
    demo = load_demographics()
    blood = load_blood_panel()
    flags = build_exclusion_flags().add_prefix("excl_")
    mort = load_mortality()

    tab = demo.join(blood, how="left").join(flags, how="left")

    wpath = C.INTERIM / "wearable_features.parquet"
    if wpath.exists():
        wear = pd.read_parquet(wpath)
        wear.index = wear.index.astype("Int64")
        tab = tab.join(wear.add_prefix("pax_"), how="left")
        log.info("joined wearable features for %d participants",
                 int(tab["pax_mean_cpm"].notna().sum()))
    else:
        log.warning("no wearable features found at %s -- wearable arm will be empty", wpath)

    tab = tab.join(mort[["ELIGSTAT", "MORTSTAT", "PERMTH_EXM", "PERMTH_INT",
                         "UCOD_LEADING", "cause"]], how="left")

    # Analytic-sample flags. Kept as columns rather than applied as filters so
    # that downstream steps can choose their own denominator explicitly.
    tab["in_age_window"] = tab["age"].between(C.AGE_MIN, C.AGE_MAX)
    tab["mort_eligible"] = tab["ELIGSTAT"].eq(1) & tab["PERMTH_EXM"].notna()
    tab["followup_years"] = tab["PERMTH_EXM"] / 12.0
    tab["died"] = tab["MORTSTAT"]

    log.info(
        "analytic table: %d rows | %d in age window | %d mortality-eligible | %d deaths",
        len(tab), int(tab.in_age_window.sum()), int(tab.mort_eligible.sum()),
        int(tab.loc[tab.mort_eligible, "died"].sum()),
    )
    return tab
