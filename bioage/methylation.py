"""Phase 3 -- the methylation arm, via published clock coefficients.

WHY NOT BUILD A CLOCK
---------------------
Per-feature curve inversion does not scale to ~473,000 CpG sites, and refitting
an epigenetic clock is a research programme, not a pipeline stage. Published
clocks have public coefficients; this arm applies them.

CLOCK CHOICE
------------
First-generation clocks (Horvath, Hannum) were trained to predict chronological
age and are very good at it -- which is exactly the problem here. A near-perfect
chronological predictor has a near-zero residual, and the residual IS the
product. Their gap is mostly measurement noise.

Second-generation clocks (PhenoAge, GrimAge) were trained toward mortality and
morbidity, so their gap is meaningful by construction. That also makes them
consistent with the blood and wearable arms, which are likewise weighted toward
an outcome rather than toward age.

GrimAge is primary (strongest mortality-prediction record head-to-head);
PhenoAge is secondary (clinically interpretable inputs, so it is easier to
explain WHY a score moved). First-generation clocks are computed anyway, purely
as a contrast -- `residual_spread_table` quantifies the argument above rather
than asserting it.

# SCOPE: THIS ARM USES A DIFFERENT COHORT.
NHANES has never collected DNA methylation in any public-use cycle, so there is
no SEQN to join on and no honest way to merge this arm with the other two at the
individual level. It is therefore analysed on its own cohort and reported as a
demonstration arm. No cross-cohort join is fabricated anywhere in this codebase.

# SCOPE: GSE40279 CARRIES NO MORTALITY LINKAGE.
The combiner is specified to fit against mortality. This cohort has chronological
age only -- no vital status, no follow-up time. Per the project rule that a
missing mortality target must be flagged rather than silently replaced with a
chronological-age target, this arm gets NO fitted combiner weight. It is
reported in the per-modality breakdown and excluded from the combined score
unless a caller explicitly opts in to a clearly-labelled provisional weight.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

PRIMARY_CLOCK = "GrimAgeV2"
SECONDARY_CLOCK = "PhenoAge"
# Computed for contrast only -- see residual_spread_table.
FIRST_GEN_CLOCKS = ("Horvathv1", "Hannum")
CLOCKS = (PRIMARY_CLOCK, "GrimAgeV1", SECONDARY_CLOCK, *FIRST_GEN_CLOCKS)


def load_geo(accession: str = C.METHYLATION_COHORT):
    """Load a GEO methylation dataset through Biolearn's data library."""
    from biolearn.data_library import DataLibrary

    log.info("loading %s via Biolearn (large download on first run)", accession)
    data = DataLibrary().get(accession).load()
    log.info("%s: dnam %s, metadata columns %s",
             accession, data.dnam.shape, list(data.metadata.columns))
    return data


def run_clocks(data, clocks: tuple[str, ...] = CLOCKS) -> pd.DataFrame:
    """Apply each published clock and return one column of predicted age each.

    GrimAge additionally requires chronological age and sex as model inputs --
    it is not a pure CpG -> age map. Biolearn raises if either is absent, so a
    missing-metadata failure is loud rather than silent.
    """
    from biolearn.model_gallery import ModelGallery

    gallery = ModelGallery()
    out = pd.DataFrame(index=data.metadata.index)
    for name in clocks:
        try:
            # Biolearn returns two DIFFERENT shapes and picking the first column
            # blindly is a silent, plausible-looking error:
            #   linear clocks -> single column "Predicted"
            #   GrimAge       -> wide frame whose FIRST column is "DNAmADM", a
            #                    plasma-protein sub-clock, not an age at all.
            #                    The age estimate is "DNAmGrimAge".
            # Selecting by name rather than position is load-bearing.
            pred = gallery.get(name).predict(data)
            if isinstance(pred, pd.DataFrame):
                for cand in ("DNAmGrimAge", "Predicted"):
                    if cand in pred.columns:
                        col = pred[cand]
                        break
                else:
                    raise KeyError(
                        f"no recognised age column in {list(pred.columns)[:6]}"
                    )
            else:
                col = pred
            out[name] = pd.to_numeric(col, errors="coerce").reindex(out.index)
            log.info("  %-12s ok   mean=%.1f sd=%.1f n=%d",
                     name, out[name].mean(), out[name].std(), out[name].notna().sum())
        except Exception as exc:
            log.warning("  %-12s FAILED: %s", name, exc)
    return out


def build_methylation_table(accession: str = C.METHYLATION_COHORT) -> pd.DataFrame:
    """Predicted ages, chronological age, and gaps for the methylation cohort."""
    data = load_geo(accession)
    preds = run_clocks(data)

    meta = data.metadata
    if "age" not in meta.columns:
        raise RuntimeError(f"{accession} has no chronological age; cannot form gaps")

    tab = preds.copy()
    tab["age"] = pd.to_numeric(meta["age"], errors="coerce")
    if "sex" in meta.columns:
        # Biolearn codes sex 0/1; GrimAge treats 0 as female (see GrimageModel).
        tab["sex"] = meta["sex"].map({0: "female", 1: "male"}).fillna("unknown")
    else:
        tab["sex"] = "unknown"

    for clock in preds.columns:
        tab[f"gap_{clock}"] = tab[clock] - tab["age"]

    log.info("methylation cohort %s: n=%d, age %.0f-%.0f",
             accession, len(tab), tab["age"].min(), tab["age"].max())
    return tab


def residual_spread_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Compare clocks by age correlation and surviving residual spread.

    For each clock: correlation with chronological age, MAE against it, the SD
    of its gap, and how much survives removing the age trend.

    WHAT THIS CAN AND CANNOT SETTLE. The motivation for preferring
    second-generation clocks is that first-generation ones were trained to
    predict chronological age, so little residual is left to combine. This table
    was built to check that argument -- and on GSE40279 it does NOT cleanly
    support it: Hannum (1st gen) leaves residual SD 4.25 while GrimAgeV1 (2nd
    gen) leaves 3.93, the opposite ordering.

    The reason is that residual SPREAD conflates signal with noise. A clock that
    is merely imprecise also has a wide residual; PhenoAge's 6.96 here is the
    largest of all five and cannot be read as the most information. Separating
    the two requires regressing the residual on an OUTCOME, and GSE40279 carries
    chronological age only -- no vital status. So the generational argument is
    adopted here on the published mortality evidence, not on this cohort, and
    this table is reported as a descriptive comparison rather than as support
    for the clock choice. See the second `# SCOPE` note above.
    """
    rows = []
    for clock in [c for c in CLOCKS if c in tab.columns]:
        g = tab[f"gap_{clock}"]
        m = g.notna() & tab["age"].notna()
        if m.sum() < 30:
            continue
        resid = g[m] - np.poly1d(np.polyfit(tab["age"][m], g[m], 1))(tab["age"][m])
        rows.append(dict(
            clock=clock,
            generation="2nd" if clock not in FIRST_GEN_CLOCKS else "1st",
            n=int(m.sum()),
            r_with_chron_age=float(tab[clock][m].corr(tab["age"][m])),
            mae_vs_chron_age=float(np.abs(g[m]).mean()),
            gap_sd=float(g[m].std()),
            residual_gap_sd=float(resid.std()),
        ))
    return pd.DataFrame(rows).sort_values("generation")
