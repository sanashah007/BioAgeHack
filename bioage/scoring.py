"""Phases 2.7 and 4 -- outcome-weighted modality bioages, gaps, and the combiner.

THE CENTRAL RULE
----------------
Never combine raw predicted ages, and never fit any combination step against
chronological age. Averaging several good chronological-age predictors just
re-derives the birthdate: the better each component gets, the closer the average
gets to the age we already knew, and the residual -- which is the entire product
-- goes to zero. Every combination in this module therefore operates on GAPS
(predicted minus chronological) and is fitted against linked mortality.

Two combination stages, both outcome-fitted:
  1. within modality  -- per-feature implied ages -> one modality bioage,
                         weighted by each feature's own mortality association
  2. across modality  -- modality gaps -> one combined gap, weighted by an
                         elastic net / Cox model fitted to time-to-death

CROSS-FITTING
-------------
Both stages learn their weights from the same mortality outcome the pipeline is
later validated against. Scoring a participant with weights fitted on data that
included them leaks the outcome and inflates every validation metric. All
reported scores are therefore OUT-OF-FOLD: weights are fitted on K-1 folds and
applied to the held-out fold, so no participant is ever scored by a model that
saw their death.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import config as C

log = logging.getLogger(__name__)

N_FOLDS = 5
MIN_EVENTS_PER_FEATURE = 15  # refuse to weight a feature on too few deaths


# --------------------------------------------------------------------------
# Stage 1: per-feature gaps -> modality bioage
# --------------------------------------------------------------------------
def feature_mortality_weights(
    gaps: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    min_events: int = MIN_EVENTS_PER_FEATURE,
) -> pd.Series:
    """Weight each feature by how strongly its own gap predicts death.

    For each feature, fit

        Cox(time = follow-up, event = died) ~ gap_feature + age + sex

    and take the z-statistic on the gap term. Chronological age MUST be in the
    model. Without it the weights are contaminated by age itself: an older
    participant is more likely to die for reasons that have nothing to do with
    their gap, and any feature whose gap correlates with age would look
    predictive purely through that channel.

    Weights are clipped at zero. A negative coefficient means the feature's gap
    runs the wrong way -- an older implied age predicting LOWER mortality --
    which is a sign of an inverted or noisy curve, not of a protective effect,
    so the feature is dropped rather than allowed to subtract.
    """
    rows = {}
    for feat in gaps.columns:
        m = gaps[feat].notna() & outcome["died"].notna() & outcome["time"].gt(0)
        n_ev = int(outcome.loc[m, "died"].sum())
        if m.sum() < 100 or n_ev < min_events:
            rows[feat] = dict(z=0.0, hr=np.nan, n=int(m.sum()), events=n_ev,
                              note="insufficient events")
            continue
        d = pd.DataFrame({
            "T": outcome.loc[m, "time"].to_numpy(float),
            "E": outcome.loc[m, "died"].to_numpy(int),
            "gap": gaps.loc[m, feat].to_numpy(float),
            "age": outcome.loc[m, "age"].to_numpy(float),
            "female": (outcome.loc[m, "sex"] == "female").astype(int).to_numpy(),
        })
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(d, duration_col="T", event_col="E")
            z = float(cph.summary.loc["gap", "z"])
            hr = float(cph.summary.loc["gap", "exp(coef)"])
            rows[feat] = dict(z=z, hr=hr, n=int(m.sum()), events=n_ev, note="")
        except Exception as exc:
            rows[feat] = dict(z=0.0, hr=np.nan, n=int(m.sum()), events=n_ev,
                              note=f"fit failed: {exc}")

    info = pd.DataFrame(rows).T
    w = info["z"].astype(float).clip(lower=0.0)
    if w.sum() <= 0:
        log.warning("no feature gap predicts mortality; falling back to equal weights")
        w = pd.Series(1.0, index=gaps.columns)
    w = w / w.sum()
    feature_mortality_weights.last_info = info.assign(weight=w)  # type: ignore[attr-defined]
    return w


def modality_bioage(
    gaps: pd.DataFrame, weights: pd.Series, age: pd.Series, *, min_features: int = 3
) -> pd.Series:
    """Weighted mean of per-feature gaps, re-expressed as an age.

    Weights are renormalised per participant over the features they actually
    have, so someone missing a lab is scored on what is present rather than
    having the missing feature silently treated as a zero gap.
    """
    g = gaps.reindex(columns=weights.index)
    w = weights.to_numpy(float)[None, :]
    present = g.notna().to_numpy()
    wmat = np.where(present, w, 0.0)
    denom = wmat.sum(axis=1)
    num = np.nansum(np.where(present, g.to_numpy(float), 0.0) * wmat, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_gap = np.where(denom > 0, num / denom, np.nan)
    mean_gap[present.sum(axis=1) < min_features] = np.nan
    return pd.Series(age.to_numpy(float) + mean_gap, index=gaps.index)


# --------------------------------------------------------------------------
# Bias correction
# --------------------------------------------------------------------------
def split_half_reliability(
    gaps_feat: pd.DataFrame, weights: pd.Series, *, n_splits: int = 25,
    seed: int = C.RANDOM_SEED,
) -> float:
    """Estimate how much of the modality gap is signal rather than noise.

    Split the feature set in half at random, score each half independently, and
    correlate the two resulting gaps. Two halves of the same person's biology
    should agree; whatever they disagree about is measurement noise. The
    Spearman-Brown correction converts the half-length correlation into the
    reliability of the full-length score.
    """
    feats = [f for f in weights.index if f in gaps_feat.columns and weights[f] > 0]
    if len(feats) < 4:
        return 1.0
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(feats)
        a, b = list(perm[: len(perm) // 2]), list(perm[len(perm) // 2:])
        ga = (gaps_feat[a] * weights[a]).sum(axis=1) / gaps_feat[a].notna().mul(weights[a]).sum(axis=1)
        gb = (gaps_feat[b] * weights[b]).sum(axis=1) / gaps_feat[b].notna().mul(weights[b]).sum(axis=1)
        m = ga.notna() & gb.notna()
        if m.sum() > 50:
            r = ga[m].corr(gb[m])
            if np.isfinite(r):
                rs.append(r)
    if not rs:
        return 1.0
    r_half = float(np.median(rs))
    # Spearman-Brown: a half-length test is less reliable than the full test.
    rel = (2 * r_half) / (1 + r_half) if r_half > 0 else 0.0
    return float(np.clip(rel, 0.0, 1.0))


def shrink_to_reliability(gap: pd.Series, reliability: float) -> pd.Series:
    """Shrink a gap toward the population mean in proportion to its noise.

    An unshrunk gap is reported at full spread as if it were measured exactly.
    It is not: with reliability rho, the observed spread is inflated by
    measurement error, which is why raw inversion happily reports a sedentary
    20-year-old as "wearable age 84". Multiplying the centred gap by rho is the
    standard regression-to-the-mean correction and yields the best least-squares
    estimate of the person's true gap.

    This is a monotone linear rescaling, so it changes NO validation result --
    hazard ratios per SD, C-index and AUC are all invariant. It only makes the
    number that gets shown to a person defensible.
    """
    if not np.isfinite(reliability) or reliability >= 1.0:
        return gap
    mu = gap.mean(skipna=True)
    return (gap - mu) * reliability + mu


def calibrate_to_age_scale(
    gap: pd.Series, outcome: pd.DataFrame, *, label: str = ""
) -> tuple[pd.Series, float]:
    """Rescale a gap into CHRONOLOGICAL-AGE-EQUIVALENT YEARS of mortality risk.

    Curve inversion returns an implied age on whatever scale the marker happens
    to have, and that scale is arbitrary across modalities. Activity varies far
    more between people of the same age than it does between ages, so inverting
    it maps ordinary differences onto enormous age swings -- the raw wearable
    arm will happily report a fit 68-year-old as 30. The number is not
    meaningless, but it is not in years either, and putting it beside a blood
    gap that IS roughly in years invites a false comparison.

    The fix is the one PhenoAge uses: fit a mortality model, then convert back
    to an age equivalent. With

        Cox(death) ~ b_gap * gap + b_age * age + sex

    the ratio b_gap / b_age is how many years of chronological age carry the
    same hazard as one unit of gap. Multiplying by it puts every modality on one
    honest scale, where "+5 years" means "the mortality risk of someone five
    years older" -- which is the only definition of a bioage gap that is
    comparable across arms.

    This is a positive linear rescaling, so it leaves every validation metric
    (HR per SD, C-index, AUC) untouched. It changes only what the number means.
    """
    m = gap.notna() & outcome["died"].notna() & outcome["time"].gt(0)
    if m.sum() < 100 or int(outcome.loc[m, "died"].sum()) < 20:
        log.warning("  %s: too little data to calibrate; leaving raw scale", label)
        return gap, 1.0
    d = pd.DataFrame({
        "T": outcome.loc[m, "time"].to_numpy(float),
        "E": outcome.loc[m, "died"].astype(int).to_numpy(),
        "gap": gap[m].to_numpy(float),
        "age": outcome.loc[m, "age"].to_numpy(float),
        "female": (outcome.loc[m, "sex"] == "female").astype(int).to_numpy(),
    })
    try:
        cph = CoxPHFitter().fit(d, "T", "E")
        b_gap = float(cph.params_["gap"])
        b_age = float(cph.params_["age"])
    except Exception as exc:
        log.warning("  %s: calibration fit failed (%s); leaving raw scale", label, exc)
        return gap, 1.0

    if not np.isfinite(b_age) or b_age <= 1e-6:
        log.warning("  %s: chronological age carries no hazard here; cannot "
                    "calibrate, leaving raw scale", label)
        return gap, 1.0
    scale = b_gap / b_age
    if scale <= 0:
        log.warning("  %s: gap runs opposite to age in the hazard model "
                    "(scale %.3f); leaving raw scale and flagging", label, scale)
        return gap, 1.0
    log.info("  %s: 1 raw unit = %.3f chronological-age-equivalent years", label, scale)
    return gap * scale, float(scale)


def deattenuate(gap: pd.Series, age: pd.Series) -> pd.Series:
    """Remove the mechanical dependence of the gap on chronological age.

    Any predictor that is imperfectly correlated with age regresses toward the
    mean: it over-predicts the young and under-predicts the old, so the raw gap
    is systematically positive at young ages and negative at old ones. That
    trend is an artifact of the estimator, not evidence that young people are
    biologically older. Left uncorrected it makes the gap partly a proxy for age
    itself, and since age is the strongest mortality predictor there is, the
    validation would then look impressive while measuring almost nothing.

    The field's standard fix (the "age acceleration residual"): regress the gap
    on chronological age and keep the residual. Chronological age is ALSO kept
    as a covariate in every downstream outcome model, belt and braces.
    """
    m = gap.notna() & age.notna()
    if m.sum() < 50:
        return gap
    x = age[m].to_numpy(float)
    y = gap[m].to_numpy(float)
    b, a = np.polyfit(x, y, 1)
    out = gap.copy()
    out[m] = y - (a + b * x)
    return out


# --------------------------------------------------------------------------
# Stage 2: modality gaps -> combined gap
# --------------------------------------------------------------------------
@dataclass
class Combiner:
    """Linear-on-gaps mortality model. Linear by design: because the score is a
    weighted sum of modality gaps, "which modality is driving this person's
    score" is read directly off weight x gap, with no attribution machinery."""

    method: str
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    modalities: tuple[str, ...] = ()
    scale: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_events: int = 0

    def contributions(self, gaps: dict[str, float]) -> dict[str, float]:
        """Per-modality contribution to the combined gap, in years."""
        return {
            m: self.weights.get(m, 0.0) * gaps[m]
            for m in self.modalities
            if gaps.get(m) is not None and np.isfinite(gaps.get(m, np.nan))
        }

    def predict_gap(self, gaps: dict[str, float]) -> tuple[float, list[str], list[str]]:
        """Combined gap with graceful degradation over missing modalities.

        Missing modalities are DROPPED and the remaining weights renormalised,
        rather than imputed as zero. Imputing zero would silently assert "this
        person is exactly average on the channel we did not measure", pulling
        every partially-measured person toward the mean and making a
        single-modality reading look falsely reassuring.
        """
        used = [m for m in self.modalities
                if gaps.get(m) is not None and np.isfinite(gaps.get(m, np.nan))]
        missing = [m for m in self.modalities if m not in used]
        if not used:
            return np.nan, [], list(self.modalities)
        wsum = sum(abs(self.weights.get(m, 0.0)) for m in used)
        total = sum(abs(self.weights.get(m, 0.0)) for m in self.modalities)
        if wsum <= 0:
            return np.nan, used, missing
        raw = sum(self.weights.get(m, 0.0) * gaps[m] for m in used)
        return float(raw * (total / wsum)), used, missing


def fit_combiner(
    gaps: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    method: str = "cox",
    modalities: tuple[str, ...] | None = None,
    penalizer: float = 0.01,
) -> Combiner:
    """Fit the across-modality combiner against MORTALITY.

    method="cox"      -- Cox proportional hazards on time-to-death. Preferred:
                         it uses the follow-up time, so a death at 6 months and
                         one at 14 years are not treated as the same event, and
                         it handles the varying censoring that a fixed 2019
                         cut-off imposes on participants examined in different
                         years.
    method="logistic" -- elastic-net logistic regression on binary vital status.
                         Simpler and robust when events are few, but discards
                         timing and is sensitive to the follow-up window.

    There is no chronological-age target anywhere in here. If the mortality
    outcome were unavailable this function raises rather than silently
    substituting one.
    """
    mods = modalities or tuple(m for m in C.MODALITIES if f"gap_{m}" in gaps.columns)
    cols = [f"gap_{m}" for m in mods]
    if outcome["died"].notna().sum() == 0:
        raise RuntimeError(
            "combiner requires linked mortality; refusing to fall back to a "
            "chronological-age target"
        )

    m = gaps[cols].notna().all(axis=1) & outcome["died"].notna() & outcome["time"].gt(0)
    X = gaps.loc[m, cols].astype(float)
    n_ev = int(outcome.loc[m, "died"].sum())
    if m.sum() < 50 or n_ev < 10:
        raise RuntimeError(f"too little training data: n={int(m.sum())}, events={n_ev}")

    # Standardise so weights are comparable across modalities whose gaps have
    # very different spreads; converted back to the year scale afterwards.
    scale = {c: float(X[c].std()) or 1.0 for c in cols}
    Xs = X / pd.Series(scale)

    if method == "cox":
        d = Xs.copy()
        d["T"] = outcome.loc[m, "time"].to_numpy(float)
        d["E"] = outcome.loc[m, "died"].to_numpy(int)
        d["age"] = outcome.loc[m, "age"].to_numpy(float)
        d["female"] = (outcome.loc[m, "sex"] == "female").astype(int).to_numpy()
        # LIGHT penalty only. GrimAge's elastic net was regularising 12 protein
        # surrogates plus covariates; this combiner has two or three gaps and
        # 546 events, which is not a high-dimensional problem. Measured on this
        # data, penalizer=0.1 collapsed the blood weight from 0.46 to 0.06 and
        # COST 0.008 of C-index (0.8586 -> 0.8502) -- at that strength the
        # penalty was discarding a real predictor rather than controlling
        # overfitting. 0.01 keeps the fit stable if two modalities are nearly
        # collinear without throwing one away.
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5)
        cph.fit(d, duration_col="T", event_col="E")
        coefs = {c: float(cph.params_[c]) for c in cols}
        intercept = 0.0
    elif method == "logistic":
        y = outcome.loc[m, "died"].to_numpy(int)
        Xa = Xs.copy()
        Xa["age"] = outcome.loc[m, "age"].to_numpy(float)
        Xa["female"] = (outcome.loc[m, "sex"] == "female").astype(int).to_numpy()
        lr = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga",
                                C=1.0, max_iter=5000)
        lr.fit(Xa, y)
        coefs = {c: float(lr.coef_[0][i]) for i, c in enumerate(cols)}
        intercept = float(lr.intercept_[0])
    else:
        raise ValueError(f"unknown method {method}")

    # Re-express on the year scale, then normalise so the combined gap is in
    # interpretable years rather than log-hazard units.
    raw = {m_: coefs[f"gap_{m_}"] / scale[f"gap_{m_}"] for m_ in mods}
    tot = sum(abs(v) for v in raw.values())
    weights = {k: (v / tot if tot > 0 else 0.0) for k, v in raw.items()}

    log.info("combiner (%s): n=%d events=%d weights=%s",
             method, int(m.sum()), n_ev,
             {k: round(v, 3) for k, v in weights.items()})
    return Combiner(method=method, weights=weights, intercept=intercept,
                    modalities=tuple(mods), scale=scale,
                    n_train=int(m.sum()), n_events=n_ev)


# --------------------------------------------------------------------------
# Cross-fitted end-to-end scoring
# --------------------------------------------------------------------------
def combined_feature_weights(
    outcome_w: pd.Series, precision_w: pd.Series | None
) -> pd.Series:
    """Fuse the mortality-association weight with the inverse-variance weight.

    The spec's rule is that features are weighted by how well they predict the
    outcome -- a marker that barely predicts death should not get equal say. But
    outcome association alone is not sufficient: it says nothing about how
    PRECISELY a feature can locate someone on the age axis. A flat, scattered
    curve yields an implied age that is nearly noise, and multiplying noise by a
    modest mortality z-score still leaves noise.

    The product of the two keeps the spec's rule intact (zero outcome weight
    still means zero say) while suppressing features whose inversion is
    unreliable. This is the Klemera-Doubal inverse-variance term applied to the
    combine step -- see curves.curve_precision.
    """
    if precision_w is None:
        return outcome_w
    w = outcome_w.mul(precision_w.reindex(outcome_w.index).fillna(0.0))
    if w.sum() <= 0:
        log.warning("precision weighting zeroed every feature; using outcome weights")
        return outcome_w
    return w / w.sum()


def crossfit_modality(
    implied: pd.DataFrame,
    tab: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    precision_w: pd.Series | None = None,
    n_folds: int = N_FOLDS,
    seed: int = C.RANDOM_SEED,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Out-of-fold modality bioage and gap.

    Returns (bioage, gap, weight_table). Feature weights are refitted inside
    every fold so that no participant's score is informed by their own outcome.
    """
    age = tab["age"]
    gaps_feat = implied.sub(age, axis=0)

    usable = outcome["died"].notna() & outcome["time"].gt(0) & gaps_feat.notna().any(axis=1)
    idx = gaps_feat.index[usable]
    y = outcome.loc[idx, "died"].astype(int)

    bioage = pd.Series(np.nan, index=gaps_feat.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(idx)), y):
        tr_idx, te_idx = idx[tr], idx[te]
        w = combined_feature_weights(
            feature_mortality_weights(gaps_feat.loc[tr_idx], outcome.loc[tr_idx]),
            precision_w,
        )
        bioage.loc[te_idx] = modality_bioage(
            gaps_feat.loc[te_idx], w, age.loc[te_idx]
        )

    # Full-sample weights, reported for interpretation and used to score people
    # who are outside the mortality-eligible training set (e.g. new inputs at
    # inference time, where there is no outcome to hold out).
    ow = feature_mortality_weights(gaps_feat.loc[idx], outcome.loc[idx])
    info = getattr(feature_mortality_weights, "last_info", pd.DataFrame())
    w_full = combined_feature_weights(ow, precision_w)
    info = info.assign(
        precision=(precision_w.reindex(info.index) if precision_w is not None else np.nan),
        final_weight=w_full.reindex(info.index),
    )
    missing = bioage.isna() & gaps_feat.notna().any(axis=1)
    bioage.loc[missing] = modality_bioage(
        gaps_feat.loc[missing], w_full, age.loc[missing]
    )

    gap = bioage - age
    rel = split_half_reliability(gaps_feat.loc[idx], w_full)
    gap = shrink_to_reliability(gap, rel)
    crossfit_modality.last_reliability = rel  # type: ignore[attr-defined]
    log.info("  split-half reliability %.3f -> gap SD %.2f y after shrinkage",
             rel, gap.std())
    return bioage, gap, info
