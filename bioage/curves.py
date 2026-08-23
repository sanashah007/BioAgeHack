"""Phase 2 -- survey-weighted, sex-stratified reference curves and inversion.

The approach is CURVE INVERSION rather than direct regression on age: build
f(age) -> expected marker value from the weighted population, then run it
backwards -- given an individual's value, find the age at which the population
curve matches it. A growth chart read in reverse. This yields a per-feature
implied age, which is both interpretable on its own and the natural input to an
outcome-weighted combination.

Relationship to published work: this is a close cousin of the Klemera-Doubal
method, which also regresses each biomarker on age and inverts, weighting each
biomarker by its regression slope over its residual scatter. The difference here
is that the age relationship is fitted non-parametrically (KDM assumes linear)
and the final weights come from mortality association rather than from the age
regression itself. The variance normalisation in `invert` plays the role of
KDM's slope/scatter weighting at the per-feature level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def _tricube(u: np.ndarray) -> np.ndarray:
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u**3) ** 3


def weighted_loess(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    grid: np.ndarray,
    frac: float = C.LOESS_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Survey-weighted local linear regression.

    Returns (fitted mean, local SD, effective N) evaluated on `grid`.

    Survey weights are NOT optional here. NHANES deliberately oversamples older
    adults, several race/ethnicity groups and low-income households, so an
    unweighted curve describes the sample rather than the US population and is
    biased toward whoever was oversampled. The weights enter as multiplicative
    case weights alongside the tricube kernel.

    statsmodels' lowess does not accept observation weights, so the local linear
    fit is done directly. `frac` is a nearest-neighbour bandwidth: at each grid
    point the bandwidth is the distance covering that fraction of observations,
    which automatically widens the window where data is sparse -- the upper age
    tail, exactly where an unadapted fixed bandwidth goes jagged.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n = x.size
    k = max(int(np.ceil(frac * n)), 20)

    mean = np.full(grid.size, np.nan)
    sd = np.full(grid.size, np.nan)
    neff = np.zeros(grid.size)

    order = np.argsort(x)
    xs, ys, ws = x[order], y[order], w[order]

    for i, g in enumerate(grid):
        d = np.abs(xs - g)
        # Nearest-neighbour bandwidth, with a floor so the kernel never collapses.
        h = np.partition(d, min(k, n) - 1)[min(k, n) - 1]
        h = max(h, 1.0)
        kern = _tricube(d / h)
        ww = kern * ws
        m = ww > 0
        if m.sum() < 10:
            continue
        xw, yw, wv = xs[m], ys[m], ww[m]

        # Weighted local linear fit, centred at g for numerical stability.
        dx = xw - g
        sw = wv.sum()
        sx = (wv * dx).sum()
        sxx = (wv * dx * dx).sum()
        sy = (wv * yw).sum()
        sxy = (wv * dx * yw).sum()
        det = sw * sxx - sx * sx
        if abs(det) < 1e-12:
            fit = sy / sw
            slope = 0.0
        else:
            fit = (sxx * sy - sx * sxy) / det   # intercept at dx = 0, i.e. at g
            slope = (sw * sxy - sx * sy) / det
        mean[i] = fit

        resid = yw - (fit + slope * dx)
        var = (wv * resid**2).sum() / sw
        sd[i] = np.sqrt(max(var, 0.0))
        # Kish effective sample size -- how much independent information the
        # weighted local neighbourhood actually carries.
        neff[i] = (wv.sum() ** 2) / (wv**2).sum()

    return mean, sd, neff


@dataclass
class ReferenceCurve:
    """A fitted age curve for one feature in one sex."""

    feature: str
    sex: str
    grid: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    neff: np.ndarray
    log_transform: bool
    n_obs: int
    n_excluded: int
    exclusions_applied: tuple[str, ...]
    # Monotonicity screening results
    monotonic: bool = False
    direction: str = "none"          # "increase" | "decrease" | "none"
    seg_lo: float = np.nan           # usable (monotonic) age span
    seg_hi: float = np.nan
    usable: bool = False
    reject_reason: str = ""

    @property
    def key(self) -> str:
        return f"{self.feature}|{self.sex}"

    def segment_mask(self) -> np.ndarray:
        return (self.grid >= self.seg_lo) & (self.grid <= self.seg_hi)


def screen_monotonic(curve: ReferenceCurve) -> ReferenceCurve:
    """Flag non-monotonic curves and find the longest usable monotonic segment.

    This step is mandatory before inversion, not a nicety. Nearest-match
    inversion assumes the curve is one-to-one. Against a curve that rises then
    falls -- total cholesterol and triglycerides both peak in mid-life, HDL and
    waist are non-monotonic in at least one sex -- a single marker value matches
    TWO ages, and the inverter silently returns whichever the search happens to
    hit first. The result is not noisy, it is arbitrary, and it looks perfectly
    reasonable in the output.
    """
    m = np.isfinite(curve.mean)
    if m.sum() < 10:
        curve.reject_reason = "too few fitted grid points"
        return curve

    g, mu = curve.grid[m], curve.mean[m]
    if np.allclose(mu, mu[0]):
        curve.reject_reason = "flat curve"
        return curve

    def violation(i: int, j: int) -> tuple[float, str]:
        """Magnitude-weighted non-monotonicity of mu[i:j+1], in [0, 1].

        Counting the SIGN of each step is the obvious metric and it is wrong:
        on a 0.5-year grid a genuinely smooth curve with a flat stretch flips
        sign constantly on numerically negligible wiggle, so plateaus get
        reported as 40%-reversing and real markers get excluded for noise.
        Weighting by magnitude makes the measure scale-free and insensitive to
        flat regions -- it is 1 - |net change| / total variation, i.e. the share
        of the curve's total travel that goes the wrong way.
        """
        d = np.diff(mu[i : j + 1])
        tv = np.abs(d).sum()
        if tv <= 0:
            return 1.0, "none"
        net = mu[j] - mu[i]
        direction = "increase" if net >= 0 else "decrease"
        back = np.abs(d[d < 0]).sum() if net >= 0 else np.abs(d[d > 0]).sum()
        return float(back / tv), direction

    frac, direction = violation(0, mu.size - 1)
    curve.direction = direction
    if frac <= C.MONOTONIC_TOL:
        curve.monotonic = True
        curve.seg_lo, curve.seg_hi = float(g[0]), float(g[-1])
        curve.usable = True
        return curve

    # Non-monotonic overall: find the WIDEST sub-interval that is monotonic to
    # within tolerance. Scanning all (i, j) pairs is ~8k evaluations on this
    # grid -- cheap, and it finds wider valid segments than greedy sign-runs do.
    best = (-1, -1, 0.0, "none")
    n = mu.size
    for i in range(n - 1):
        if g[n - 1] - g[i] <= best[2]:
            break  # cannot beat the incumbent from here on
        for j in range(n - 1, i, -1):
            span = g[j] - g[i]
            if span <= best[2]:
                break
            f, dirn = violation(i, j)
            if f <= C.MONOTONIC_TOL:
                best = (i, j, span, dirn)
                break

    i, j, span, dirn = best
    if i >= 0 and span >= C.MIN_MONOTONIC_SPAN:
        curve.seg_lo, curve.seg_hi = float(g[i]), float(g[j])
        curve.direction = dirn
        curve.usable = True
        curve.reject_reason = (
            f"non-monotonic overall ({frac:.0%} of travel reverses); restricted "
            f"to monotonic segment {curve.seg_lo:.0f}-{curve.seg_hi:.0f}y"
        )
    else:
        curve.reject_reason = (
            f"non-monotonic ({frac:.0%} of travel reverses) and widest monotonic "
            f"span is only {max(span, 0):.0f}y < {C.MIN_MONOTONIC_SPAN}y -- EXCLUDED"
        )
    return curve


def fit_curves(
    tab: pd.DataFrame,
    features: list[C.Feature],
    *,
    healthy_reference: bool = True,
    weight_col: str | None = None,
) -> dict[str, ReferenceCurve]:
    """Fit one curve per (feature, sex).

    healthy_reference:
        True  -- apply each feature's own exclusion rules (default). Matches
                 clinical reference-range practice and published bioage work.
                 A population-average curve bakes prevalent disease into the
                 baseline, so an abnormal marker is scored against a sick
                 population and looks less abnormal than it is.
        False -- population average: everyone, including prevalent disease.
                 Reflects "typical" but understates abnormality.
    """
    grid = np.arange(C.AGE_MIN, C.AGE_MAX + 1e-9, C.CURVE_AGE_GRID_STEP)
    curves: dict[str, ReferenceCurve] = {}

    for feat in features:
        col = f"pax_{feat.var}" if feat.file == "PAX" else feat.var
        if col not in tab.columns:
            log.warning("  %-22s SKIP -- column %s absent", feat.name, col)
            continue

        # Weight selection: use the narrowest sub-sample weight the feature
        # requires. Glucose and triglycerides come from the morning fasting
        # sub-sample and must use WTSAF2YR; using WTMEC2YR would weight them as
        # if the whole exam sample had been measured.
        wcol = weight_col or (
            C.NHANES_FILES[feat.file].weight if feat.file in C.NHANES_FILES else "WTMEC2YR"
        )
        if wcol not in tab.columns:
            log.warning("  %-22s weight %s absent, falling back to WTMEC2YR", feat.name, wcol)
            wcol = "WTMEC2YR"

        base = tab[tab["in_age_window"]].copy()
        vals = base[col].astype(float)
        if feat.valid_range:
            lo, hi = feat.valid_range
            n_oor = int(((vals < lo) | (vals > hi)).sum())
            if n_oor:
                log.info("  %-22s %d values outside plausible range %s -> dropped",
                         feat.name, n_oor, feat.valid_range)
            vals = vals.where(vals.between(lo, hi))
        base = base.assign(_y=vals)

        applied: list[str] = []
        n_before = int(base["_y"].notna().sum())
        if healthy_reference:
            for rule in feat.exclude_if:
                fcol = f"excl_{rule}"
                if fcol in base.columns:
                    base = base[~base[fcol].fillna(False).astype(bool)]
                    applied.append(rule)
                else:
                    log.warning("  %-22s exclusion '%s' unavailable", feat.name, rule)
        n_after = int(base["_y"].notna().sum())

        for sex in ("male", "female"):
            sub = base[(base["sex"] == sex) & base["_y"].notna() & base[wcol].gt(0)]
            if len(sub) < C.MIN_N_PER_SEX:
                log.warning("  %-22s %-6s SKIP -- only %d observations",
                            feat.name, sex, len(sub))
                continue

            y = sub["_y"].to_numpy(float)
            if feat.log_transform:
                # Fit in log space: these markers are right-skewed, so a
                # mean-based local fit on the raw scale is dragged by outliers.
                y = np.log(np.clip(y, 1e-9, None))

            mu, sd, neff = weighted_loess(
                sub["age"].to_numpy(float), y, sub[wcol].to_numpy(float), grid
            )
            c = ReferenceCurve(
                feature=feat.name, sex=sex, grid=grid, mean=mu, sd=sd, neff=neff,
                log_transform=feat.log_transform, n_obs=len(sub),
                n_excluded=n_before - n_after,
                exclusions_applied=tuple(applied),
            )
            c = screen_monotonic(c)
            curves[c.key] = c
            log.info(
                "  %-22s %-6s n=%5d  %-9s %s",
                feat.name, sex, len(sub), c.direction,
                "OK" if c.monotonic else (c.reject_reason or "?"),
            )
    return curves


def invert(
    value: float, curve: ReferenceCurve
) -> tuple[float, float, bool]:
    """Invert one curve: value -> implied age.

    Returns (implied_age, normalised_distance, extrapolated).

    Selection is argmin over the usable segment of

        |curve.mean(a) - value| / curve.sd(a)

    The variance normalisation is what makes the match meaningful. Raw value
    distance is not comparable across ages because marker spread is strongly
    heteroscedastic -- CRP and triglycerides fan out sharply with age, so a raw
    nearest-match is drawn toward whichever end of the curve happens to be
    steepest rather than toward the age the value actually resembles.

    `extrapolated` marks values falling outside the curve's range, where the
    implied age is pinned to a segment endpoint and is a bound, not an estimate.
    """
    if not curve.usable or not np.isfinite(value):
        return np.nan, np.nan, False

    v = np.log(max(value, 1e-9)) if curve.log_transform else value
    m = curve.segment_mask() & np.isfinite(curve.mean) & np.isfinite(curve.sd)
    if m.sum() < 3:
        return np.nan, np.nan, False

    g, mu, sd = curve.grid[m], curve.mean[m], curve.sd[m]
    sd = np.where(sd > 1e-9, sd, np.nanmedian(sd[sd > 1e-9]) if np.any(sd > 1e-9) else 1.0)

    dist = np.abs(mu - v) / sd
    i = int(np.argmin(dist))

    lo, hi = (mu.min(), mu.max())
    extrapolated = bool(v < lo or v > hi)

    # Sub-grid refinement by linear interpolation between the two bracketing
    # points, so the implied age is not quantised to the 0.5-year grid.
    age = float(g[i])
    if not extrapolated and 0 < i < g.size - 1:
        for a, b in ((i - 1, i), (i, i + 1)):
            if (mu[a] - v) * (mu[b] - v) <= 0 and mu[b] != mu[a]:
                t = (v - mu[a]) / (mu[b] - mu[a])
                age = float(g[a] + t * (g[b] - g[a]))
                break

    return age, float(dist[i]), extrapolated


def curve_precision(curve: ReferenceCurve) -> float:
    """How precisely this curve can pin down an age, in 1 / years^2.

    Inverting a curve converts a value into an age, and the uncertainty of that
    conversion is roughly (local scatter) / (local slope): a marker that barely
    changes with age but varies a lot between people of the SAME age cannot
    locate anyone on the age axis, however faithfully it was fitted. Its implied
    age is near-noise, swinging to whichever segment endpoint the participant
    happens to fall past.

    Weighting by 1 / SE^2 is the standard inverse-variance combination and is
    what the Klemera-Doubal method uses (its per-biomarker weight is the squared
    slope-to-scatter ratio). Without it, a handful of flat, noisy markers inflate
    the modality gap far beyond the 5-8 year spread published clocks report,
    because every extreme per-feature implied age enters the average at full
    strength.
    """
    if not curve.usable:
        return 0.0
    m = curve.segment_mask() & np.isfinite(curve.mean) & np.isfinite(curve.sd)
    if m.sum() < 5:
        return 0.0
    g, mu, sd = curve.grid[m], curve.mean[m], curve.sd[m]
    slope = np.gradient(mu, g)
    good = (sd > 1e-9) & np.isfinite(slope)
    if good.sum() < 5:
        return 0.0
    se_age = np.abs(sd[good] / np.where(np.abs(slope[good]) > 1e-12, slope[good], np.nan))
    se_age = se_age[np.isfinite(se_age)]
    if se_age.size == 0:
        return 0.0
    return float(1.0 / max(np.median(se_age), 1e-6) ** 2)


def precision_weights(
    features: list[C.Feature], curves: dict[str, ReferenceCurve]
) -> pd.Series:
    """Per-feature inverse-variance weight, averaged over the sexes."""
    out = {}
    for feat in features:
        vals = [curve_precision(curves[f"{feat.name}|{s}"])
                for s in ("male", "female") if f"{feat.name}|{s}" in curves]
        vals = [v for v in vals if v > 0]
        out[feat.name] = float(np.mean(vals)) if vals else 0.0
    s = pd.Series(out)
    return s / s.sum() if s.sum() > 0 else s


def implied_ages(
    tab: pd.DataFrame, features: list[C.Feature], curves: dict[str, ReferenceCurve]
) -> pd.DataFrame:
    """Per-feature implied ages for every participant. One column per feature."""
    out = pd.DataFrame(index=tab.index)
    extrap = pd.DataFrame(index=tab.index)

    for feat in features:
        col = f"pax_{feat.var}" if feat.file == "PAX" else feat.var
        if col not in tab.columns:
            continue
        ages = np.full(len(tab), np.nan)
        ex = np.zeros(len(tab), dtype=bool)
        vals = tab[col].to_numpy(float)
        sexes = tab["sex"].to_numpy()

        for sex in ("male", "female"):
            c = curves.get(f"{feat.name}|{sex}")
            if c is None or not c.usable:
                continue
            idx = np.where(sexes == sex)[0]
            for i in idx:
                if np.isfinite(vals[i]):
                    a, _, e = invert(vals[i], c)
                    ages[i], ex[i] = a, e
        out[feat.name] = ages
        extrap[feat.name] = ex

    n_cov = out.notna().sum(axis=1)
    log.info("implied ages: %d features inverted; median coverage %.0f features/person",
             out.shape[1], n_cov.median())
    return out


def curves_to_frame(curves: dict[str, ReferenceCurve]) -> pd.DataFrame:
    """Tabular summary of the monotonicity screen, for the audit log."""
    rows = []
    for c in curves.values():
        rows.append(dict(
            feature=c.feature, sex=c.sex, n_obs=c.n_obs, n_excluded=c.n_excluded,
            exclusions="+".join(c.exclusions_applied) or "-",
            direction=c.direction, monotonic=c.monotonic, usable=c.usable,
            seg_lo=c.seg_lo, seg_hi=c.seg_hi, note=c.reject_reason or "monotonic across full range",
        ))
    return pd.DataFrame(rows).sort_values(["feature", "sex"]).reset_index(drop=True)
