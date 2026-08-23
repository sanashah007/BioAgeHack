"""Phase 1b -- engineer wearable features from NHANES minute-level accelerometry.

SCOPE / HONEST LIMITATION
------------------------
The 2005-2006 NHANES physical activity monitor is a uniaxial, waist-worn
ActiGraph AM-7164 recording a single "activity count" per minute during waking
hours for 7 days. It provides NO heart-rate variability, NO resting heart rate,
NO sleep staging and NO SpO2 -- the channels that carry most of the aging signal
in a modern consumer wearable (Apple Watch, Oura, Whoop). What is recoverable is
activity VOLUME, INTENSITY DISTRIBUTION and CIRCADIAN RHYTHM/FRAGMENTATION.
This arm should therefore be read as a lower bound on what a real wearable arm
would contribute, not as an estimate of it.

A second, subtler limitation: because the device is removed for sleep, the
"rest" period is largely non-wear. L5 and interdaily stability are computed over
wear time only and so describe daytime rest, not sleep.

DATA GOTCHA
-----------
pandas.read_sas decodes SAS transport zeros as the denormal 5.397605e-79 rather
than 0.0. Every non-wear and sedentary rule keys off "counts == 0", so failing to
normalise this silently disables non-wear detection entirely (every minute looks
like wear, wear time becomes 10080 min/day, and the features become garbage that
still looks plausible). _denorm() below is load-bearing, not defensive.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

MIN_PER_DAY = 1440

# --- QC constants ---------------------------------------------------------
# PAXCAL == 1 : device passed post-collection calibration
# PAXSTAT == 1: NCHS judged the record reliable
QC_OK = 1

# Troiano et al. 2008 (Med Sci Sports Exerc) non-wear rule, the field standard
# for NHANES 2003-2006 accelerometry.
NONWEAR_MIN_RUN = 60        # >=60 consecutive zero-count minutes = not worn
NONWEAR_SPIKE_TOL = 2       # up to 2 interrupting minutes are allowed...
NONWEAR_SPIKE_MAX = 100     # ...provided their counts stay below this

# Troiano 2008 adult intensity cut-points, counts per minute.
CUT_SEDENTARY = 100         # < 100 cpm
CUT_MODERATE = 2020         # >= 2020 cpm is moderate
CUT_VIGOROUS = 5999         # >= 5999 cpm is vigorous

VALID_DAY_MIN_WEAR = 600    # >=10 h of wear makes a day analysable
MIN_VALID_DAYS = 4          # >=4 valid days makes a participant analysable

_DENORM_EPS = 1e-70


def _denorm(a: np.ndarray) -> np.ndarray:
    """Collapse the SAS-transport denormal-zero artifact to true 0.0."""
    a = np.asarray(a, dtype="float64")
    a[np.abs(a) < _DENORM_EPS] = 0.0
    return a


def _nonwear_mask(counts: np.ndarray) -> np.ndarray:
    """Troiano non-wear detection. True = minute is NON-WEAR.

    A non-wear bout is >=60 consecutive zero-count minutes, tolerating up to 2
    interrupting minutes whose counts are below 100. Implemented by first
    marking "effectively zero" minutes, then scanning maximal runs and rejecting
    any run whose interruption budget is exceeded.
    """
    n = counts.size
    nonwear = np.zeros(n, dtype=bool)
    if n == 0:
        return nonwear

    is_zero = counts == 0
    is_spike_ok = (counts > 0) & (counts < NONWEAR_SPIKE_MAX)

    i = 0
    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        # Extend a candidate bout from i, spending the interruption budget.
        j = i
        spikes = 0
        last_zero = i
        while j < n:
            if is_zero[j]:
                last_zero = j
                j += 1
            elif is_spike_ok[j] and spikes < NONWEAR_SPIKE_TOL:
                spikes += 1
                j += 1
            else:
                break
        # Trim trailing tolerated spikes: a bout must end on a zero minute.
        end = last_zero + 1
        if end - i >= NONWEAR_MIN_RUN:
            nonwear[i:end] = True
        i = max(end, i + 1)
    return nonwear


def _iv_is(hourly: np.ndarray, n_days: int) -> tuple[float, float]:
    """Intradaily variability and interdaily stability from hourly means.

    IV  -- mean squared hour-to-hour difference over the total variance. High IV
           means a fragmented rhythm (many rest/activity switches), which rises
           with age and is a documented frailty correlate.
    IS  -- between-hour variance over total variance: how reproducible the 24-h
           profile is from day to day. High IS means a regular rhythm.

    The textbook formulae assume a complete, gap-free 24-h x n-day matrix. Wear
    data is not gap-free (the device comes off), and applying them naively lets
    IS exceed its [0, 1] bound because the observation count and the profile
    length are drawn from different denominators. Both are therefore computed
    here in variance-ratio form, which is what the textbook versions reduce to
    under complete data and which stays correctly bounded under missingness.
    """
    n_days_avail = hourly.size // 24
    if n_days_avail < 2 or n_days < 2:
        return np.nan, np.nan
    mat = hourly[: n_days_avail * 24].reshape(n_days_avail, 24)

    valid = ~np.isnan(hourly)
    x = hourly[valid]
    if x.size < 24:
        return np.nan, np.nan
    xbar = x.mean()
    total_var = np.mean((x - xbar) ** 2)
    if total_var <= 0:
        return np.nan, np.nan

    # IV: use only consecutive hour pairs where BOTH hours were observed, so a
    # gap does not register as a huge artificial jump.
    a, b = hourly[:-1], hourly[1:]
    both = ~np.isnan(a) & ~np.isnan(b)
    iv = float(np.mean((b[both] - a[both]) ** 2) / total_var) if both.sum() >= 12 else np.nan

    # IS: variance of the hour-of-day profile, weighted by how many days
    # actually contributed to each hour.
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(mat, axis=0)
    n_h = (~np.isnan(mat)).sum(axis=0)
    ok = n_h > 0
    if ok.sum() < 12:
        return iv, np.nan
    between_var = np.sum(n_h[ok] * (prof[ok] - xbar) ** 2) / np.sum(n_h[ok])
    is_ = float(np.clip(between_var / total_var, 0.0, 1.0))
    return iv, is_


def _rolling_window_mean(x: np.ndarray, hours: int, mode: str,
                         min_wear_frac: float = 0.8) -> float:
    """Max (M10) or min (L5) mean over a contiguous window of `hours` hours.

    Windows are computed over WEAR minutes only. Treating non-wear as zero (the
    obvious implementation) is wrong here and quietly destroys the feature: the
    device is removed for sleep, so on nearly every day some 5-hour window is
    mostly device-off, driving L5 to exactly 0 and pinning relative amplitude at
    1.0 for the whole cohort. Windows with less than `min_wear_frac` wear are
    rejected instead, so L5 describes genuine daytime rest.
    """
    w = hours * 60
    if x.size < w:
        return np.nan
    obs = ~np.isnan(x)
    filled = np.nan_to_num(x, nan=0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(obs.astype("float64"))])
    tot = csum[w:] - csum[:-w]
    cnt = ccnt[w:] - ccnt[:-w]
    ok = cnt >= min_wear_frac * w
    if not ok.any():
        return np.nan
    means = tot[ok] / cnt[ok]
    return float(means.max() if mode == "max" else means.min())


def _participant_features(g: pd.DataFrame) -> dict | None:
    """Compute wearable features for one participant from their minute rows."""
    counts = _denorm(g["PAXINTEN"].to_numpy())
    day = g["PAXDAY"].to_numpy().astype(int)
    hour = _denorm(g["PAXHOUR"].to_numpy()).astype(int)

    day_feats: list[dict] = []
    hourly_series: list[float] = []

    for d in range(1, 8):
        m = day == d
        if m.sum() < MIN_PER_DAY * 0.9:
            continue
        c = counts[m]
        h = hour[m]
        nonwear = _nonwear_mask(c)
        wear = ~nonwear
        wear_min = int(wear.sum())
        if wear_min < VALID_DAY_MIN_WEAR:
            continue

        cw = c[wear]
        sed = int((cw < CUT_SEDENTARY).sum())
        mod = int(((cw >= CUT_MODERATE) & (cw < CUT_VIGOROUS)).sum())
        vig = int((cw >= CUT_VIGOROUS).sum())

        # Fragmentation: rate of transitions out of active bouts. Equivalent to
        # 1 / mean active-bout length; higher = more broken-up activity.
        active = (cw >= CUT_SEDENTARY).astype(int)
        transitions = int(np.sum(np.diff(active) == -1))
        active_min = int(active.sum())
        frag = transitions / active_min if active_min > 0 else np.nan

        # Hour-of-day means over wear minutes only (non-wear -> NaN, so the
        # device being off does not masquerade as genuine rest).
        cc = c.astype("float64").copy()
        cc[nonwear] = np.nan
        hh = pd.Series(cc).groupby(h).mean()
        prof = np.full(24, np.nan)
        prof[hh.index.to_numpy()] = hh.to_numpy()
        hourly_series.extend(prof.tolist())

        day_feats.append(
            dict(
                wear_min=wear_min,
                mean_cpm=float(cw.mean()),
                sedentary_frac=sed / wear_min,
                mvpa_min_per_day=mod + vig,
                activity_cv=float(cw.std() / cw.mean()) if cw.mean() > 0 else np.nan,
                m10=_rolling_window_mean(cc, 10, "max"),
                l5=_rolling_window_mean(cc, 5, "min"),
                activity_bout_frag=frag,
            )
        )

    if len(day_feats) < MIN_VALID_DAYS:
        return None

    df = pd.DataFrame(day_feats)
    out = {k: float(df[k].mean()) for k in df.columns}
    out["n_valid_days"] = len(df)

    m10, l5 = out["m10"], out["l5"]
    out["relative_amplitude"] = (m10 - l5) / (m10 + l5) if (m10 + l5) > 0 else np.nan

    iv, is_ = _iv_is(np.asarray(hourly_series, dtype="float64"), len(df))
    out["intradaily_variability"] = iv
    out["interdaily_stability"] = is_
    return out


def build_wearable_features(path=None, *, chunksize: int = 4_000_000) -> pd.DataFrame:
    """Stream PAXRAW_D and produce one feature row per analysable participant.

    The file is ~75M rows / 3 GB, sorted by SEQN. Participants are buffered
    across chunk boundaries so nobody is split and silently half-analysed.
    """
    path = path or (C.RAW / "nhanes" / "paxraw" / "paxraw_d.xpt")
    rows: list[dict] = []
    carry: pd.DataFrame | None = None
    n_seen = n_qc_fail = 0

    def flush(block: pd.DataFrame) -> None:
        nonlocal n_seen, n_qc_fail
        for seqn, g in block.groupby("SEQN", sort=False):
            n_seen += 1
            # Whole-record QC: NCHS flags the entire wear period, so a single
            # failing minute condemns the participant, not just that minute.
            if not ((g["PAXCAL"] == QC_OK).all() and (g["PAXSTAT"] == QC_OK).all()):
                n_qc_fail += 1
                continue
            f = _participant_features(g)
            if f is not None:
                f[C.JOIN_KEY] = int(seqn)
                rows.append(f)

    usecols = ["SEQN", "PAXSTAT", "PAXCAL", "PAXDAY", "PAXHOUR", "PAXINTEN"]
    for chunk in pd.read_sas(path, format="xport", chunksize=chunksize):
        chunk = chunk[usecols]
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["SEQN"].iloc[-1]
        complete = chunk[chunk["SEQN"] != last]
        carry = chunk[chunk["SEQN"] == last]
        if len(complete):
            flush(complete)
            log.info("  processed %d participants (%d usable)", n_seen, len(rows))
    if carry is not None and len(carry):
        flush(carry)

    out = pd.DataFrame(rows)
    log.info(
        "wearable: %d participants seen, %d failed device QC, %d had >=%d valid days -> %d usable",
        n_seen, n_qc_fail, len(out), MIN_VALID_DAYS, len(out),
    )
    if len(out):
        out = out.set_index(C.JOIN_KEY).sort_index()
    return out
