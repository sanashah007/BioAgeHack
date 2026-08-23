"""Phase 5 -- validation. This is the headline result, not an afterthought.

Four questions, each answered with a table and a figure:
  1. Does each modality's gap predict mortality ON ITS OWN? (does it earn a seat)
  2. Does the COMBINED score beat the best single modality? (the value proposition)
  3. Are the modality gaps redundant? (do they measure the same thing)
  4. Where do the modalities disagree? (the cases that make the case)

Every outcome model adjusts for chronological age and sex. Without the age
adjustment these numbers would mostly be measuring age, which we already knew.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from . import config as C

log = logging.getLogger(__name__)

# Palette: slots 1-3 of the validated categorical order for the three
# modalities, slot 7 (violet) for the combined score. Verified with the
# skill's validator on the light surface under --pairs all (scatter is used
# below, so adjacent-pair validation is not sufficient): worst CVD dE 9.2,
# worst normal-vision dE 16.3. Aqua sits below 3:1 on the light surface, so
# every figure carries a legend and direct labels rather than relying on hue.
PALETTE = {
    "blood": "#2a78d6",
    "wearable": "#eb6834",
    "methylation": "#1baf7a",
    "combined": "#4a3aa7",
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "legend.frameon": False, "figure.dpi": 130,
})


# --------------------------------------------------------------------------
# 1 + 2. Mortality association, per modality and combined
# --------------------------------------------------------------------------
def evaluate_gap(
    gap: pd.Series, outcome: pd.DataFrame, label: str, *, horizon_years: float = 10.0
) -> dict:
    """Cox HR per SD, Harrell C-index, and time-horizon AUC for one gap."""
    m = gap.notna() & outcome["died"].notna() & outcome["time"].gt(0)
    n, n_ev = int(m.sum()), int(outcome.loc[m, "died"].sum())
    if n_ev < 10:
        return dict(score=label, n=n, events=n_ev, note="too few events")

    g = gap[m]
    sd = float(g.std())
    z = (g - g.mean()) / sd
    d = pd.DataFrame({
        "T": outcome.loc[m, "time"].to_numpy(float),
        "E": outcome.loc[m, "died"].astype(int).to_numpy(),
        "gap_sd": z.to_numpy(float),
        "age": outcome.loc[m, "age"].to_numpy(float),
        "female": (outcome.loc[m, "sex"] == "female").astype(int).to_numpy(),
    })
    cph = CoxPHFitter()
    cph.fit(d, "T", "E")
    s = cph.summary.loc["gap_sd"]

    # The published convention for age-acceleration measures is HR per YEAR of
    # gap, adjusted for chronological age (Chen 2016, Liu 2018, Lu 2019). HR per
    # SD is also reported here because it is comparable across modalities whose
    # gaps have different spreads, but per-year is what benchmarks against the
    # literature. Both come from the same fit -- one SD is `sd` years, so the
    # per-year coefficient is just the per-SD coefficient divided by sd.
    hr_per_year = float(np.exp(np.log(float(s["exp(coef)"])) / sd)) if sd > 0 else np.nan

    # Age+sex-only baseline, so the marginal value of the gap is explicit.
    base = CoxPHFitter().fit(d[["T", "E", "age", "female"]], "T", "E")
    c_base = float(base.concordance_index_)

    # AUC for death within the horizon, among those observed that long.
    obs = (d["T"] >= horizon_years) | (d["E"] == 1)
    y = ((d["T"] <= horizon_years) & (d["E"] == 1)).astype(int)
    auc = auc_adj = np.nan
    if obs.sum() > 50 and 0 < y[obs].sum() < obs.sum():
        auc = float(roc_auc_score(y[obs], d.loc[obs, "gap_sd"]))
        risk = cph.predict_partial_hazard(d.loc[obs])
        auc_adj = float(roc_auc_score(y[obs], risk))

    return dict(
        score=label, n=n, events=n_ev, gap_sd_years=sd,
        hr_per_year=hr_per_year,
        hr_per_sd=float(s["exp(coef)"]),
        hr_lo=float(s["exp(coef) lower 95%"]), hr_hi=float(s["exp(coef) upper 95%"]),
        p=float(s["p"]),
        c_index=float(cph.concordance_index_), c_index_age_sex_only=c_base,
        c_gain=float(cph.concordance_index_) - c_base,
        auc_gap_alone=auc, auc_with_age_sex=auc_adj,
    )


def validation_table(
    gaps: dict[str, pd.Series], outcome: pd.DataFrame, *, horizon_years: float = 10.0
) -> pd.DataFrame:
    """Per-modality and combined mortality validation, on a COMMON sample.

    The comparison is restricted to participants who have every score, because
    a combined score evaluated on a different (typically healthier, better
    measured) subset than the single-modality scores would beat them for
    reasons that have nothing to do with fusion.
    """
    common = pd.concat(gaps.values(), axis=1).notna().all(axis=1)
    common &= outcome["died"].notna() & outcome["time"].gt(0)
    log.info("validation on common sample: n=%d, deaths=%d",
             int(common.sum()), int(outcome.loc[common, "died"].sum()))

    rows = [evaluate_gap(g[common], outcome[common], name, horizon_years=horizon_years)
            for name, g in gaps.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Redundancy
# --------------------------------------------------------------------------
def redundancy(gaps: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise correlation between modality gaps.

    High correlation would mean the modalities are noisy proxies for one common
    signal and fusion buys little. Low-to-moderate correlation means they carry
    complementary information -- which is the interesting result, and the one
    that justifies a multi-modal product at all.
    """
    df = pd.concat(gaps, axis=1)
    return df.corr(min_periods=50)


# --------------------------------------------------------------------------
# 4. Disagreement cases
# --------------------------------------------------------------------------
def disagreement_cases(
    gaps: dict[str, pd.Series], tab: pd.DataFrame, outcome: pd.DataFrame, *, n: int = 12
) -> pd.DataFrame:
    """Individuals whose modalities tell materially different stories.

    These are the demo examples: someone with strong wearable-derived fitness
    alongside poor metabolic bloodwork is invisible to a single-source product
    and is exactly what the per-modality breakdown is for.
    """
    keys = [k for k in gaps if k != "combined"]
    if len(keys) < 2:
        return pd.DataFrame()
    df = pd.concat({k: gaps[k] for k in keys}, axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    z = (df - df.mean()) / df.std()
    spread = z.max(axis=1) - z.min(axis=1)
    out = df.copy()
    out["spread_sd"] = spread
    out["age"] = tab["age"].reindex(out.index)
    out["sex"] = tab["sex"].reindex(out.index)
    out["died"] = outcome["died"].reindex(out.index)
    out["followup_y"] = outcome["time"].reindex(out.index).round(1)
    out["faster_aging"] = z.idxmax(axis=1)
    out["slower_aging"] = z.idxmin(axis=1)
    return out.sort_values("spread_sd", ascending=False).head(n).round(2)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_reference_curves(curves: dict, features: list, path, *, title: str) -> None:
    """Every fitted curve, both sexes, with the usable (monotonic) span marked.

    The spec requires plotting every curve -- this is the artifact that makes
    the monotonicity screen auditable rather than a number in a log.
    """
    feats = [f for f in features if f"{f.name}|male" in curves or f"{f.name}|female" in curves]
    if not feats:
        return
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, feats):
        for sex, color, ls in (("male", "#2a78d6", "-"), ("female", "#eb6834", "--")):
            c = curves.get(f"{feat.name}|{sex}")
            if c is None:
                continue
            m = np.isfinite(c.mean)
            mu = np.exp(c.mean) if c.log_transform else c.mean
            ax.plot(c.grid[m], mu[m], ls, color=color, lw=2, label=sex)
            if c.usable:
                seg = c.segment_mask() & m
                ax.plot(c.grid[seg], mu[seg], ls, color=color, lw=4, alpha=0.35)
            else:
                ax.set_facecolor("#f7ecec")
        ax.set_title(f"{feat.label}\n({feat.units})", fontsize=8, color=INK)
        ax.set_xlabel("age (y)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes[len(feats):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title}\nthick band = usable monotonic span; pink panel = excluded",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_distributions(gaps: dict[str, pd.Series], path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, g in gaps.items():
        g = g.dropna()
        if g.empty:
            continue
        ax.hist(g, bins=60, histtype="step", lw=2,
                color=PALETTE.get(name, INK2), label=f"{name} (SD {g.std():.1f}y)",
                density=True)
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_xlabel("age gap (years)  —  positive = biologically older")
    ax.set_ylabel("density")
    ax.set_title("Age-gap distribution by modality", color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_redundancy(corr: pd.DataFrame, path) -> None:
    """Diverging map: correlation is polar (sign matters), midpoint neutral gray."""
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e8e8e4", "#eb6834"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=INK)
    ax.grid(False)
    ax.set_title("Modality gap redundancy", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_validation(vt: pd.DataFrame, path) -> None:
    """Combined vs each single modality: HR per SD and C-index gain."""
    vt = vt.dropna(subset=["hr_per_sd"])
    if vt.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    y = np.arange(len(vt))
    colors = [PALETTE.get(s, INK2) for s in vt["score"]]

    a1.errorbar(vt["hr_per_sd"], y,
                xerr=[vt["hr_per_sd"] - vt["hr_lo"], vt["hr_hi"] - vt["hr_per_sd"]],
                fmt="o", ms=9, lw=2, capsize=4, ecolor=INK2, ls="none",
                mfc="none", mec="none")
    for yi, (_, r) in zip(y, vt.iterrows()):
        a1.plot(r["hr_per_sd"], yi, "o", ms=9, color=PALETTE.get(r["score"], INK2))
        a1.annotate(f"  {r['hr_per_sd']:.2f}", (r["hr_hi"], yi), fontsize=8,
                    va="center", color=INK)
    a1.axvline(1.0, color=INK2, ls=":", lw=1)
    a1.set_yticks(y, vt["score"], fontsize=9)
    a1.set_xlabel("hazard ratio per SD of gap\n(adjusted for age + sex)")
    a1.set_title("Does the gap predict death?", color=INK)

    a2.barh(y, vt["c_gain"], color=colors, height=0.6)
    for yi, v in zip(y, vt["c_gain"]):
        a2.annotate(f" {v:+.3f}", (v, yi), fontsize=8, va="center", color=INK)
    a2.axvline(0, color=INK2, lw=1)
    a2.set_yticks(y, vt["score"], fontsize=9)
    a2.set_xlabel("C-index gain over age + sex alone")
    a2.set_title("How much does it add?", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_km_by_tertile(gap: pd.Series, outcome: pd.DataFrame, path, *,
                       label: str = "combined") -> None:
    m = gap.notna() & outcome["died"].notna() & outcome["time"].gt(0)
    if m.sum() < 100:
        return
    g = gap[m]
    tert = pd.qcut(g, 3, labels=["slower aging (bottom 3rd)", "middle",
                                 "faster aging (top 3rd)"])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    kmf = KaplanMeierFitter()
    for name, color in zip(tert.cat.categories, ["#2a78d6", "#8a8983", "#eb6834"]):
        sel = tert == name
        kmf.fit(outcome.loc[m, "time"][sel], outcome.loc[m, "died"][sel], label=str(name))
        kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_show=True, ci_alpha=0.12)
    ax.set_xlabel("years since exam")
    ax.set_ylabel("survival probability")
    ax.set_title(f"Survival by {label} age-gap tertile", color=INK)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement(gaps: dict[str, pd.Series], path) -> None:
    keys = [k for k in gaps if k != "combined"][:2]
    if len(keys) < 2:
        return
    df = pd.concat({k: gaps[k] for k in keys}, axis=1).dropna()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(df[keys[0]], df[keys[1]], s=9, alpha=0.35,
               color="#2a78d6", edgecolors="none")
    lim = np.nanpercentile(np.abs(df.values), 99)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, ls=":", lw=1)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    r = df[keys[0]].corr(df[keys[1]])
    ax.set_xlabel(f"{keys[0]} gap (years)")
    ax.set_ylabel(f"{keys[1]} gap (years)")
    ax.set_title(f"Where modalities disagree  (r = {r:.2f}, n = {len(df)})", color=INK)
    ax.annotate("fit body,\nailing bloodwork", (-lim * 0.75, lim * 0.75), fontsize=8,
                color=INK2, ha="center")
    ax.annotate("clean bloodwork,\nsedentary", (lim * 0.75, -lim * 0.75), fontsize=8,
                color=INK2, ha="center")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
