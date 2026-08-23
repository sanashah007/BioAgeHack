#!/usr/bin/env python
"""End-to-end multi-modal biological age pipeline.

    python scripts/run_pipeline.py [--skip-download] [--skip-methylation]
                                   [--population-reference] [--combiner cox|logistic]

Runs Phases 1-5 and writes every table and figure to outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioage import (acquire, config as C, curves, methylation, nhanes, scorer,  # noqa: E402
                    scoring, validate, wearable)

warnings.filterwarnings("ignore")
log = logging.getLogger("pipeline")


def setup_logging() -> None:
    C.LOGS.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(C.LOGS / "pipeline.log", mode="w")],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def banner(msg: str) -> None:
    log.info("=" * 78)
    log.info(msg)
    log.info("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-methylation", action="store_true",
                    help="skip the GEO arm (large download)")
    ap.add_argument("--population-reference", action="store_true",
                    help="use a population-average reference instead of a healthy one")
    ap.add_argument("--combiner", default="cox", choices=["cox", "logistic"])
    ap.add_argument("--rebuild-wearable", action="store_true")
    args = ap.parse_args()
    setup_logging()

    healthy = not args.population_reference
    manifest: dict = {
        "cycle": f"NHANES {C.CYCLE_YEARS}",
        "reference_population": "healthy" if healthy else "population-average",
        "combiner_method": args.combiner,
    }

    # ---------------------------------------------------------------- Phase 1
    banner("PHASE 1 -- acquisition")
    if not args.skip_download:
        acquire.fetch_nhanes()
        acquire.fetch_mortality()
        acquire.fetch_paxraw()

    wpath = C.INTERIM / "wearable_features.parquet"
    if args.rebuild_wearable or not wpath.exists():
        log.info("engineering wearable features from minute-level accelerometry")
        wearable.build_wearable_features().to_parquet(wpath)

    tab = nhanes.build_analytic_table()
    tab.to_parquet(C.INTERIM / "analytic_table.parquet")

    sel = tab["in_age_window"] & tab["mort_eligible"]
    tab_s = tab[sel].copy()
    outcome = pd.DataFrame({
        "died": tab_s["died"],
        "time": tab_s["PERMTH_EXM"] / 12.0,
        "age": tab_s["age"],
        "sex": tab_s["sex"],
    })
    manifest["n_analytic"] = int(len(tab_s))
    manifest["n_deaths"] = int(outcome["died"].sum())
    manifest["max_followup_years"] = float(outcome["time"].max())

    # ---------------------------------------------------------------- Phase 2
    banner("PHASE 2 -- reference curves (weighted, sex-stratified, screened)")
    fitted = {}
    for mod, feats in C.FEATURES_BY_MODALITY.items():
        log.info("--- %s ---", mod)
        fitted[mod] = curves.fit_curves(tab, feats, healthy_reference=healthy)
    pickle.dump(fitted, open(C.INTERIM / "curves.pkl", "wb"))

    screen = curves.curves_to_frame({**fitted["blood"], **fitted["wearable"]})
    screen.to_csv(C.TABLES / "curve_screening.csv", index=False)
    manifest["curves_fitted"] = int(len(screen))
    manifest["curves_fully_monotonic"] = int(screen["monotonic"].sum())
    manifest["curves_segment_restricted"] = int((screen["usable"] & ~screen["monotonic"]).sum())
    manifest["curves_excluded"] = int((~screen["usable"]).sum())
    log.info("curves: %d fitted | %d monotonic | %d segment-restricted | %d EXCLUDED",
             manifest["curves_fitted"], manifest["curves_fully_monotonic"],
             manifest["curves_segment_restricted"], manifest["curves_excluded"])
    for _, r in screen[~screen["usable"]].iterrows():
        log.info("  EXCLUDED %s|%s -- %s", r["feature"], r["sex"], r["note"])

    validate.plot_reference_curves(fitted["blood"], C.BLOOD_FEATURES,
                                   C.FIGURES / "curves_blood.png",
                                   title="Blood reference curves (survey-weighted, by sex)")
    validate.plot_reference_curves(fitted["wearable"], C.WEARABLE_FEATURES,
                                   C.FIGURES / "curves_wearable.png",
                                   title="Wearable reference curves (survey-weighted, by sex)")

    # ------------------------------------------------------------- Phase 2.7
    banner("PHASE 2.7 -- inversion and outcome-weighted modality bioages")
    gaps: dict[str, pd.Series] = {}
    bioages: dict[str, pd.Series] = {}
    mod_params: dict[str, scorer.ModalityParams] = {}
    for mod, feats in C.FEATURES_BY_MODALITY.items():
        implied = curves.implied_ages(tab_s, feats, fitted[mod])
        implied.to_parquet(C.INTERIM / f"implied_ages_{mod}.parquet")
        prec = curves.precision_weights(feats, fitted[mod])
        ba, gap, winfo = scoring.crossfit_modality(
            implied, tab_s, outcome, precision_w=prec
        )
        winfo.to_csv(C.TABLES / f"feature_weights_{mod}.csv")
        mp = dict(getattr(scoring.crossfit_modality, "last_params", {}))
        gap_c, (d_a, d_b) = scoring.deattenuate(gap, tab_s["age"])
        gap_c, scale = scoring.calibrate_to_age_scale(gap_c, outcome, label=mod)
        # Freeze the constants a new individual needs to land on this scale.
        mod_params[mod] = scorer.ModalityParams(
            reliability=mp.get("reliability", 1.0),
            gap_mean_raw=mp.get("gap_mean_raw", 0.0),
            deatt_intercept=d_a, deatt_slope=d_b, scale=scale,
            weights=mp.get("weights", {}),
        )
        manifest[f"{mod}_age_equivalent_scale"] = round(scale, 4)
        manifest[f"{mod}_reliability"] = round(
            getattr(scoring.crossfit_modality, "last_reliability", np.nan), 4)
        bioages[mod] = tab_s["age"] + gap_c
        gaps[mod] = gap_c
        log.info("%s: scored n=%d | gap SD %.2fy (age-equivalent) | "
                 "corr(gap, age) %+.3f -> %+.3f after correction",
                 mod, int(ba.notna().sum()), gap_c.std(),
                 gap.corr(tab_s["age"]), gap_c.corr(tab_s["age"]))

    # ---------------------------------------------------------------- Phase 3
    meth_tab = None
    if not args.skip_methylation:
        banner("PHASE 3 -- methylation arm (separate cohort)")
        try:
            meth_tab = methylation.build_methylation_table()
            meth_tab.to_parquet(C.INTERIM / "methylation_table.parquet")
            spread = methylation.residual_spread_table(meth_tab)
            spread.to_csv(C.TABLES / "methylation_clock_comparison.csv", index=False)
            log.info("\n%s", spread.round(3).to_string(index=False))
            manifest["methylation_n"] = int(len(meth_tab))
            manifest["methylation_cohort"] = C.METHYLATION_COHORT
        except Exception as exc:
            log.error("methylation arm failed: %s", exc)
            manifest["methylation_error"] = str(exc)

    # ---------------------------------------------------------------- Phase 4
    banner("PHASE 4 -- combiner fitted against MORTALITY")
    gap_df = pd.DataFrame({f"gap_{k}": v for k, v in gaps.items()})
    combiner = scoring.fit_combiner(gap_df, outcome, method=args.combiner)

    combined_gap, used_counts = [], []
    for i in gap_df.index:
        g = {m: gap_df.loc[i, f"gap_{m}"] for m in combiner.modalities}
        val, used, _ = combiner.predict_gap(g)
        combined_gap.append(val)
        used_counts.append(len(used))
    gaps["combined"] = pd.Series(combined_gap, index=gap_df.index)
    bioages["combined"] = tab_s["age"] + gaps["combined"]
    manifest["combiner_weights"] = {k: round(v, 4) for k, v in combiner.weights.items()}
    manifest["combiner_n_train"] = combiner.n_train
    manifest["combiner_n_events"] = combiner.n_events

    pickle.dump(combiner, open(C.PROCESSED / "combiner.pkl", "wb"))

    # Deployable scoring artifact: everything needed to score a NEW person with
    # no cohort data present. Also the source of truth the browser mirrors.
    bundle = scorer.ScoringBundle(curves=fitted, params=mod_params, combiner=combiner)
    scorer.save(bundle)
    bundle.to_json(C.PROCESSED / "scoring_bundle.json")
    log.info("scoring bundle saved (%d modalities, %d usable curves)",
             len(mod_params), sum(len(v) for v in bundle.to_json()["curves"].values()))
    out_scores = pd.DataFrame({
        "age": tab_s["age"], "sex": tab_s["sex"],
        **{f"bioage_{k}": v for k, v in bioages.items()},
        **{f"gap_{k}": v for k, v in gaps.items()},
        "died": outcome["died"], "followup_years": outcome["time"],
        "n_modalities": used_counts,
    })
    out_scores.to_parquet(C.PROCESSED / "scores.parquet")

    # ---------------------------------------------------------------- Phase 5
    banner("PHASE 5 -- validation")
    vt = validate.validation_table(gaps, outcome)
    vt.to_csv(C.TABLES / "validation.csv", index=False)
    log.info("\n%s", vt.round(4).to_string(index=False))

    singles = vt[vt["score"].isin(C.MODALITIES) & vt["hr_per_sd"].notna()]
    comb = vt[vt["score"] == "combined"]
    if len(singles) and len(comb):
        best = singles.loc[singles["c_index"].idxmax()]
        c = comb.iloc[0]
        manifest["best_single_modality"] = str(best["score"])
        manifest["best_single_c_index"] = round(float(best["c_index"]), 4)
        manifest["combined_c_index"] = round(float(c["c_index"]), 4)
        manifest["combined_beats_best_single"] = bool(c["c_index"] > best["c_index"])
        log.info("HEADLINE: combined C=%.4f vs best single (%s) C=%.4f -> %s",
                 c["c_index"], best["score"], best["c_index"],
                 "FUSION WINS" if c["c_index"] > best["c_index"] else "no gain")

    corr = validate.redundancy({k: v for k, v in gaps.items() if k != "combined"})
    corr.to_csv(C.TABLES / "redundancy.csv")
    log.info("redundancy:\n%s", corr.round(3).to_string())
    manifest["gap_correlations"] = {
        f"{a}|{b}": round(float(corr.loc[a, b]), 4)
        for i, a in enumerate(corr.index) for b in corr.columns[i + 1:]
    }

    dis = validate.disagreement_cases(gaps, tab_s, outcome)
    dis.to_csv(C.TABLES / "disagreement_cases.csv")
    if not dis.empty:
        log.info("top disagreement cases:\n%s", dis.head(8).to_string())

    validate.plot_gap_distributions(gaps, C.FIGURES / "gap_distributions.png")
    validate.plot_redundancy(corr, C.FIGURES / "redundancy.png")
    validate.plot_validation(vt, C.FIGURES / "validation.png")
    validate.plot_km_by_tertile(gaps["combined"], outcome, C.FIGURES / "km_combined.png")
    validate.plot_disagreement(gaps, C.FIGURES / "disagreement.png")

    (C.OUTPUTS / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    banner("DONE -- outputs in " + str(C.OUTPUTS))
    log.info(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
