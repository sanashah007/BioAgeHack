#!/usr/bin/env python
"""Phase 6b -- build the self-contained scoring console.

    python scripts/build_console.py

Serialises the fitted reference curves and the frozen calibration constants into
web/bioage_console.html, which then scores an individual entirely in the browser.

WHY EVERYTHING IS INLINED
-------------------------
The page runs under a strict CSP with no fetch and no server to call, so the
curves have to travel inside the document (~180 KB). The upside is a genuinely
standalone file: open it from disk, mail it, host it anywhere, and nobody's
health data leaves their machine -- scoring is pure client-side arithmetic.

The template is kept separate from the built file on purpose: committing a
generated blob with no generator leaves nobody able to change it. Edit
web/console.template.html, re-run this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioage import config as C, scorer  # noqa: E402

WEB = C.ROOT / "web"
TEMPLATE = WEB / "console.template.html"
BUILT = WEB / "bioage_console.html"
PLACEHOLDER = "__DATA__"

# Panels group the markers the way a lab report does, so someone holding a blood
# panel can find their values instead of hunting an alphabetical list.
GROUPS = {
    "blood": [
        ("Metabolic", ["glucose_fasting", "hba1c", "triglycerides", "hdl",
                       "total_chol", "waist"]),
        ("Kidney & liver", ["creatinine", "urea_nitrogen", "uric_acid", "albumin",
                            "total_protein", "alk_phos", "alt", "ggt"]),
        ("Blood count", ["hemoglobin", "rdw", "mcv", "wbc", "lymph_pct"]),
        ("Inflammation & vascular", ["crp", "systolic_bp", "pulse_pressure"]),
    ],
    "wearable": [
        ("Activity volume", ["mean_cpm", "mvpa_min", "m10", "l5"]),
        ("Pattern", ["sedentary_frac", "activity_bout_frag", "activity_cv",
                     "relative_amplitude", "intradaily_variability",
                     "interdaily_stability"]),
    ],
}


def _raw_values(tab: pd.DataFrame, seqn: int) -> dict:
    r = tab.loc[seqn]
    out = {}
    for mod, feats in C.FEATURES_BY_MODALITY.items():
        d = {}
        for f in feats:
            col = f"pax_{f.var}" if f.file == "PAX" else f.var
            if col in tab.columns and pd.notna(r.get(col)):
                d[f.name] = round(float(r[col]), 4)
        out[mod] = d
    return out


def build_payload() -> dict:
    bundle = scorer.load()
    payload = bundle.to_json(grid_step=1)
    payload["groups"] = GROUPS

    # The interim analytic table holds raw per-participant values and is used
    # below only to build worked examples. It is gitignored and only produced
    # by a full scripts/run_pipeline.py run (~3.3 GB), unlike scoring_bundle
    # and scores.parquet, which are committed. When it's absent, fall back to
    # a small committed snapshot of the same 5 examples instead (see
    # console_examples.json below) rather than require the full download just
    # to rebuild the console after an unrelated template edit.
    analytic_table = C.INTERIM / "analytic_table.parquet"
    tab = pd.read_parquet(analytic_table) if analytic_table.exists() else None
    scores = pd.read_parquet(C.PROCESSED / "scores.parquet")

    # Percentile ladders, so a gap can be reported as a position in the cohort
    # rather than only as a bare number of years.
    pct = {}
    for m in ("blood", "wearable", "combined"):
        col = f"gap_{m}"
        if col in scores:
            v = scores[col].dropna().to_numpy()
            pct[m] = [round(float(x), 3)
                      for x in np.percentile(v, np.arange(0, 101))]
    payload["pct"] = pct

    # Worked examples, so the page is usable the moment it opens and so anyone
    # can check the arithmetic against a real participant. Needs the raw
    # per-participant values in `tab`; see the note above for why that can be
    # absent.
    #
    # Fallback order when `tab` is absent: a committed snapshot of the same 5
    # examples (data/processed/console_examples.json -- ~6 KB, real NHANES
    # participants, extracted once from a build that did have the full
    # pipeline run) rather than shipping an empty "worked example" row. If
    # that snapshot is also missing, fall back to no examples rather than fail
    # the whole build -- the console is still fully usable with none.
    examples = []
    example_fallback = C.PROCESSED / "console_examples.json"
    if tab is not None:
        both = scores.dropna(subset=["gap_blood", "gap_wearable"])
        z = ((both[["gap_blood", "gap_wearable"]] - both[["gap_blood", "gap_wearable"]].mean())
             / both[["gap_blood", "gap_wearable"]].std())
        spread = (z.max(axis=1) - z.min(axis=1)).sort_values(ascending=False)
        picks: list[tuple[int, str]] = []
        for seqn in spread.index[:40]:
            r = scores.loc[seqn]
            n_blood = len(_raw_values(tab, seqn)["blood"])
            if n_blood >= 12:
                lab = ("fit body, ailing labs" if r["gap_wearable"] < r["gap_blood"]
                       else "clean labs, sedentary")
                picks.append((int(seqn), lab))
            if len(picks) >= 3:
                break
        tot = (both["gap_blood"] + both["gap_wearable"]).sort_values()
        for seqn, lab in ((tot.index[-1], "both arms aging fast"),
                          (tot.index[0], "both arms aging slow")):
            if len(_raw_values(tab, int(seqn))["blood"]) >= 12:
                picks.append((int(seqn), lab))

        for seqn, lab in picks:
            r = scores.loc[seqn]
            examples.append(dict(
                seqn=int(seqn), label=lab, age=int(r["age"]), sex=str(r["sex"]),
                values=_raw_values(tab, seqn),
                expected={m: (None if pd.isna(r[f"gap_{m}"]) else round(float(r[f"gap_{m}"]), 2))
                          for m in ("blood", "wearable", "combined")},
            ))
    elif example_fallback.exists():
        examples = json.loads(example_fallback.read_text(encoding="utf-8"))
        print(f"note: data/interim/analytic_table.parquet not found -- using "
              f"the {len(examples)} committed example(s) from "
              f"{example_fallback.relative_to(C.ROOT)} instead. Run "
              f"scripts/run_pipeline.py for a fresh pick.", file=sys.stderr)
    else:
        print("note: data/interim/analytic_table.parquet not found and no "
              f"fallback at {example_fallback.relative_to(C.ROOT)} -- building "
              "without worked examples. Run scripts/run_pipeline.py once to "
              "include them.", file=sys.stderr)
    payload["examples"] = examples

    validation = pd.read_csv(C.TABLES / "validation.csv")
    payload["V"] = [
        {k: (None if pd.isna(r[k]) else (round(float(r[k]), 4)
             if isinstance(r[k], (int, float, np.floating)) else r[k]))
         for k in ["score", "hr_per_year", "hr_per_sd", "hr_lo", "hr_hi",
                   "c_index", "c_gain"]}
        for _, r in validation.iterrows()
    ]
    red = pd.read_csv(C.TABLES / "redundancy.csv", index_col=0)
    payload["RED"] = round(float(red.loc["blood", "wearable"]), 3)
    return payload


def main() -> int:
    if not TEMPLATE.exists():
        print(f"missing template: {TEMPLATE}", file=sys.stderr)
        return 1
    if not (C.PROCESSED / "scoring_bundle.pkl").exists():
        print("no scoring bundle -- run scripts/run_pipeline.py first", file=sys.stderr)
        return 1

    payload = build_payload()
    blob = json.dumps(payload, separators=(",", ":"))
    # A literal </ inside the JSON would close the host <script> early. Escaping
    # the slash keeps it valid JSON and inert HTML.
    blob = blob.replace("</", "<\\/")

    html = TEMPLATE.read_text()
    if PLACEHOLDER not in html:
        print(f"template has no {PLACEHOLDER} placeholder", file=sys.stderr)
        return 1
    BUILT.write_text(html.replace(PLACEHOLDER, blob))

    n_curves = sum(len(v) for v in payload["curves"].values())
    print(f"built {BUILT.relative_to(C.ROOT)}")
    print(f"  {n_curves} curves | {len(payload['examples'])} examples "
          f"| payload {len(blob)/1024:.0f} KB | page {BUILT.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
