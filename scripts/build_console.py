#!/usr/bin/env python
"""Phase 6b -- build the self-contained interactive console.

    python scripts/build_console.py

Reads the pipeline outputs, serialises them to a compact JSON payload, and
splices that payload into web/console.template.html to produce
web/bioage_console.html.

WHY THE DATA IS INLINED
-----------------------
The page is published as an Artifact, which enforces a strict CSP: no fetch, no
XHR, no external hosts. There is no server to call, so the payload has to travel
inside the document. ~144 KB for 4,623 participants, which is cheap. The upside
is that the built file is genuinely standalone -- open it from disk, mail it,
drop it on any static host, and it works with no backend.

The template is kept separate from the built file on purpose: committing a
173 KB generated blob with no generator would leave nobody able to change it.
Edit the template, re-run this script.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioage import config as C  # noqa: E402

WEB = C.ROOT / "web"
TEMPLATE = WEB / "console.template.html"
BUILT = WEB / "bioage_console.html"
PLACEHOLDER = "__DATA__"


def _n(x, nd: int = 2):
    return None if pd.isna(x) else round(float(x), nd)


def build_payload() -> dict:
    scores = pd.read_parquet(C.PROCESSED / "scores.parquet")
    combiner = pickle.load(open(C.PROCESSED / "combiner.pkl", "rb"))
    validation = pd.read_csv(C.TABLES / "validation.csv")
    clocks = pd.read_csv(C.TABLES / "methylation_clock_comparison.csv")
    redundancy = pd.read_csv(C.TABLES / "redundancy.csv", index_col=0)
    weights = {
        m: pd.read_csv(C.TABLES / f"feature_weights_{m}.csv", index_col=0)
        for m in ("blood", "wearable")
    }

    # Positional rows rather than objects: ~3x smaller than repeating key names
    # 4,623 times. Column order is mirrored by the destructuring in the page JS.
    people = [
        [int(seqn), int(r["age"]), 1 if r["sex"] == "female" else 0,
         _n(r["gap_blood"]), _n(r["gap_wearable"]),
         None if pd.isna(r["died"]) else int(r["died"]), _n(r["followup_years"], 1)]
        for seqn, r in scores.iterrows()
    ]

    # Cases worth opening the app for: the largest standardised disagreements
    # between the arms, plus one concordant extreme at each end for contrast.
    both = scores.dropna(subset=["gap_blood", "gap_wearable"]).copy()
    cols = ["gap_blood", "gap_wearable"]
    z = (both[cols] - both[cols].mean()) / both[cols].std()
    both["spread"] = z.max(axis=1) - z.min(axis=1)
    notable = [
        {"seqn": int(seqn),
         "label": "fit body, ailing labs" if r["gap_wearable"] < r["gap_blood"]
                  else "clean labs, sedentary"}
        for seqn, r in both.nlargest(8, "spread").iterrows()
    ]
    total = both["gap_blood"] + both["gap_wearable"]
    notable.append({"seqn": int(total.idxmax()), "label": "both arms aging fast"})
    notable.append({"seqn": int(total.idxmin()), "label": "both arms aging slow"})

    feat = {
        m: [{"name": i, "z": _n(r["z"]), "w": _n(r["final_weight"], 4)}
            for i, r in df.sort_values("final_weight", ascending=False).head(6).iterrows()
            if r["final_weight"] > 0]
        for m, df in weights.items()
    }

    vcols = ["score", "n", "events", "gap_sd_years", "hr_per_year", "hr_per_sd",
             "hr_lo", "hr_hi", "c_index", "c_index_age_sex_only", "c_gain",
             "auc_gap_alone"]
    return dict(
        P=people,
        W={k: round(float(v), 4) for k, v in combiner.weights.items()},
        MOD=list(combiner.modalities),
        V=[{k: (_n(r[k], 4) if isinstance(r[k], (int, float, np.floating)) else r[k])
            for k in vcols} for _, r in validation.iterrows()],
        MC=[{k: (_n(r[k], 3) if k not in ("clock", "generation") else r[k])
             for k in clocks.columns} for _, r in clocks.iterrows()],
        RED=_n(redundancy.loc["blood", "wearable"], 3),
        FEAT=feat,
        NOTABLE=notable,
    )


def main() -> int:
    if not TEMPLATE.exists():
        print(f"missing template: {TEMPLATE}", file=sys.stderr)
        return 1
    if not (C.PROCESSED / "scores.parquet").exists():
        print("no scores found -- run scripts/run_pipeline.py first", file=sys.stderr)
        return 1

    payload = build_payload()
    blob = json.dumps(payload, separators=(",", ":"))

    # </script> anywhere inside the JSON would close the host <script> tag early
    # and break the page. Escaping the slash keeps it valid JSON and inert HTML.
    blob = blob.replace("</", "<\\/")

    html = TEMPLATE.read_text()
    if PLACEHOLDER not in html:
        print(f"template has no {PLACEHOLDER} placeholder", file=sys.stderr)
        return 1
    BUILT.write_text(html.replace(PLACEHOLDER, blob))

    print(f"built {BUILT.relative_to(C.ROOT)}")
    print(f"  {len(payload['P']):,} participants | payload {len(blob)/1024:.0f} KB "
          f"| page {BUILT.stat().st_size/1024:.0f} KB")
    print(f"  combiner weights: {payload['W']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
