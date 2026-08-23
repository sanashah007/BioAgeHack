#!/usr/bin/env python
"""Phase 6 -- the output interface.

    # score a cohort participant (demo)
    python scripts/bioage_report.py --seqn 31311
    python scripts/bioage_report.py --demo            # a curated disagreement case
    python scripts/bioage_report.py --seqn 31311 --drop wearable   # missing-modality

    # score arbitrary input
    python scripts/bioage_report.py --input person.json
    python scripts/bioage_report.py --seqn 31311 --json

The report always surfaces the PER-MODALITY BREAKDOWN alongside the combined
number, never the combined number alone. A single scalar ("your bioage is 34")
is the weaker product: it is not actionable, and it hides the case that actually
matters -- one system aging faster than another. "Your cardiovascular signal is
aging faster than your metabolic signal" is what a person can do something about.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioage import config as C, curves  # noqa: E402

BAR_W = 22


def _bar(gap: float, lo: float = -15, hi: float = 15) -> str:
    """Centred text bar; left of centre = biologically younger."""
    mid = BAR_W // 2
    cells = [" "] * BAR_W
    cells[mid] = "|"
    pos = int(round(mid + np.clip(gap, lo, hi) / hi * mid))
    pos = int(np.clip(pos, 0, BAR_W - 1))
    ch = "#" if gap >= 0 else "="
    for i in range(min(mid, pos), max(mid, pos) + 1):
        cells[i] = ch
    cells[mid] = "|" if pos == mid else cells[mid]
    return "".join(cells)


def load_artifacts():
    scores = pd.read_parquet(C.PROCESSED / "scores.parquet")
    combiner = pickle.load(open(C.PROCESSED / "combiner.pkl", "rb"))
    return scores, combiner


def build_report(
    age: float,
    sex: str,
    gaps: dict[str, float | None],
    combiner,
    *,
    subject: str = "subject",
) -> dict:
    """Assemble the full report structure from a per-modality gap vector."""
    present = {m: v for m, v in gaps.items()
               if v is not None and np.isfinite(v)}
    combined_gap, used, missing = combiner.predict_gap(gaps)

    contributions = {}
    if used:
        # The combiner is linear on gaps, so "what is driving this score" is
        # just weight x gap -- read directly off the model, not reconstructed
        # by a separate attribution method.
        total_w = sum(abs(combiner.weights.get(m, 0.0)) for m in combiner.modalities)
        used_w = sum(abs(combiner.weights.get(m, 0.0)) for m in used)
        renorm = (total_w / used_w) if used_w > 0 else 1.0
        for m in used:
            contributions[m] = combiner.weights.get(m, 0.0) * present[m] * renorm

    driver = max(contributions, key=lambda k: abs(contributions[k])) if contributions else None
    return dict(
        subject=subject,
        chronological_age=round(float(age), 1),
        sex=sex,
        combined_bioage=(round(float(age + combined_gap), 1)
                         if np.isfinite(combined_gap) else None),
        combined_gap=(round(float(combined_gap), 1)
                      if np.isfinite(combined_gap) else None),
        per_modality_gap={m: round(float(v), 1) for m, v in present.items()},
        modality_contribution_years={m: round(float(v), 2)
                                     for m, v in contributions.items()},
        combiner_weights={m: round(float(combiner.weights.get(m, 0.0)), 3)
                          for m in combiner.modalities},
        modalities_used=used,
        modalities_missing=missing,
        driver=driver,
    )


def render(rep: dict) -> str:
    L: list[str] = []
    W = 66
    L.append("=" * W)
    L.append(f"  BIOLOGICAL AGE REPORT   —   {rep['subject']}")
    L.append("=" * W)
    L.append(f"  Chronological age : {rep['chronological_age']:.1f}   ({rep['sex']})")

    if rep["combined_bioage"] is None:
        L.append("  Combined bioage   : UNAVAILABLE — no usable modality supplied")
        L.append("=" * W)
        return "\n".join(L)

    g = rep["combined_gap"]
    verdict = "older" if g > 0 else "younger"
    L.append(f"  Combined bioage   : {rep['combined_bioage']:.1f}"
             f"   ({abs(g):.1f}y {verdict} than chronological)")
    L.append("")
    L.append("  PER-MODALITY BREAKDOWN")
    L.append(f"    {'modality':<13} {'gap':>8}   {'younger  |  older':^{BAR_W}}   {'weight':>7}")
    for m in ("blood", "wearable", "methylation"):
        if m in rep["per_modality_gap"]:
            gm = rep["per_modality_gap"][m]
            w = rep["combiner_weights"].get(m)
            wtxt = f"{w:.3f}" if w is not None else "  n/a"
            L.append(f"    {m:<13} {gm:>+7.1f}y   {_bar(gm)}   {wtxt:>7}")
        else:
            L.append(f"    {m:<13} {'—':>8}   {'not supplied':^{BAR_W}}   {'':>7}")

    if rep["driver"]:
        contrib = rep["modality_contribution_years"][rep["driver"]]
        L.append("")
        L.append(f"  DRIVER: {rep['driver']} "
                 f"({contrib:+.1f}y of the {g:+.1f}y combined gap)")

    gaps = rep["per_modality_gap"]
    if len(gaps) >= 2:
        hi = max(gaps, key=lambda k: gaps[k])
        lo = min(gaps, key=lambda k: gaps[k])
        if gaps[hi] - gaps[lo] > 5:
            L.append(f"  DIVERGENCE: {hi} is aging {gaps[hi] - gaps[lo]:.1f}y "
                     f"faster than {lo} — the modalities disagree.")

    L.append("")
    L.append(f"  Modalities used   : {', '.join(rep['modalities_used']) or 'none'}")
    if rep["modalities_missing"]:
        L.append(f"  Modalities MISSING: {', '.join(rep['modalities_missing'])}")
        L.append("                      (score uses the available arms only,")
        L.append("                       renormalised — not imputed as average)")
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqn", type=int, help="score an NHANES participant by SEQN")
    ap.add_argument("--demo", action="store_true",
                    help="score the strongest modality-disagreement case")
    ap.add_argument("--input", type=Path, help="JSON with age, sex and modality gaps")
    ap.add_argument("--drop", action="append", default=[],
                    help="drop a modality, to demo graceful degradation")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    scores, combiner = load_artifacts()

    if args.input:
        payload = json.loads(args.input.read_text())
        age, sex = float(payload["age"]), payload.get("sex", "unknown")
        gaps = {m: payload.get(f"gap_{m}") for m in combiner.modalities}
        subject = payload.get("id", str(args.input))
    else:
        if args.demo:
            avail = scores.dropna(subset=["gap_blood", "gap_wearable"])
            seqn = (avail["gap_blood"] - avail["gap_wearable"]).abs().idxmax()
        elif args.seqn:
            seqn = args.seqn
        else:
            ap.error("provide --seqn, --demo or --input")
        if seqn not in scores.index:
            print(f"SEQN {seqn} not in the scored cohort", file=sys.stderr)
            return 1
        row = scores.loc[seqn]
        age, sex = float(row["age"]), str(row["sex"])
        gaps = {m: (float(row[f"gap_{m}"]) if f"gap_{m}" in row
                    and pd.notna(row[f"gap_{m}"]) else None)
                for m in combiner.modalities}
        subject = f"NHANES SEQN {seqn}"

    for m in args.drop:
        gaps[m] = None

    rep = build_report(age, sex, gaps, combiner, subject=subject)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
