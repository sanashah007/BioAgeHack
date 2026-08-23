#!/usr/bin/env python
"""Phase 7 -- personalized recommendations on top of the bioage report.

    # score raw markers directly -- the path that matters
    python scripts/recommend.py --markers markers.json --age 41 --sex male

    # replay a cohort participant (channel-level advice only, unless
    # data/interim/analytic_table.parquet exists locally -- see below)
    python scripts/recommend.py --seqn 31311
    python scripts/recommend.py --demo            # the strongest disagreement case

    # score arbitrary precomputed gaps, same --input shape as bioage_report.py
    python scripts/recommend.py --input person.json

    # any mode
    python scripts/recommend.py --demo --json
    python scripts/recommend.py --demo --out report.json

`markers.json` for --markers is `{"blood": {"crp": 0.34, "hba1c": 5.7, ...}}`,
keyed by the feature names in bioage.config.BLOOD_FEATURES / WEARABLE_FEATURES
(not lab-report names -- those live in bioage.recommendations.bioage_bridge).

This calls a real LLM API and costs real money per run -- provider defaults
to any OpenAI-compatible endpoint (RECOMMENDATIONS_PROVIDER=openai_compatible,
the default) or Anthropic direct/via a compatible gateway
(RECOMMENDATIONS_PROVIDER=anthropic); see .env.example for the full set of
options. Grading a panel with no LLM call is available via
bioage.recommendations.evaluate_profile if you just want to sanity-check
marker parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252; the reference-range strings this module
# prints contain U+2264 (<=) and U+2013 (en dash).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from bioage import config as C  # noqa: E402
from bioage.recommendations import HealthProfile, generate_recommendations  # noqa: E402
from bioage.recommendations.bioage_bridge import (  # noqa: E402
    report_to_profile,
    score_markers,
)
from bioage.recommendations.service import RecommendationError  # noqa: E402

from bioage_report import build_report, load_artifacts  # noqa: E402


def _raw_features_for_seqn(seqn: int) -> dict | None:
    """Pull a cohort participant's raw analyte values, if they are available.

    scores.parquet (committed) holds only modality-level gaps. Raw values live
    in data/interim/analytic_table.parquet, which only exists if someone has
    run scripts/run_pipeline.py locally -- it is gitignored and regenerated
    every run. Return None rather than raising: the caller degrades to
    channel-level advice and warns.
    """
    path = C.INTERIM / "analytic_table.parquet"
    if not path.exists():
        return None

    import pandas as pd

    tab = pd.read_parquet(path)
    if seqn not in tab.index:
        return None
    row = tab.loc[seqn]
    out: dict[str, float] = {}
    for feat in C.BLOOD_FEATURES:
        if feat.var in row.index and pd.notna(row[feat.var]):
            out[feat.name] = float(row[feat.var])
    return out or None


def _report_from_cohort(seqn: int | None, demo: bool) -> tuple[dict, dict | None]:
    """(report, raw_features) for --seqn / --demo, via the existing report builder."""
    scores, combiner = load_artifacts()

    if demo:
        avail = scores.dropna(subset=["gap_blood", "gap_wearable"])
        seqn = (avail["gap_blood"] - avail["gap_wearable"]).abs().idxmax()
    if seqn not in scores.index:
        print(f"SEQN {seqn} not in the scored cohort", file=sys.stderr)
        raise SystemExit(1)

    row = scores.loc[seqn]
    import pandas as pd

    age, sex = float(row["age"]), str(row["sex"])
    gaps = {
        m: (float(row[f"gap_{m}"]) if f"gap_{m}" in row and pd.notna(row[f"gap_{m}"])
            else None)
        for m in combiner.modalities
    }
    report = build_report(age, sex, gaps, combiner, subject=f"NHANES SEQN {seqn}")
    return report, _raw_features_for_seqn(seqn)


def _report_from_input(path: Path) -> tuple[dict, dict | None]:
    """(report, raw_features) for --input, same JSON shape bioage_report.py takes."""
    _, combiner = load_artifacts()
    payload = json.loads(path.read_text(encoding="utf-8"))
    age, sex = float(payload["age"]), payload.get("sex", "unknown")
    gaps = {m: payload.get(f"gap_{m}") for m in combiner.modalities}
    report = build_report(age, sex, gaps, combiner, subject=payload.get("id", str(path)))
    return report, payload.get("raw_features")


def _report_from_markers(path: Path, age: float, sex: str) -> tuple[dict, dict | None]:
    """(report, raw_features) for --markers, via the shared scoring helper."""
    values = json.loads(path.read_text(encoding="utf-8"))
    report, _profile = score_markers(age=age, sex=sex, markers=values)
    return report, values.get("blood")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seqn", type=int, help="cohort participant by SEQN")
    ap.add_argument("--demo", action="store_true",
                     help="the strongest modality-disagreement cohort case")
    ap.add_argument("--input", type=Path,
                     help="JSON with age, sex and precomputed modality gaps "
                          "(same shape as bioage_report.py --input)")
    ap.add_argument("--markers", type=Path,
                     help="JSON of raw marker values, scored via bioage.scorer")
    ap.add_argument("--age", type=float, help="required with --markers")
    ap.add_argument("--sex", choices=("male", "female"), help="required with --markers")
    ap.add_argument("--json", action="store_true", help="emit the recommendation JSON")
    ap.add_argument("--out", type=Path, help="write output to a file instead of stdout")
    args = ap.parse_args()

    modes = [args.seqn is not None, args.demo, args.input is not None,
              args.markers is not None]
    if sum(modes) != 1:
        ap.error("provide exactly one of --seqn, --demo, --input, --markers")

    if args.markers:
        if args.age is None or args.sex is None:
            ap.error("--markers requires --age and --sex")
        report, raw_features = _report_from_markers(args.markers, args.age, args.sex)
    elif args.input:
        report, raw_features = _report_from_input(args.input)
    else:
        report, raw_features = _report_from_cohort(args.seqn, args.demo)
        if raw_features is None:
            print(
                f"warning: no raw marker values for {report['subject']} -- "
                "recommendations will address channel-level gaps only. "
                "Use --markers to score a full panel instead.",
                file=sys.stderr,
            )

    profile: HealthProfile = report_to_profile(report, raw_features=raw_features)

    try:
        result = generate_recommendations(profile)
    except RecommendationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = (
        json.dumps(result.response.model_dump(), indent=2, ensure_ascii=False)
        if args.json
        else _render_text(result.response)
    )

    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"written to {args.out}", file=sys.stderr)
    else:
        print(output)

    cost = (f"~${result.estimated_cost_usd:.3f}" if result.estimated_cost_usd is not None
            else "unknown -- non-Anthropic provider, no pricing table for it here")
    print(
        f"{result.input_tokens} in / {result.output_tokens} out tokens ({cost})",
        file=sys.stderr,
    )
    return 0


def _render_text(response) -> str:
    r = response.report
    lines = [r.summary, ""]
    if r.biological_age_drivers:
        lines.append("DRIVERS:")
        lines += [f"  - {d}" for d in r.biological_age_drivers]
        lines.append("")
    if r.top_priorities:
        lines.append("TOP PRIORITIES:")
        for p in r.top_priorities:
            lines.append(f"  {p.rank}. {p.focus}")
            lines.append(f"     {p.why_now}")
        lines.append("")
    for rec in r.recommendations:
        lines.append(f"[{rec.severity.value.upper()}] {rec.marker} "
                      f"({rec.current_value}, target {rec.healthy_range})")
        lines.append(f"  {rec.why_it_matters}")
        for a in rec.actions:
            lines.append(f"  - {a.action} ({a.effort.value}, {a.timeframe})")
        if rec.clinician_flag:
            lines.append("  >> discuss with a clinician")
        lines.append("")
    if r.caveats:
        lines.append("CAVEATS:")
        lines += [f"  - {c}" for c in r.caveats]
        lines.append("")
    lines.append(response.disclaimer)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
