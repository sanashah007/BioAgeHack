#!/usr/bin/env python
"""Invariant checks for the bioage pipeline.

These are not unit tests of arithmetic -- they are guards on the properties that
make the output meaningful, and that would fail SILENTLY and plausibly if broken.

    python scripts/test_invariants.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from bioage import config as C, curves, nhanes, scoring  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"{PASS if cond else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    print("=" * 74)
    print("  BIOAGE PIPELINE INVARIANTS")
    print("=" * 74)

    # ---------------------------------------------------------------- linkage
    print("\n[mortality linkage]")
    mort = nhanes.load_mortality()
    check("SEQN within the NHANES 2005-2006 range",
          bool(mort.index.min() >= 31127 and mort.index.max() <= 41474),
          f"{mort.index.min()}-{mort.index.max()}")
    check("MORTSTAT is binary or missing",
          set(mort["MORTSTAT"].dropna().unique()) <= {0, 1})
    check("follow-up does not exceed the 2019 horizon",
          bool(mort["PERMTH_EXM"].max() <= 180), f"max {mort['PERMTH_EXM'].max():.0f} months")
    check("only mortality-eligible records carry a vital status",
          bool(mort.loc[mort["ELIGSTAT"] != 1, "MORTSTAT"].isna().all()))
    n_deaths = int(mort["MORTSTAT"].sum())
    check("death count is in a plausible range", 500 < n_deaths < 2000, f"{n_deaths} deaths")

    # ------------------------------------------------------------ topcode/age
    print("\n[demographics]")
    demo = nhanes.load_demographics()
    check("age top-code matches config",
          float(demo["age"].max()) == float(C.AGE_TOPCODE),
          f"observed {demo['age'].max():.0f}, config {C.AGE_TOPCODE}")
    check("analysis window excludes the top-code", C.AGE_MAX < C.AGE_TOPCODE)

    # ----------------------------------------------------------------- curves
    print("\n[reference curves]")
    cpath = C.INTERIM / "curves.pkl"
    if not cpath.exists():
        print("  (skipped -- run scripts/run_pipeline.py first)")
    else:
        fitted = pickle.load(open(cpath, "rb"))
        allc = {**fitted["blood"], **fitted["wearable"]}
        check("curves were fitted separately by sex",
              all(any(f"{k.split('|')[0]}|{s}" in allc for s in ("male", "female"))
                  for k in allc))
        usable = [c for c in allc.values() if c.usable]
        check("every usable curve has a monotonic span >= the minimum",
              all(c.seg_hi - c.seg_lo >= C.MIN_MONOTONIC_SPAN - 1e-6 for c in usable),
              f"{len(usable)} usable curves")
        check("inversion is confined to the usable segment",
              all(np.isfinite(c.seg_lo) and np.isfinite(c.seg_hi) for c in usable))

        # Inversion must be a genuine inverse: feeding a curve its own expected
        # value at age a must return approximately a.
        errs = []
        for c in usable[:40]:
            mid = (c.seg_lo + c.seg_hi) / 2
            i = int(np.argmin(np.abs(c.grid - mid)))
            if not np.isfinite(c.mean[i]):
                continue
            v = np.exp(c.mean[i]) if c.log_transform else c.mean[i]
            got, _, _ = curves.invert(v, c)
            if np.isfinite(got):
                errs.append(abs(got - c.grid[i]))
        check("inverting a curve's own value recovers its age",
              bool(errs) and float(np.median(errs)) < 1.0,
              f"median error {np.median(errs):.3f} y over {len(errs)} curves")

    # ------------------------------------------------------------ the big rule
    print("\n[the governing rule: no chronological-age target]")
    idx = pd.Index(range(400), name="SEQN")
    rng = np.random.default_rng(0)
    fake_gaps = pd.DataFrame({"gap_blood": rng.normal(size=400),
                              "gap_wearable": rng.normal(size=400)}, index=idx)
    no_outcome = pd.DataFrame({"died": [np.nan] * 400, "time": [5.0] * 400,
                               "age": rng.uniform(30, 80, 400),
                               "sex": ["male"] * 400}, index=idx)
    try:
        scoring.fit_combiner(fake_gaps, no_outcome)
        check("combiner refuses to fit without mortality", False,
              "it returned a model instead of raising")
    except RuntimeError as exc:
        check("combiner refuses to fit without mortality", True, str(exc)[:52])

    # ---------------------------------------------- graceful missing-modality
    print("\n[graceful degradation]")
    cpath = C.PROCESSED / "combiner.pkl"
    if not cpath.exists():
        print("  (skipped -- run scripts/run_pipeline.py first)")
    else:
        comb = pickle.load(open(cpath, "rb"))
        full, used_f, miss_f = comb.predict_gap({"blood": 4.0, "wearable": 4.0})
        one, used_o, miss_o = comb.predict_gap({"blood": 4.0, "wearable": None})
        none_, used_n, _ = comb.predict_gap({"blood": None, "wearable": None})
        check("single-modality input still yields a score",
              np.isfinite(one) and used_o == ["blood"], f"gap {one:+.2f} y")
        check("missing modalities are reported, not hidden", miss_o == ["wearable"])
        check("all-missing input returns no score rather than a fake one",
              (not np.isfinite(none_)) and used_n == [])
        # Renormalisation, not zero-imputation: with both gaps equal, dropping
        # one must leave the combined gap unchanged rather than halving it.
        check("dropping a modality renormalises instead of imputing zero",
              abs(full - one) < 1e-9,
              f"both={full:+.3f} vs blood-only={one:+.3f}")

    # --------------------------------------------------------- scale sanity
    print("\n[reported scale]")
    spath = C.PROCESSED / "scores.parquet"
    if not spath.exists():
        print("  (skipped -- run scripts/run_pipeline.py first)")
    else:
        s = pd.read_parquet(spath)
        for m in ("blood", "wearable", "combined"):
            col = f"gap_{m}"
            if col not in s:
                continue
            sd = float(s[col].std())
            check(f"{m} gap SD is physiologically plausible (2-12 y)",
                  2.0 < sd < 12.0, f"SD {sd:.2f} y")
            r = float(s[col].corr(s["age"]))
            check(f"{m} gap is not a proxy for chronological age (|r| < 0.2)",
                  abs(r) < 0.2, f"corr {r:+.3f}")

    print("\n" + "=" * 74)
    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"  {len(results) - n_fail}/{len(results)} passed"
          + (f", {n_fail} FAILED" if n_fail else ""))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
