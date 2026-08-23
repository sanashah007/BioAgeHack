"""Score a NEW person against the fitted reference curves.

The pipeline fits curves and a combiner on NHANES. This module is the other
half: it takes one individual's raw marker values and walks them through the
identical chain, so the number they see means what the validation says it means.

    raw value
      -> invert the sex-matched reference curve   -> per-feature implied age
      -> weighted mean of per-feature gaps        -> raw modality gap
      -> shrink by split-half reliability         -> shrunk gap
      -> subtract the cohort's gap-vs-age trend   -> age-acceleration residual
      -> x age-equivalent scale                   -> gap in years of mortality risk
      -> combine across modalities, renormalised  -> combined gap
      -> + chronological age                      -> combined biological age

Every constant in that chain is estimated on the cohort and frozen into a
ScoringBundle, so scoring an individual is deterministic and requires no cohort
data at inference time. The same bundle is exported to JSON and re-implemented
in the browser console; `scripts/test_invariants.py` checks the two agree.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from .curves import ReferenceCurve, invert


@dataclass
class ModalityParams:
    """Frozen constants that place one modality's gap on the cohort scale."""

    reliability: float          # split-half, Spearman-Brown corrected
    gap_mean_raw: float         # centre used when shrinking
    deatt_intercept: float      # gap-vs-age trend fitted on the cohort
    deatt_slope: float
    scale: float                # 1 raw unit = `scale` chronological-age-equiv years
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class ScoringBundle:
    curves: dict[str, dict[str, ReferenceCurve]]   # modality -> "feat|sex" -> curve
    params: dict[str, ModalityParams]              # modality -> constants
    combiner: object                               # scoring.Combiner
    min_features: int = 3

    # ---------------------------------------------------------------- scoring
    def implied_ages(
        self, modality: str, values: dict[str, float], sex: str
    ) -> dict[str, dict]:
        """Invert every supplied marker against its sex-matched curve."""
        out: dict[str, dict] = {}
        for feat, val in values.items():
            if val is None or not np.isfinite(val):
                continue
            curve = self.curves.get(modality, {}).get(f"{feat}|{sex}")
            if curve is None or not curve.usable:
                continue
            age, dist, extrap = invert(float(val), curve)
            if not np.isfinite(age):
                continue
            out[feat] = dict(implied_age=float(age), distance=float(dist),
                             extrapolated=bool(extrap),
                             weight=float(self.params[modality].weights.get(feat, 0.0)))
        return out

    def modality_gap(
        self, modality: str, values: dict[str, float], sex: str, age: float
    ) -> dict:
        """One modality's calibrated gap, plus the per-feature detail."""
        p = self.params[modality]
        per = self.implied_ages(modality, values, sex)
        usable = {k: v for k, v in per.items() if v["weight"] > 0}
        if len(usable) < self.min_features:
            return dict(gap=None, n_features=len(usable), per_feature=per,
                        reason=f"needs at least {self.min_features} weighted markers")

        wsum = sum(v["weight"] for v in usable.values())
        raw = sum(v["weight"] * (v["implied_age"] - age) for v in usable.values()) / wsum

        # Same three transforms the cohort gaps went through, in the same order.
        g = (raw - p.gap_mean_raw) * p.reliability + p.gap_mean_raw
        g = g - (p.deatt_intercept + p.deatt_slope * age)
        g = g * p.scale
        return dict(gap=float(g), raw_gap=float(raw), n_features=len(usable),
                    per_feature=per, reason="")

    def score(
        self, age: float, sex: str, values: dict[str, dict[str, float]]
    ) -> dict:
        """Full report for one person. `values` is {modality: {feature: value}}."""
        gaps, detail = {}, {}
        for mod in C.FEATURES_BY_MODALITY:
            res = self.modality_gap(mod, values.get(mod, {}) or {}, sex, age)
            detail[mod] = res
            gaps[mod] = res["gap"]
        gaps["methylation"] = None  # see methylation.py -- different cohort, no weight

        combined, used, missing = self.combiner.predict_gap(gaps)
        contrib = {}
        if used:
            total = sum(abs(self.combiner.weights.get(m, 0.0))
                        for m in self.combiner.modalities)
            wsum = sum(abs(self.combiner.weights.get(m, 0.0)) for m in used)
            rn = (total / wsum) if wsum > 0 else 1.0
            contrib = {m: self.combiner.weights.get(m, 0.0) * gaps[m] * rn for m in used}

        return dict(
            chronological_age=float(age), sex=sex,
            combined_gap=(None if not np.isfinite(combined) else float(combined)),
            combined_bioage=(None if not np.isfinite(combined) else float(age + combined)),
            per_modality_gap={k: v for k, v in gaps.items() if v is not None},
            contribution=contrib, modalities_used=used, modalities_missing=missing,
            detail=detail,
        )

    # ---------------------------------------------------------------- export
    def to_json(self, path: Path | None = None, grid_step: int = 1) -> dict:
        """Serialise curves + constants for the browser.

        `grid_step` thins the 0.5-year fitting grid; 1 (yearly) halves the
        payload and costs nothing, since inversion interpolates between points
        anyway.
        """
        curves: dict[str, dict] = {}
        for mod, cs in self.curves.items():
            curves[mod] = {}
            for key, c in cs.items():
                if not c.usable:
                    continue
                keep = np.isfinite(c.mean) & np.isfinite(c.sd)
                idx = np.where(keep)[0][::grid_step]
                curves[mod][key] = dict(
                    g=[round(float(c.grid[i]), 1) for i in idx],
                    m=[round(float(c.mean[i]), 5) for i in idx],
                    s=[round(float(c.sd[i]), 5) for i in idx],
                    log=bool(c.log_transform),
                    lo=round(float(c.seg_lo), 1), hi=round(float(c.seg_hi), 1),
                    dir=c.direction, n=int(c.n_obs),
                )
        payload = dict(
            curves=curves,
            params={m: dict(reliability=round(p.reliability, 6),
                            gap_mean_raw=round(p.gap_mean_raw, 6),
                            deatt_intercept=round(p.deatt_intercept, 6),
                            deatt_slope=round(p.deatt_slope, 6),
                            scale=round(p.scale, 6),
                            weights={k: round(v, 6) for k, v in p.weights.items()})
                    for m, p in self.params.items()},
            combiner=dict(weights={k: round(float(v), 6)
                                   for k, v in self.combiner.weights.items()},
                          modalities=list(self.combiner.modalities)),
            min_features=self.min_features,
            features={
                mod: [dict(name=f.name, label=f.label, units=f.units,
                           file=f.file, log=f.log_transform,
                           range=list(f.valid_range) if f.valid_range else None)
                      for f in feats]
                for mod, feats in C.FEATURES_BY_MODALITY.items()
            },
        )
        if path:
            path.write_text(json.dumps(payload, separators=(",", ":")))
        return payload


def save(bundle: ScoringBundle, path: Path | None = None) -> Path:
    path = path or (C.PROCESSED / "scoring_bundle.pkl")
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load(path: Path | None = None) -> ScoringBundle:
    path = path or (C.PROCESSED / "scoring_bundle.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)
