"""LLM-generated health recommendations for a biological age report.

Public surface:

    from bioage.recommendations import HealthProfile, generate_recommendations

`router` (the FastAPI surface) is deliberately NOT imported here — it pulls in
`fastapi`, which is only a dependency once the HTTP service is wired up. Import
it directly when you need it: `from bioage.recommendations.router import router`.
Everything else in this package works with no web framework installed.
"""

from .prompt import DISCLAIMER
from .reference_ranges import evaluate_profile
from .schemas import (
    BiologicalAgeResult,
    BloodMarker,
    Demographics,
    HealthProfile,
    Lifestyle,
    MethylationResult,
    RecommendationReport,
    RecommendationResponse,
    Severity,
    Sex,
    WearableMetrics,
)
from .service import RecommendationError, generate_recommendations

__all__ = [
    "BiologicalAgeResult",
    "BloodMarker",
    "DISCLAIMER",
    "Demographics",
    "HealthProfile",
    "Lifestyle",
    "MethylationResult",
    "RecommendationError",
    "RecommendationReport",
    "RecommendationResponse",
    "Severity",
    "Sex",
    "WearableMetrics",
    "evaluate_profile",
    "generate_recommendations",
]
