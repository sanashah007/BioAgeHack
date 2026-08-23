"""The BioAgeHack-specific endpoint: raw markers in, score + recommendations out.

Kept separate from `router.py`, which stays provider-agnostic (it accepts an
already-built `HealthProfile` and never imports `bioage`). This module is the
one place that imports `bioage.scorer` at the HTTP layer -- mount it only where
you actually want the scoring dependency (numpy/pandas/lifelines/scikit-learn)
pulled in.

Mount with:

    from bioage.recommendations.analyze_router import router as analyze_router
    app.include_router(analyze_router)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .bioage_bridge import score_markers
from .schemas import Lifestyle, RecommendationResponse
from .service import RecommendationError, generate_recommendations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    age: float
    sex: str = Field(..., description="male | female")
    markers: Dict[str, Dict[str, float]] = Field(
        ..., description='{"blood": {"crp": 0.34, ...}, "wearable": {...}}, '
                         "keyed by bioage.config feature names."
    )
    lifestyle: Optional[Lifestyle] = None


class AnalyzeResponse(BaseModel):
    score: dict = Field(..., description="Normalized bioage.scorer report.")
    recommendations: RecommendationResponse


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Score a raw panel and generate recommendations in one round trip.

    `def`, not `async def` -- both the scorer's curve inversion and the LLM
    call are synchronous and the latter is slow (reasoning/thinking can take
    20-60s), so FastAPI runs this in its threadpool.
    """
    try:
        report, profile = score_markers(
            age=request.age,
            sex=request.sex,
            markers=request.markers,
            lifestyle=request.lifestyle,
        )
    except ValueError as exc:
        # score_markers -> report_to_biological_age raises ValueError when no
        # modality produced a usable gap (e.g. fewer than the min-feature floor).
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = generate_recommendations(profile)
    except RecommendationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    cost = (f"~${result.estimated_cost_usd:.3f}" if result.estimated_cost_usd is not None
            else "unknown (non-Anthropic provider)")
    logger.info(
        "analyze: driver=%s flagged=%d %d in / %d out tokens (%s)",
        report.get("driver"),
        result.response.flagged_marker_count,
        result.input_tokens,
        result.output_tokens,
        cost,
    )
    return AnalyzeResponse(score=report, recommendations=result.response)
