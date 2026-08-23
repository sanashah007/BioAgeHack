"""FastAPI router. Mount with:

    from recommendations.router import router as recommendations_router
    app.include_router(recommendations_router)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .reference_ranges import evaluate_profile
from .schemas import HealthProfile, RecommendationResponse
from .service import RecommendationError, generate_recommendations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
def create_recommendations(profile: HealthProfile) -> RecommendationResponse:
    """Generate personalized recommendations from a graded health profile.

    Defined with `def`, not `async def`: the Anthropic call is synchronous and
    slow (adaptive thinking at high effort can take 30s+), so FastAPI runs this
    in its threadpool instead of blocking the event loop.
    """
    try:
        result = generate_recommendations(profile)
    except RecommendationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    cost = (f"~${result.estimated_cost_usd:.3f}" if result.estimated_cost_usd is not None
            else "unknown (non-Anthropic provider)")
    logger.info(
        "recommendations generated: %d flagged, %d in / %d out tokens (%s)",
        result.response.flagged_marker_count,
        result.input_tokens,
        result.output_tokens,
        cost,
    )
    return result.response


@router.post("/preview")
def preview_flagged_markers(profile: HealthProfile) -> dict:
    """Grade a profile without calling the LLM.

    Useful while wiring up the upload/analysis half: it confirms marker names
    and units are being parsed into something the reference table recognizes,
    and it costs nothing.
    """
    flagged, unrecognized = evaluate_profile(profile)
    return {
        "graded": [m.as_prompt_dict() for m in flagged],
        "ungraded": unrecognized,
    }
