"""Claude call that turns a graded profile into a recommendations report."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic
from pydantic import ValidationError

from .prompt import DISCLAIMER, SYSTEM_PROMPT, build_user_message, has_actionable_findings
from .reference_ranges import FlaggedMarker, evaluate_profile
from .schemas import (
    HealthProfile,
    Priority,
    RecommendationReport,
    RecommendationResponse,
    Severity,
)

logger = logging.getLogger(__name__)

#: Generous because the request is streamed — a full panel with a long
#: report and adaptive thinking runs well past a 16k ceiling.
MAX_TOKENS = 64000
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "medium"

#: Service name under which the key is stored in the OS credential store.
KEYRING_SERVICE = "bio-age-recommendations"
KEYRING_USERNAME = "api-key"

_client: Optional[anthropic.Anthropic] = None


def get_model() -> str:
    """Read the model at call time, not import time — see `get_client`."""
    return os.environ.get("RECOMMENDATIONS_MODEL") or DEFAULT_MODEL


def get_effort() -> str:
    """Thinking effort: low | medium | high | xhigh | max. Same env pattern as get_model."""
    return os.environ.get("RECOMMENDATIONS_EFFORT") or DEFAULT_EFFORT


def _key_from_keyring() -> Optional[str]:
    """Read the key from the OS credential store, if one is available.

    Windows Credential Manager, macOS Keychain, or Secret Service on Linux —
    `keyring` picks the right backend. Encrypted at rest under the logged-in
    user's account rather than sitting in a readable file.

    Optional: if `keyring` is not installed, or there is no usable backend
    (common on headless Linux), fall through silently to the other sources.
    """
    try:
        import keyring
    except ImportError:
        return None

    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:  # noqa: BLE001 - any backend failure is a fall-through
        logger.debug("No usable keyring backend; falling back.", exc_info=True)
        return None


def get_client() -> anthropic.Anthropic:
    """Lazily construct a shared client.

    Environment is read here rather than at module import so that a `.env`
    loaded by the app entry point still applies. Reading env into module-level
    constants makes configuration depend on import order, which fails silently
    and confusingly.

    With nothing set, the zero-arg constructor resolves credentials the normal
    way — ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
    profile. Never hardcode a key here.

    `RECOMMENDATIONS_BASE_URL` points the SDK at a compatible gateway instead
    of api.anthropic.com. OpenRouter serves an Anthropic-format endpoint at
    https://openrouter.ai/api; pair it with an OpenRouter key in
    `RECOMMENDATIONS_API_KEY` and a namespaced model id.

    The key is resolved in this order, first hit wins:

    1. `RECOMMENDATIONS_API_KEY` — explicit override, and what a deployment
       platform's secret injection will set.
    2. The OS credential store (see `set_key.py`) — preferred for local dev,
       since nothing lands in a file you could commit or screen-share.
    3. Whatever the SDK resolves on its own (ANTHROPIC_API_KEY, an
       `ant auth login` profile, and so on).
    """
    global _client
    if _client is None:
        kwargs = {}
        base_url = os.environ.get("RECOMMENDATIONS_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

        api_key = os.environ.get("RECOMMENDATIONS_API_KEY") or _key_from_keyring()
        if api_key:
            kwargs["api_key"] = api_key

        _client = anthropic.Anthropic(**kwargs)
    return _client


class RecommendationError(RuntimeError):
    """Raised when a report could not be produced."""

    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class GenerationResult:
    response: RecommendationResponse
    flagged: List[FlaggedMarker]
    input_tokens: int
    output_tokens: int

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost at Claude Opus 5 list pricing ($5 / $25 per 1M tokens)."""
        return (self.input_tokens * 5.0 + self.output_tokens * 25.0) / 1_000_000


def generate_recommendations(profile: HealthProfile) -> GenerationResult:
    """Grade the profile, then have Claude write personalized recommendations."""
    flagged, unrecognized = evaluate_profile(profile)

    if not flagged:
        raise RecommendationError(
            "No recognized markers in the submitted profile — nothing to "
            "analyze. Check that blood marker names and units match the "
            "expected contract.",
            status_code=422,
        )

    if not has_actionable_findings(flagged):
        # Everything came back optimal. Cheaper and more honest to say so
        # directly than to ask a model to manufacture concerns.
        return GenerationResult(
            response=RecommendationResponse(
                report=RecommendationReport(
                    summary=(
                        "Every marker we were able to grade sits in its optimal "
                        "range. There is nothing here that needs correcting — "
                        "the useful move is to keep doing what you are doing and "
                        "re-test to confirm the trend holds."
                    ),
                    biological_age_drivers=[],
                    top_priorities=[
                        Priority(
                            rank=1,
                            focus="Maintain and re-test",
                            why_now=(
                                "A single optimal panel is a snapshot. Repeating "
                                "it in 6–12 months turns it into a trend."
                            ),
                            markers_addressed=[m.marker for m in flagged[:5]],
                        )
                    ],
                    recommendations=[],
                    caveats=[
                        "No marker was flagged, so no recommendations were "
                        "generated.",
                    ],
                ),
                flagged_marker_count=0,
                model="none (no findings)",
                disclaimer=DISCLAIMER,
            ),
            flagged=flagged,
            input_tokens=0,
            output_tokens=0,
        )

    user_message = build_user_message(profile, flagged, unrecognized)
    model = get_model()
    effort = get_effort()

    try:
        # Streamed rather than a plain parse() call: a full panel produces a
        # large report, and the max_tokens headroom needed for that is high
        # enough to risk an HTTP timeout on a non-streaming request.
        with get_client().messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": user_message}],
            output_format=RecommendationReport,
        ) as stream:
            response = stream.get_final_message()
    except ValidationError as exc:
        # Reached when the model's JSON is truncated or otherwise unparseable.
        # Truncation is the overwhelmingly likely cause, so say so plainly
        # rather than surfacing a pydantic traceback.
        raise RecommendationError(
            "The model's report could not be parsed — it was most likely cut "
            f"off by the {MAX_TOKENS}-token limit. Reduce the number of "
            "submitted markers or raise MAX_TOKENS."
        ) from exc
    except anthropic.AuthenticationError as exc:
        raise RecommendationError(
            "Claude API credentials are missing or invalid. Set ANTHROPIC_API_KEY.",
            status_code=500,
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RecommendationError(
            "Rate limited by the Claude API. Try again shortly.", status_code=429
        ) from exc
    except anthropic.APIStatusError as exc:
        logger.exception("Claude API returned %s", exc.status_code)
        raise RecommendationError(
            f"Claude API error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RecommendationError(
            "Could not reach the Claude API. Check network connectivity.",
            status_code=504,
        ) from exc

    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None)
        raise RecommendationError(
            "The model declined to produce a report for this input."
            + (f" ({detail})" if detail else ""),
            status_code=422,
        )

    if response.stop_reason == "max_tokens":
        raise RecommendationError(
            "The report was truncated before it could be parsed. Reduce the "
            "number of submitted markers or raise MAX_TOKENS.",
        )

    report = response.parsed_output
    if report is None:
        raise RecommendationError("Claude returned no parseable report.")

    report = _postprocess(report, flagged)

    return GenerationResult(
        response=RecommendationResponse(
            report=report,
            flagged_marker_count=sum(
                1 for m in flagged if m.severity != Severity.optimal
            ),
            model=model,
            disclaimer=DISCLAIMER,
        ),
        flagged=flagged,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def _postprocess(
    report: RecommendationReport, flagged: List[FlaggedMarker]
) -> RecommendationReport:
    """Enforce what we can check deterministically after generation.

    The model is told to flag every `high_concern` marker for a clinician, but
    that is an instruction, not a guarantee — so we re-assert it here, and sort
    by severity rather than trusting the emitted order.
    """
    from .schemas import SEVERITY_ORDER

    high_concern = {
        m.marker.lower() for m in flagged if m.severity == Severity.high_concern
    }
    for rec in report.recommendations:
        if rec.marker.lower() in high_concern:
            rec.clinician_flag = True

    report.recommendations.sort(key=lambda r: SEVERITY_ORDER[r.severity])
    report.top_priorities.sort(key=lambda p: p.rank)
    return report
