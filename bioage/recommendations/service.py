"""The LLM call that turns a graded profile into a recommendations report.

Two providers, selected by `RECOMMENDATIONS_PROVIDER`:

- `openai_compatible` (default) — any OpenAI-compatible endpoint: OpenAI
  itself, OpenRouter's native endpoint, Google AI Studio / Gemini's
  OpenAI-compat layer, and others. One code path covers all of them because
  they share a request/response shape; only `base_url` + `api_key` + `model`
  change. NOT live-verified against OpenRouter's native endpoint or Google AI
  Studio in this codebase -- see the note on `_call_openai_compatible` below.
- `anthropic` — the Anthropic SDK direct, or via a gateway that speaks its
  wire format (e.g. OpenRouter's Anthropic-compatible endpoint). Live-verified
  end to end, real API calls, both directly and through OpenRouter.

Both share everything upstream (grading, prompt construction) and downstream
(postprocessing, the disclaimer, error wrapping into `RecommendationError`) --
only the API call itself differs, in `_call_openai_compatible` /
`_call_anthropic`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

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

#: Generous because the request is streamed (Anthropic path) or otherwise
#: unbounded by a tight ceiling (OpenAI-compatible path) -- a full panel with
#: a long report and reasoning tokens runs well past a 16k ceiling.
MAX_TOKENS = 64000

#: Anthropic-only: a real, current, verified model id (see the claude-api
#: skill / Anthropic's model table this session was built against).
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"

#: No default for the OpenAI-compatible path, deliberately. "gpt-something"
#: or "gemini-something" would be a guess at a model id this codebase has no
#: verified source for, and a wrong guess fails in a more confusing way than
#: a clear startup error asking for RECOMMENDATIONS_MODEL. Pick the model
#: that matches whichever endpoint RECOMMENDATIONS_BASE_URL points at.
DEFAULT_EFFORT = "medium"

#: OpenAI-compatible `reasoning_effort` doesn't have Anthropic's `xhigh`/`max`
#: -- observed values across OpenAI's own reasoning models and Gemini's
#: OpenAI-compat layer are minimal/low/medium/high/none. Clamp the two
#: Anthropic-only levels down to `high` rather than send a value a given
#: provider might reject.
EFFORT_TO_REASONING_EFFORT = {
    "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high",
}

#: Service name under which the key is stored in the OS credential store.
KEYRING_SERVICE = "bio-age-recommendations"
KEYRING_USERNAME = "api-key"

_anthropic_client = None
_openai_client = None


def get_provider() -> str:
    """`openai_compatible` (default) or `anthropic`."""
    return os.environ.get("RECOMMENDATIONS_PROVIDER") or "openai_compatible"


def get_model() -> str:
    """Read the model at call time, not import time — see `get_anthropic_client`."""
    model = os.environ.get("RECOMMENDATIONS_MODEL")
    if model:
        return model
    if get_provider() == "anthropic":
        return ANTHROPIC_DEFAULT_MODEL
    raise RecommendationError(
        "RECOMMENDATIONS_MODEL is not set. The default provider "
        "(openai_compatible) has no safe default model to fall back to -- set "
        "RECOMMENDATIONS_MODEL to whatever model your RECOMMENDATIONS_BASE_URL "
        "endpoint expects (e.g. an OpenRouter or Gemini model id).",
        status_code=500,
    )


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


def _resolve_api_key() -> Optional[str]:
    """`RECOMMENDATIONS_API_KEY` env var, else the OS credential store.

    Shared by both providers. If neither is set, each client falls back to
    its SDK's own default resolution (OPENAI_API_KEY / ANTHROPIC_API_KEY, an
    `ant auth login` profile, etc.) — see each get_*_client for specifics.
    """
    return os.environ.get("RECOMMENDATIONS_API_KEY") or _key_from_keyring()


def get_openai_client():
    """Lazily construct a shared OpenAI-compatible client.

    `RECOMMENDATIONS_BASE_URL` points this at any OpenAI-compatible endpoint:
    unset for real OpenAI, `https://openrouter.ai/api/v1` for OpenRouter's
    native endpoint, or `https://generativelanguage.googleapis.com/v1beta/openai/`
    for Google AI Studio / Gemini. Model ids are provider-specific — set
    RECOMMENDATIONS_MODEL to match whichever endpoint this points at.

    Key resolution: RECOMMENDATIONS_API_KEY -> OS credential store -> the
    OpenAI SDK's own default (OPENAI_API_KEY). Never hardcode a key here.
    """
    global _openai_client
    if _openai_client is None:
        import openai

        kwargs = {}
        base_url = os.environ.get("RECOMMENDATIONS_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = _resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        _openai_client = openai.OpenAI(**kwargs)
    return _openai_client


def get_anthropic_client():
    """Lazily construct a shared Anthropic client.

    With nothing set, the zero-arg constructor resolves credentials the normal
    way — ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
    profile. Never hardcode a key here.

    `RECOMMENDATIONS_BASE_URL` points the SDK at a compatible gateway instead
    of api.anthropic.com — e.g. OpenRouter's Anthropic-format endpoint at
    https://openrouter.ai/api, paired with an OpenRouter key and a namespaced
    model id (verified: this combination works end to end).

    Key resolution: RECOMMENDATIONS_API_KEY -> OS credential store -> the
    Anthropic SDK's own default.
    """
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        kwargs = {}
        base_url = os.environ.get("RECOMMENDATIONS_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = _resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        _anthropic_client = anthropic.Anthropic(**kwargs)
    return _anthropic_client


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
    provider: str = "anthropic"

    @property
    def estimated_cost_usd(self) -> Optional[float]:
        """Rough cost at Claude Sonnet 5 list pricing ($3 / $15 per 1M tokens).

        Anthropic only. For `openai_compatible`, the model behind
        RECOMMENDATIONS_BASE_URL could be anything from any provider at any
        price point -- a dollar figure here would be confidently wrong rather
        than approximately right, so this returns None instead of guessing.
        """
        if self.provider != "anthropic":
            return None
        return (self.input_tokens * 3.0 + self.output_tokens * 15.0) / 1_000_000


def generate_recommendations(profile: HealthProfile) -> GenerationResult:
    """Grade the profile, then have the model write personalized recommendations."""
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
    provider = get_provider()
    model = get_model()
    effort = get_effort()

    if provider == "anthropic":
        report, input_tokens, output_tokens = _call_anthropic(user_message, model, effort)
    elif provider == "openai_compatible":
        report, input_tokens, output_tokens = _call_openai_compatible(
            user_message, model, effort
        )
    else:
        raise RecommendationError(
            f"Unknown RECOMMENDATIONS_PROVIDER {provider!r}. "
            "Use 'openai_compatible' or 'anthropic'.",
            status_code=500,
        )

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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider=provider,
    )


def _call_anthropic(user_message: str, model: str, effort: str):
    """Live-verified path: real API calls, both direct and via OpenRouter's
    Anthropic-compatible endpoint. See README / INTEGRATION_PLAN.md."""
    import anthropic

    try:
        # Streamed rather than a plain parse() call: a full panel produces a
        # large report, and the max_tokens headroom needed for that is high
        # enough to risk an HTTP timeout on a non-streaming request.
        with get_anthropic_client().messages.stream(
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
            "API credentials are missing or invalid. Set RECOMMENDATIONS_API_KEY "
            "or ANTHROPIC_API_KEY.",
            status_code=500,
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RecommendationError(
            "Rate limited by the API. Try again shortly.", status_code=429
        ) from exc
    except anthropic.APIStatusError as exc:
        logger.exception("Anthropic-compatible API returned %s", exc.status_code)
        raise RecommendationError(
            f"API error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RecommendationError(
            "Could not reach the API. Check network connectivity and "
            "RECOMMENDATIONS_BASE_URL.",
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
        raise RecommendationError("The model returned no parseable report.")

    return report, response.usage.input_tokens, response.usage.output_tokens


def _call_openai_compatible(user_message: str, model: str, effort: str):
    """Default path: any OpenAI-compatible endpoint (OpenAI, OpenRouter's
    native endpoint, Google AI Studio / Gemini).

    Live-verified against two of the three: OpenRouter's native endpoint
    (model anthropic/claude-sonnet-5, confirming this path works even when
    the underlying model is a Claude model reached the OpenAI-compatible way
    rather than through the `anthropic` branch) and Google AI Studio (model
    gemini-3.6-flash -- note gemini-2.5-flash, an earlier obvious choice, was
    retired for new users as of this testing and 404s; Google's own error
    named the replacement). Real OpenAI itself has not been tested here, but
    shares the same SDK and endpoint shape as both of the above.
    """
    import openai

    try:
        completion = get_openai_client().beta.chat.completions.parse(
            model=model,
            max_completion_tokens=MAX_TOKENS,
            reasoning_effort=EFFORT_TO_REASONING_EFFORT.get(effort, "medium"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=RecommendationReport,
        )
    except ValidationError as exc:
        raise RecommendationError(
            "The model's report could not be parsed — it was most likely cut "
            f"off by the {MAX_TOKENS}-token limit. Reduce the number of "
            "submitted markers or raise MAX_TOKENS."
        ) from exc
    except openai.AuthenticationError as exc:
        raise RecommendationError(
            "API credentials are missing or invalid. Set RECOMMENDATIONS_API_KEY "
            "or OPENAI_API_KEY.",
            status_code=500,
        ) from exc
    except openai.RateLimitError as exc:
        raise RecommendationError(
            "Rate limited by the API. Try again shortly.", status_code=429
        ) from exc
    except openai.APIStatusError as exc:
        logger.exception("OpenAI-compatible API returned %s", exc.status_code)
        raise RecommendationError(
            f"API error ({exc.status_code}): {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        raise RecommendationError(
            "Could not reach the API. Check network connectivity and "
            "RECOMMENDATIONS_BASE_URL.",
            status_code=504,
        ) from exc

    choice = completion.choices[0]
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        raise RecommendationError(
            f"The model declined to produce a report for this input. ({refusal})",
            status_code=422,
        )

    if choice.finish_reason == "length":
        raise RecommendationError(
            "The report was truncated before it could be parsed. Reduce the "
            "number of submitted markers or raise MAX_TOKENS.",
        )

    report = choice.message.parsed
    if report is None:
        raise RecommendationError("The model returned no parseable report.")

    usage = completion.usage
    return report, usage.prompt_tokens, usage.completion_tokens


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
