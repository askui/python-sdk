"""Model-aware defaults for thinking configuration.

Lives in ``models.shared`` because it is used by the provider-agnostic agents
(computer, web, ...), which may run on any provider. The known model families are
currently Anthropic-only; branches for other providers (Gemini, OpenAI,
open-source, ...) can be added here as they gain thinking controls.

Anthropic changed how thinking is configured across model generations:

- **Older models** (e.g. Claude Sonnet 4, Sonnet 4.5, Opus 4.1/4.5, Haiku 4.5)
  use a fixed integer token budget: ``{"type": "enabled", "budget_tokens": N}``.
- **Newer models** (e.g. Claude Sonnet 4.6, Sonnet 5, Opus 4.6/4.7/4.8, Fable 5)
  use *adaptive* thinking (``{"type": "adaptive"}``) and control depth with the
  string ``effort`` instead. Sending ``budget_tokens`` to these models is
  rejected, and ``effort`` is a separate parameter sent via
  ``output_config.effort`` (not part of ``thinking``).

The budget-token generation is a closed set, so classification is inverted:
every Claude model *not* in the frozen legacy list is treated as adaptive,
which makes unknown future models work by default. Model IDs are matched on
their ``claude-...`` core, so gateway-prefixed identifiers (Bedrock
``anthropic.claude-opus-4-8`` or ``us.anthropic.claude-...-v1:0``, LiteLLM
``anthropic/claude-...``, Vertex ``claude-...@20260401``) resolve like the
bare model ID.

`make_thinking_settings()` returns the `MessageSettings` keyword arguments that
enable thinking for a given ``model_id``, so agents can turn thinking on by
default without knowing which generation they run on. Callers can still override
`thinking` (and `provider_options`) explicitly.
"""

from typing import Any

from askui.models.shared.agent_message_param import EffortLevel

_DEFAULT_BUDGET_TOKENS = 2048

# Model-ID prefixes (after normalization) of the Anthropic model families that
# take the fixed integer `budget_tokens`. This set is FROZEN: budget thinking
# was replaced by adaptive thinking with the 4.6 generation, so no future model
# will ever be added here.
_LEGACY_BUDGET_THINKING_MODEL_PREFIXES = (
    "claude-2",
    "claude-instant",
    "claude-3-",
    "claude-haiku-4-5",
    "claude-sonnet-4-0",
    "claude-sonnet-4-1",
    "claude-sonnet-4-2",  # dated snapshots, e.g. "claude-sonnet-4-20250514"
    "claude-sonnet-4-5",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2",  # dated snapshots, e.g. "claude-opus-4-20250514"
    "claude-opus-4-5",
)

# The one adaptive-thinking generation that still accepts sampling parameters
# (temperature/top_p/top_k). From Opus 4.7 / Sonnet 5 / Fable 5 onward the API
# rejects them with a 400.
_SAMPLING_CAPABLE_ADAPTIVE_MODEL_PREFIXES = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)

# Models where thinking is always on: an explicit {"type": "disabled"} is
# rejected with a 400, so the thinking field must be omitted entirely.
_ALWAYS_ON_THINKING_MODEL_PREFIXES = (
    "claude-fable-5",
    "claude-mythos",
)


def _normalize(model_id: str) -> str | None:
    """Extract the ``claude-...`` core of a model ID.

    Gateway wrappers then match like bare IDs (e.g.
    ``"us.anthropic.claude-opus-4-8-v1:0"`` and ``"anthropic/claude-opus-4-8"``
    both normalize to ``"claude-opus-4-8..."``).

    Args:
        model_id (str): The (possibly gateway-prefixed) model identifier.

    Returns:
        str | None: The model ID from its ``claude-`` core onward, or ``None``
            if the ID does not reference a Claude model.
    """
    index = model_id.find("claude-")
    return None if index < 0 else model_id[index:]


def uses_adaptive_thinking(model_id: str) -> bool:
    """Whether ``model_id`` uses adaptive thinking instead of a token budget.

    True for every Claude model outside the frozen legacy budget families (so
    unknown future models default to adaptive); False for non-Claude model IDs.

    Args:
        model_id (str): The model identifier (bare or gateway-prefixed).

    Returns:
        bool: ``True`` if the model expects ``{"type": "adaptive"}`` and the
            `effort` setting, ``False`` if it expects a fixed ``budget_tokens``.
    """
    normalized = _normalize(model_id)
    return normalized is not None and not normalized.startswith(
        _LEGACY_BUDGET_THINKING_MODEL_PREFIXES
    )


def accepts_sampling_params(model_id: str) -> bool:
    """Whether the model accepts sampling parameters such as ``temperature``.

    False for adaptive-thinking Claude models newer than the 4.6 generation
    (Opus 4.7/4.8, Sonnet 5, Fable 5, and future models), which reject them
    with a 400. True for older Claude models and non-Claude model IDs (other
    providers manage their own sampling parameters).

    Args:
        model_id (str): The model identifier (bare or gateway-prefixed).

    Returns:
        bool: ``True`` if sampling parameters may be sent to the model.
    """
    normalized = _normalize(model_id)
    return normalized is None or normalized.startswith(
        _LEGACY_BUDGET_THINKING_MODEL_PREFIXES
        + _SAMPLING_CAPABLE_ADAPTIVE_MODEL_PREFIXES
    )


def supports_disabled_thinking(model_id: str) -> bool:
    """Whether the model accepts an explicit ``{"type": "disabled"}`` thinking config.

    False for always-on-thinking models (Fable 5, Mythos 5), which reject it
    with a 400 — omit the thinking field there.

    Args:
        model_id (str): The model identifier (bare or gateway-prefixed).

    Returns:
        bool: ``True`` if ``{"type": "disabled"}`` may be sent to the model.
    """
    normalized = _normalize(model_id)
    return normalized is None or not normalized.startswith(
        _ALWAYS_ON_THINKING_MODEL_PREFIXES
    )


def make_thinking_settings(
    model_id: str,
    effort: EffortLevel | None = None,
) -> dict[str, Any]:
    """Return `MessageSettings` keyword arguments enabling thinking for a model.

    Splat the result into `MessageSettings` alongside any other fields, e.g.::

        MessageSettings(
            system=create_computer_agent_prompt(),
            **make_thinking_settings(self._vlm_provider.model_id),
        )

    Models that support adaptive thinking get
    ``thinking={"type": "adaptive", "display": "summarized"}`` (with ``effort``
    sent via ``provider_options["output_config"]`` when given); older models
    get a fixed token budget of
    ``thinking={"type": "enabled", "budget_tokens": 2048}`` and ignore
    ``effort``. ``display`` is set explicitly because the newest adaptive
    models (Sonnet 5 generation onward) default it to ``"omitted"``, which
    returns thinking blocks whose text is EMPTY while the full thinking
    tokens are still billed — reasoning silently disappears from reports and
    logs. ``"summarized"`` restores the visible text at no extra cost (billing
    is identical for both display modes).

    Args:
        model_id (str): The model identifier (bare or gateway-prefixed).
        effort (EffortLevel | None, optional): How much the model should think
            and act (``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` or
            ``"max"``). Only applied for models that support adaptive thinking.
            Default: None (the model uses its own default).

    Returns:
        dict[str, Any]: `MessageSettings` keyword arguments (``thinking`` and,
            when applicable, ``provider_options``).
    """
    if uses_adaptive_thinking(model_id):
        settings: dict[str, Any] = {
            "thinking": {"type": "adaptive", "display": "summarized"}
        }
        if effort is not None:
            settings["provider_options"] = {"output_config": {"effort": effort}}
        return settings
    return {"thinking": {"type": "enabled", "budget_tokens": _DEFAULT_BUDGET_TOKENS}}


def make_non_thinking_settings(model_id: str) -> dict[str, Any]:
    """Return `MessageSettings` keyword arguments for thinking-off, deterministic runs.

    Used by device agents (Android) that historically pinned
    ``thinking={"type": "disabled"}`` and ``temperature=0.0``. Each field is
    included only where the model still accepts it: models from the Opus 4.7
    generation onward reject sampling parameters, and always-on-thinking models
    (Fable 5) reject an explicit ``"disabled"``.

    Args:
        model_id (str): The model identifier (bare or gateway-prefixed).

    Returns:
        dict[str, Any]: `MessageSettings` keyword arguments (``thinking``
            and/or ``temperature``, possibly empty).
    """
    settings: dict[str, Any] = {}
    if supports_disabled_thinking(model_id):
        settings["thinking"] = {"type": "disabled"}
    if accepts_sampling_params(model_id):
        settings["temperature"] = 0.0
    return settings
