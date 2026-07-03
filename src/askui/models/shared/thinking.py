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

`make_thinking_settings()` returns the `MessageSettings` keyword arguments that
enable thinking for a given ``model_id``, so agents can turn thinking on by
default without knowing which generation they run on. Callers can still override
`thinking` (and `provider_options`) explicitly.
"""

from typing import Any

from askui.models.shared.agent_message_param import EffortLevel

_DEFAULT_BUDGET_TOKENS = 2048

# Model-ID prefixes for Anthropic models that use adaptive thinking. These are
# the models where the integer `budget_tokens` is removed or deprecated in favour
# of adaptive thinking (`{"type": "adaptive"}`) plus the `effort` setting.
# `str.startswith` matches dated snapshots too (e.g. "claude-sonnet-4-6-20260401").
# Note: "claude-sonnet-5" does not match the older "claude-sonnet-4-5".
_ADAPTIVE_THINKING_MODEL_PREFIXES = (
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
)


def uses_adaptive_thinking(model_id: str) -> bool:
    """Whether ``model_id`` uses adaptive thinking instead of a token budget.

    Args:
        model_id (str): The Anthropic model identifier.

    Returns:
        bool: ``True`` if the model expects ``{"type": "adaptive"}`` and the
            `effort` setting, ``False`` if it expects a fixed ``budget_tokens``.
    """
    return model_id.startswith(_ADAPTIVE_THINKING_MODEL_PREFIXES)


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

    Models that support adaptive thinking get ``thinking={"type": "adaptive"}``
    (with ``effort`` sent via ``provider_options["output_config"]`` when given);
    older models get a fixed token budget of
    ``thinking={"type": "enabled", "budget_tokens": 2048}`` and ignore ``effort``.

    Args:
        model_id (str): The Anthropic model identifier.
        effort (EffortLevel | None, optional): How much the model should think and
            act (``"low"``, ``"medium"``, ``"high"`` or ``"max"``). Only applied
            for models that support adaptive thinking. Default: None (the model
            uses its own default).

    Returns:
        dict[str, Any]: `MessageSettings` keyword arguments (``thinking`` and,
            when applicable, ``provider_options``).
    """
    if uses_adaptive_thinking(model_id):
        settings: dict[str, Any] = {"thinking": {"type": "adaptive"}}
        if effort is not None:
            settings["provider_options"] = {"output_config": {"effort": effort}}
        return settings
    return {"thinking": {"type": "enabled", "budget_tokens": _DEFAULT_BUDGET_TOKENS}}
