"""Unit tests for model-aware thinking defaults."""

import pytest

from askui.models.shared.thinking import (
    make_thinking_settings,
    uses_adaptive_thinking,
)

_ADAPTIVE_MODELS = [
    "claude-sonnet-5",
    "claude-sonnet-5-20260601",
    "claude-sonnet-4-6",
    "claude-sonnet-4-6-20260401",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
]

_BUDGET_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",
    "gpt-5.4",
    "some-unknown-model",
]


@pytest.mark.parametrize("model_id", _ADAPTIVE_MODELS)
def test_adaptive_models_use_adaptive_thinking(model_id: str) -> None:
    assert uses_adaptive_thinking(model_id) is True
    assert make_thinking_settings(model_id) == {"thinking": {"type": "adaptive"}}


@pytest.mark.parametrize("model_id", _BUDGET_MODELS)
def test_other_models_use_budget_tokens(model_id: str) -> None:
    assert uses_adaptive_thinking(model_id) is False
    assert make_thinking_settings(model_id) == {
        "thinking": {"type": "enabled", "budget_tokens": 2048}
    }


def test_effort_is_added_via_provider_options_for_adaptive_models() -> None:
    assert make_thinking_settings("claude-sonnet-5", effort="high") == {
        "thinking": {"type": "adaptive"},
        "provider_options": {"output_config": {"effort": "high"}},
    }


def test_effort_is_ignored_for_budget_models() -> None:
    # Older models do not support `effort`; it must not leak into the settings.
    assert make_thinking_settings("claude-sonnet-4-5-20250929", effort="high") == {
        "thinking": {"type": "enabled", "budget_tokens": 2048}
    }


def test_sonnet_5_is_not_confused_with_sonnet_4_5() -> None:
    # "claude-sonnet-5" must not match the older "claude-sonnet-4-5" prefix.
    assert uses_adaptive_thinking("claude-sonnet-5") is True
    assert uses_adaptive_thinking("claude-sonnet-4-5") is False
