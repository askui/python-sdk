"""Unit tests for model-aware thinking defaults."""

import pytest

from askui.models.shared.thinking import (
    accepts_sampling_params,
    make_non_thinking_settings,
    make_thinking_settings,
    supports_disabled_thinking,
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
    "claude-haiku-5",  # unknown future model -> adaptive by default
    # Gateway-prefixed IDs (Bedrock, LiteLLM, Vertex) classify like bare IDs.
    "anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-4-8-v1:0",
    "eu.anthropic.claude-sonnet-5-v1:0",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-fable-5",
    "bedrock/us.anthropic.claude-opus-4-7-v1:0",
    "vertex_ai/claude-sonnet-4-6",
]

_BUDGET_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic/claude-haiku-4-5",
    "claude-opus-4-5@20251101",  # Vertex version separator
    "gpt-5.4",
    "some-unknown-model",
]


@pytest.mark.parametrize("model_id", _ADAPTIVE_MODELS)
def test_adaptive_models_use_adaptive_thinking(model_id: str) -> None:
    assert uses_adaptive_thinking(model_id) is True
    assert make_thinking_settings(model_id) == {
        "thinking": {"type": "adaptive", "display": "summarized"}
    }


@pytest.mark.parametrize("model_id", _BUDGET_MODELS)
def test_other_models_use_budget_tokens(model_id: str) -> None:
    assert uses_adaptive_thinking(model_id) is False
    assert make_thinking_settings(model_id) == {
        "thinking": {"type": "enabled", "budget_tokens": 2048}
    }


def test_effort_is_added_via_provider_options_for_adaptive_models() -> None:
    assert make_thinking_settings("claude-sonnet-5", effort="high") == {
        "thinking": {"type": "adaptive", "display": "summarized"},
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


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-sonnet-4-5-20250929", True),  # legacy: sampling params fine
        ("claude-haiku-4-5", True),
        ("claude-sonnet-4-6", True),  # 4.6 generation still accepts them
        ("claude-opus-4-6", True),
        ("claude-opus-4-7", False),  # removed from Opus 4.7 onward
        ("claude-opus-4-8", False),
        ("claude-sonnet-5", False),
        ("claude-fable-5", False),
        ("anthropic.claude-opus-4-8", False),  # gateway IDs classified too
        ("anthropic/claude-sonnet-5", False),
        ("gpt-5.4", True),  # non-Claude: other providers manage their own
    ],
)
def test_accepts_sampling_params(model_id: str, expected: bool) -> None:
    assert accepts_sampling_params(model_id) is expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-8", True),
        ("claude-sonnet-5", True),
        ("claude-fable-5", False),  # always-on thinking rejects "disabled"
        ("claude-mythos-5", False),
        ("anthropic/claude-fable-5", False),
        ("gpt-5.4", True),
    ],
)
def test_supports_disabled_thinking(model_id: str, expected: bool) -> None:
    assert supports_disabled_thinking(model_id) is expected


def test_non_thinking_settings_keep_parity_on_older_models() -> None:
    assert make_non_thinking_settings("claude-sonnet-4-6") == {
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
    }


def test_non_thinking_settings_drop_temperature_from_opus_4_7_on() -> None:
    assert make_non_thinking_settings("claude-opus-4-8") == {
        "thinking": {"type": "disabled"},
    }
    assert make_non_thinking_settings("anthropic/claude-sonnet-5") == {
        "thinking": {"type": "disabled"},
    }


def test_non_thinking_settings_omit_thinking_on_always_on_models() -> None:
    assert make_non_thinking_settings("claude-fable-5") == {}


def test_effort_supports_xhigh() -> None:
    assert make_thinking_settings("claude-opus-4-8", effort="xhigh") == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "provider_options": {"output_config": {"effort": "xhigh"}},
    }
