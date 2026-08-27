"""Tests for LLM parameter identification robustness in the recording path."""

from typing import Any

from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    ToolUseBlockParam,
)
from askui.utils.caching.cache_parameter_handler import CacheParameterHandler


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [TextBlockParam(type="text", text=text)]


class _FakeVlmProvider:
    """Minimal VlmProvider stand-in returning a canned JSON response."""

    model_id = "fake-model"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def create_message(self, **_: Any) -> _FakeResponse:
        return _FakeResponse(self._response_text)


def _trajectory(value: str) -> list[ToolUseBlockParam]:
    return [ToolUseBlockParam(id="0", name="type_tool", input={"text": value})]


def _parameterize(
    response_text: str, value: str = "admin"
) -> tuple[str | None, list[ToolUseBlockParam], dict[str, str]]:
    provider = _FakeVlmProvider(response_text)
    return CacheParameterHandler.identify_and_parameterize(
        trajectory=_trajectory(value),
        goal=f"log in as {value}",
        identification_strategy="llm",
        vlm_provider=provider,  # type: ignore[arg-type]
    )


class TestParameterIdentificationRobustness:
    def test_empty_value_parameter_is_dropped_and_trajectory_intact(self) -> None:
        """An empty parameter value must not shred every string in the trajectory."""
        response = (
            '{"parameters": [{"name": "username", "value": "", '
            '"description": "the user"}]}'
        )
        goal, trajectory, params = _parameterize(response, value="Submit")
        assert params == {}
        # The input must be untouched (no '{{...}}' corruption between chars).
        assert trajectory[0].input == {"text": "Submit"}
        assert goal == "log in as Submit"

    def test_invalid_parameter_name_is_dropped(self) -> None:
        """A name that is not a valid {{identifier}} would break validation."""
        response = (
            '{"parameters": [{"name": "user name", "value": "admin", '
            '"description": "the user"}]}'
        )
        _, trajectory, params = _parameterize(response)
        assert params == {}
        assert trajectory[0].input == {"text": "admin"}

    def test_valid_parameter_is_applied(self) -> None:
        response = (
            '{"parameters": [{"name": "username", "value": "admin", '
            '"description": "the user"}]}'
        )
        goal, trajectory, params = _parameterize(response)
        assert params == {"username": "the user"}
        assert trajectory[0].input == {"text": "{{username}}"}
        assert goal == "log in as {{username}}"

    def test_malformed_response_falls_back_to_no_parameters(self) -> None:
        _, trajectory, params = _parameterize("not json at all")
        assert params == {}
        assert trajectory[0].input == {"text": "admin"}


class TestValidateParameters:
    def test_reports_missing_parameters(self) -> None:
        trajectory = [
            ToolUseBlockParam(id="0", name="type_tool", input={"text": "{{token}}"})
        ]
        is_valid, missing = CacheParameterHandler.validate_parameters(trajectory, {})
        assert is_valid is False
        assert missing == ["token"]

    def test_all_present(self) -> None:
        trajectory = [
            ToolUseBlockParam(id="0", name="type_tool", input={"text": "{{token}}"})
        ]
        is_valid, missing = CacheParameterHandler.validate_parameters(
            trajectory, {"token": "abc"}
        )
        assert is_valid is True
        assert missing == []


def test_substitute_parameters_replaces_placeholder() -> None:
    block = ToolUseBlockParam(id="0", name="type_tool", input={"text": "{{token}}"})
    result = CacheParameterHandler.substitute_parameters(block, {"token": "secret"})
    assert result.input == {"text": "secret"}


def test_message_param_import_is_available() -> None:
    # Guard that MessageParam remains importable for this module's provider stub.
    assert MessageParam is not None
