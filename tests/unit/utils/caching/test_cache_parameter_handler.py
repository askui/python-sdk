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


class _RaisingVlmProvider:
    """Provider that fails if the LLM is called - used to assert it is skipped."""

    model_id = "should-not-be-called"

    def create_message(self, **_: Any) -> _FakeResponse:
        msg = "LLM should not be called when there are no candidates"
        raise AssertionError(msg)


class TestCandidateCollection:
    def test_excludes_coordinates_actions_keys_and_tool_names(self) -> None:
        trajectory = [
            ToolUseBlockParam(
                id="0",
                name="computer_tool",
                input={"action": "left_click", "coordinate": [100, 200]},
            ),
            ToolUseBlockParam(
                id="1", name="keyboard_tool", input={"action": "key", "key": "Return"}
            ),
            ToolUseBlockParam(
                id="2",
                name="computer_tool",
                input={"action": "type", "text": "hello world"},
            ),
        ]
        candidates = CacheParameterHandler._collect_candidate_values(trajectory)
        # Only the typed free text is a candidate.
        assert candidates == ["hello world"]

    def test_dedupes_and_skips_short_and_templated_values(self) -> None:
        trajectory = [
            ToolUseBlockParam(id="0", name="t", input={"text": "repeat"}),
            ToolUseBlockParam(id="1", name="t", input={"text": "repeat"}),
            ToolUseBlockParam(id="2", name="t", input={"note": "x"}),  # too short
            ToolUseBlockParam(id="3", name="t", input={"note": "{{already}}"}),
        ]
        candidates = CacheParameterHandler._collect_candidate_values(trajectory)
        assert candidates == ["repeat"]

    def test_no_candidates_skips_llm_call(self) -> None:
        """A trajectory of only clicks must not trigger an LLM call."""
        trajectory = [
            ToolUseBlockParam(
                id="0",
                name="computer_tool",
                input={"action": "left_click", "coordinate": [10, 20]},
            )
        ]
        goal, out_traj, params = CacheParameterHandler.identify_and_parameterize(
            trajectory=trajectory,
            goal="click the button",
            identification_strategy="llm",
            vlm_provider=_RaisingVlmProvider(),  # type: ignore[arg-type]
        )
        assert params == {}
        assert out_traj == trajectory
        assert goal == "click the button"


class TestHallucinatedValueRejection:
    def test_value_not_in_candidates_is_dropped(self) -> None:
        # The model returns a value that was never a candidate -> reject it.
        response = (
            '{"parameters": [{"name": "made_up", "value": "not-a-candidate", '
            '"description": "hallucinated"}]}'
        )
        _, trajectory, params = _parameterize(response, value="admin")
        assert params == {}
        assert trajectory[0].input == {"text": "admin"}
