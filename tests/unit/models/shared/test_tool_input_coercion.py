"""Tests for coercing model-provided tool inputs against the tool's input schema.

Models occasionally emit tool arguments with the wrong JSON type, most commonly
integers as strings (e.g. `{"x": "726", "y": "122"}`). The tool-calling loop must
coerce such values to the declared schema type before invoking the tool so the
tool does not fail with `'<' not supported between instances of 'str' and 'int'`.
"""

from typing import Any

import pytest

from askui.models.shared.agent_message_param import (
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tool_input_coercion import coerce_tool_input
from askui.models.shared.tools import Tool, ToolCollection

_TAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "repeat": {"type": "integer", "default": 1},
        "repeat_delay_in_ms": {"type": "integer", "default": 50},
    },
    "required": ["x", "y", "repeat", "repeat_delay_in_ms"],
}


class TestCoerceToolInput:
    def test_string_integers_are_coerced(self) -> None:
        tool_input = {"x": "726", "y": "122", "repeat": "1", "repeat_delay_in_ms": "50"}

        assert coerce_tool_input(tool_input, _TAP_SCHEMA) == {
            "x": 726,
            "y": 122,
            "repeat": 1,
            "repeat_delay_in_ms": 50,
        }

    def test_input_is_not_mutated(self) -> None:
        tool_input = {"x": "726", "y": "122"}

        coerce_tool_input(tool_input, _TAP_SCHEMA)

        assert tool_input == {"x": "726", "y": "122"}

    def test_matching_types_are_left_untouched(self) -> None:
        tool_input = {"x": 726, "y": 122, "repeat": 1, "repeat_delay_in_ms": 50}

        assert coerce_tool_input(tool_input, _TAP_SCHEMA) == tool_input

    def test_integral_floats_are_coerced_to_integer(self) -> None:
        assert coerce_tool_input({"x": 726.0, "y": "122.0"}, _TAP_SCHEMA) == {
            "x": 726,
            "y": 122,
        }

    def test_non_integral_values_are_left_for_the_tool_to_reject(self) -> None:
        tool_input = {"x": "abc", "y": 1.5}

        assert coerce_tool_input(tool_input, _TAP_SCHEMA) == tool_input

    def test_unknown_keys_are_left_untouched(self) -> None:
        tool_input = {"x": "1", "unknown": "2"}

        assert coerce_tool_input(tool_input, _TAP_SCHEMA) == {"x": 1, "unknown": "2"}

    def test_none_is_left_untouched(self) -> None:
        assert coerce_tool_input({"x": None}, _TAP_SCHEMA) == {"x": None}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("False", False),
            ("1", True),
            (0, False),
            ("yes", "yes"),
            ("maybe", "maybe"),
            (2, 2),
        ],
    )
    def test_boolean_coercion(self, value: Any, expected: Any) -> None:
        schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}

        assert coerce_tool_input({"flag": value}, schema) == {"flag": expected}

    def test_number_coercion(self) -> None:
        schema = {"type": "object", "properties": {"ratio": {"type": "number"}}}

        assert coerce_tool_input({"ratio": "0.25"}, schema) == {"ratio": 0.25}
        assert coerce_tool_input({"ratio": 2}, schema) == {"ratio": 2}

    def test_numbers_are_coerced_to_string(self) -> None:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        assert coerce_tool_input({"text": 42}, schema) == {"text": "42"}
        assert coerce_tool_input({"text": 1.5}, schema) == {"text": "1.5"}

    def test_booleans_are_not_coerced_to_string(self) -> None:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        assert coerce_tool_input({"text": True}, schema) == {"text": True}

    def test_union_types_are_respected(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": ["integer", "null"]},
                "b": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
        }

        assert coerce_tool_input({"a": "3", "b": "3"}, schema) == {"a": 3, "b": "3"}

    def test_nested_objects_and_arrays_are_coerced(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
                "ids": {"type": "array", "items": {"type": "integer"}},
            },
        }

        assert coerce_tool_input(
            {"point": {"x": "1", "y": "2"}, "ids": ["3", 4, "x"]}, schema
        ) == {"point": {"x": 1, "y": 2}, "ids": [3, 4, "x"]}

    def test_refs_are_resolved(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Coordinate"}},
            "$defs": {"Coordinate": {"type": "integer"}},
        }

        assert coerce_tool_input({"x": "7"}, schema) == {"x": 7}

    def test_schema_without_properties_is_a_noop(self) -> None:
        tool_input = {"x": "7"}

        assert coerce_tool_input(tool_input, {"type": "object"}) == tool_input
        assert coerce_tool_input(tool_input, {}) == tool_input


class _TapLikeTool(Tool):
    """Mimics a tool that compares integer arguments, as the Android tap tool does."""

    def __init__(self) -> None:
        super().__init__(
            name="tap_like_tool",
            description="Compares integer arguments.",
            input_schema=_TAP_SCHEMA,
        )

    def __call__(
        self, x: int, y: int, repeat: int = 1, repeat_delay_in_ms: int = 50
    ) -> str:
        if repeat_delay_in_ms < 0 or repeat < 1:
            error_msg = "invalid arguments"
            raise ValueError(error_msg)
        return f"Tapped at ({x}, {y}) {repeat}x"


class TestToolCollectionCoercesInput:
    def test_string_integers_reach_the_tool_as_integers(self) -> None:
        tool = _TapLikeTool()
        collection = ToolCollection(tools=[tool])
        tool_use = ToolUseBlockParam(
            id="tool_use_1",
            input={"x": "726", "y": "122", "repeat": "2", "repeat_delay_in_ms": "50"},
            name=tool.name,
        )

        results = collection.run([tool_use])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert result.is_error is None or result.is_error is False
        assert "Tapped at (726, 122) 2x" in str(result.content)

    def test_non_coercible_value_yields_error_result(self) -> None:
        tool = _TapLikeTool()
        collection = ToolCollection(tools=[tool])
        tool_use = ToolUseBlockParam(
            id="tool_use_1",
            input={"x": 726, "y": 122, "repeat": 1, "repeat_delay_in_ms": "abc"},
            name=tool.name,
        )

        results = collection.run([tool_use])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert result.is_error is True
        assert "not supported between instances of 'str' and 'int'" in str(
            result.content
        )
