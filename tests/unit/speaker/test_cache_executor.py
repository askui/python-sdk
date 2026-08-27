"""Unit tests for the CacheExecutor speaker."""

import json
import tempfile
from pathlib import Path

import pytest

from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import ToolCollection
from askui.speaker.cache_executor import CacheExecutor, ExecutionResult
from askui.utils.caching.cache_manager import CacheManager


def _write_trajectory(path: Path, step_names: list[str]) -> None:
    """Write a cache file whose trajectory has one tool_use per given name."""
    cache_data = {
        "metadata": {
            "version": "0.3",
            "created_at": "2025-01-01T00:00:00Z",
            "is_valid": True,
            "execution_attempts": 0,
            "failures": [],
        },
        "trajectory": [
            {"id": str(i), "name": name, "input": {}, "type": "tool_use"}
            for i, name in enumerate(step_names)
        ],
        "cache_parameters": {},
    }
    path.write_text(json.dumps(cache_data), encoding="utf-8")


def _first_text_block(message: MessageParam) -> str:
    """Return the text of the first text block in a message's content."""
    assert isinstance(message.content, list)
    block = message.content[0]
    assert isinstance(block, TextBlockParam)
    return block.text


def _context(path: Path, start_from_step_index: int) -> dict:
    return {
        "trajectory_file": str(path),
        "start_from_step_index": start_from_step_index,
        "parameter_values": {},
        "toolbox": ToolCollection(),
    }


class TestStartIndexValidation:
    def test_resume_at_end_does_not_raise(self) -> None:
        """start_from_step_index == len(trajectory) means 'already complete'."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "t.json"
            _write_trajectory(path, ["click_a", "click_b", "click_c"])

            executor = CacheExecutor()
            # 3 steps -> index 3 is the "just past the end" resume index.
            executor._activate_from_context(_context(path, 3), CacheManager())

            assert executor._current_step_index == 3
            result = executor._get_next_step()
            assert result.status == "COMPLETED"

    def test_index_beyond_end_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "t.json"
            _write_trajectory(path, ["click_a", "click_b"])

            executor = CacheExecutor()
            with pytest.raises(ValueError, match="Invalid start_from_step_index"):
                executor._activate_from_context(_context(path, 3), CacheManager())

    def test_negative_index_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "t.json"
            _write_trajectory(path, ["click_a"])

            executor = CacheExecutor()
            with pytest.raises(ValueError, match="Invalid start_from_step_index"):
                executor._activate_from_context(_context(path, -1), CacheManager())

    def test_empty_trajectory_resume_at_zero_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "t.json"
            _write_trajectory(path, [])

            executor = CacheExecutor()
            executor._activate_from_context(_context(path, 0), CacheManager())
            result = executor._get_next_step()
            assert result.status == "COMPLETED"


class TestHasExecutableStepsFrom:
    def test_detects_remaining_executable_steps(self) -> None:
        executor = CacheExecutor()
        executor._trajectory = [
            ToolUseBlockParam(id="0", name="click_a", input={}),
            ToolUseBlockParam(id="1", name="click_b", input={}),
        ]
        assert executor._has_executable_steps_from(1) is True
        assert executor._has_executable_steps_from(2) is False

    def test_skippable_trailing_steps_are_ignored(self) -> None:
        executor = CacheExecutor()
        executor._trajectory = [
            ToolUseBlockParam(id="0", name="click_a", input={}),
            ToolUseBlockParam(id="1", name="switch_speaker_abc", input={}),
        ]
        # Only a skippable step remains after index 0 -> nothing executable.
        assert executor._has_executable_steps_from(1) is False


class TestNeedsAgentMessage:
    def test_last_step_message_tells_agent_not_to_resume(self) -> None:
        executor = CacheExecutor()
        executor._trajectory = [
            ToolUseBlockParam(id="0", name="click_a", input={}),
            ToolUseBlockParam(id="1", name="human_decision", input={}),
        ]
        result = ExecutionResult(
            status="NEEDS_AGENT",
            step_index=1,
            tool_result=executor._trajectory[1],
        )
        speaker_result = executor._handle_needs_agent(result)
        text = _first_text_block(speaker_result.messages_to_add[0])
        assert "FINAL step" in text
        assert "start_from_step_index" not in text

    def test_intermediate_step_message_provides_resume_index(self) -> None:
        executor = CacheExecutor()
        executor._trajectory = [
            ToolUseBlockParam(id="0", name="human_decision", input={}),
            ToolUseBlockParam(id="1", name="click_b", input={}),
        ]
        result = ExecutionResult(
            status="NEEDS_AGENT",
            step_index=0,
            tool_result=executor._trajectory[0],
        )
        speaker_result = executor._handle_needs_agent(result)
        text = _first_text_block(speaker_result.messages_to_add[0])
        assert "start_from_step_index=1" in text
