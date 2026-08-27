"""Unit tests for the caching helper logic on the Agent base class."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from askui.agent_base import Agent
from askui.models.shared.agent_message_param import MessageParam, TextBlockParam
from askui.models.shared.settings import (
    CacheExecutionSettings,
    CacheFile,
    CacheMetadata,
    CacheWritingSettings,
    CachingSettings,
)


class TestCachingSettingsRejectUnknownFields:
    def test_misplaced_delay_raises_instead_of_being_ignored(self) -> None:
        # delay_time_between_actions belongs on execution_settings, not here.
        with pytest.raises(ValidationError):
            CachingSettings(strategy="execute", delay_time_between_actions=3.0)  # type: ignore[call-arg]

    def test_delay_on_execution_settings_is_applied(self) -> None:
        settings = CachingSettings(
            strategy="execute",
            execution_settings=CacheExecutionSettings(delay_time_between_actions=3.0),
        )
        assert settings.execution_settings is not None
        assert settings.execution_settings.delay_time_between_actions == 3.0


def _cache_file(is_valid: bool = True, parameters: dict | None = None) -> CacheFile:
    return CacheFile(
        metadata=CacheMetadata(
            created_at=datetime.now(tz=timezone.utc),
            is_valid=is_valid,
            invalidation_reason=None if is_valid else "too many failures",
        ),
        trajectory=[],
        cache_parameters=parameters or {},
    )


class TestResolveCacheFilename:
    def test_prefers_top_level_filename(self) -> None:
        settings = CachingSettings(
            filename="top.json",
            writing_settings=CacheWritingSettings(filename="nested.json"),
        )
        assert Agent._resolve_cache_filename(settings) == "top.json"

    def test_falls_back_to_writing_settings(self) -> None:
        settings = CachingSettings(
            writing_settings=CacheWritingSettings(filename="nested.json")
        )
        assert Agent._resolve_cache_filename(settings) == "nested.json"

    def test_empty_when_neither_set(self) -> None:
        assert Agent._resolve_cache_filename(CachingSettings()) == ""


class TestResolveTrajectoryPath:
    def test_adds_json_suffix(self) -> None:
        assert Agent._resolve_trajectory_path("dir", "login") == Path("dir/login.json")

    def test_keeps_existing_json_suffix(self) -> None:
        assert Agent._resolve_trajectory_path("dir", "login.json") == Path(
            "dir/login.json"
        )

    def test_preserves_nested_subdirectories(self) -> None:
        assert Agent._resolve_trajectory_path(
            ".askui_cache", "mytests_1/test_something"
        ) == Path(".askui_cache/mytests_1/test_something.json")

    def test_rejects_filename_escaping_cache_dir(self) -> None:
        for bad in ("/abs/name", "../escape", "a/../../b"):
            with pytest.raises(ValueError, match="relative to cache_dir"):
                Agent._resolve_trajectory_path(".askui_cache", bad)


class TestReadTrajectoryIfPresent:
    def test_missing_file_returns_none(self) -> None:
        assert Agent._read_trajectory_if_present(Path("/nope/x.json")) is None

    def test_reads_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "c.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "0.3",
                            "created_at": "2025-01-01T00:00:00Z",
                            "is_valid": True,
                            "execution_attempts": 0,
                            "failures": [],
                        },
                        "trajectory": [],
                        "cache_parameters": {},
                    }
                ),
                encoding="utf-8",
            )
            result = Agent._read_trajectory_if_present(path)
            assert result is not None
            assert result.metadata.version == "0.3"

    def test_unreadable_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text("{ this is not valid json", encoding="utf-8")
            assert Agent._read_trajectory_if_present(path) is None


class TestBuildCacheExecutionHint:
    def test_includes_path_and_switch_instruction(self) -> None:
        hint = Agent._build_cache_execution_hint(Path("dir/login.json"), _cache_file())
        assert "<CACHED_TRAJECTORY_AVAILABLE>" in hint
        assert "dir/login.json" in hint
        assert "switch_speaker(speaker_name='CacheExecutor'" in hint
        assert "no parameters" in hint

    def test_lists_parameters(self) -> None:
        hint = Agent._build_cache_execution_hint(
            Path("dir/login.json"),
            _cache_file(parameters={"username": "the login name"}),
        )
        assert "username: the login name" in hint
        assert "'username': '<value>'" in hint

    def test_invalid_cache_is_flagged(self) -> None:
        hint = Agent._build_cache_execution_hint(
            Path("dir/login.json"), _cache_file(is_valid=False)
        )
        assert "INVALID" in hint
        assert "too many failures" in hint


class TestInjectCacheHint:
    def test_appends_to_string_content(self) -> None:
        messages = [MessageParam(role="user", content="do the thing")]
        result = Agent._inject_cache_hint(messages, "HINT")
        assert result[0].content == "do the thing\n\nHINT"

    def test_appends_block_to_list_content(self) -> None:
        messages = [
            MessageParam(
                role="user",
                content=[TextBlockParam(type="text", text="do the thing")],
            )
        ]
        result = Agent._inject_cache_hint(messages, "HINT")
        assert isinstance(result[0].content, list)
        last_block = result[0].content[-1]
        assert isinstance(last_block, TextBlockParam)
        assert last_block.text == "HINT"

    def test_empty_messages_is_noop(self) -> None:
        assert Agent._inject_cache_hint([], "HINT") == []

    def test_targets_first_user_message_not_index_zero(self) -> None:
        messages = [
            MessageParam(role="assistant", content="prior assistant turn"),
            MessageParam(role="user", content="the goal"),
        ]
        result = Agent._inject_cache_hint(messages, "HINT")
        assert result[0].content == "prior assistant turn"
        assert result[1].content == "the goal\n\nHINT"

    def test_no_user_message_is_noop(self) -> None:
        messages = [MessageParam(role="assistant", content="only assistant")]
        result = Agent._inject_cache_hint(messages, "HINT")
        assert result[0].content == "only assistant"
