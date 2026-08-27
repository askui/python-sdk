"""Tests for CacheManager recording write/skip behavior."""

import tempfile
from pathlib import Path
from typing import Any

from askui.models.shared.agent_message_param import MessageParam, ToolUseBlockParam
from askui.models.shared.settings import CacheWritingSettings
from askui.models.shared.tools import Tool, ToolCollection
from askui.utils.caching.cache_manager import CacheManager


class _CacheableTool(Tool):
    def __init__(self, cacheable: bool) -> None:
        super().__init__(name="mini_tool", description="mini")
        self.is_cacheable = cacheable

    def __call__(self, **_: Any) -> str:
        return "ok"


def _assistant_tool_use(tool_name: str) -> MessageParam:
    return MessageParam(
        role="assistant",
        content=[ToolUseBlockParam(id="0", name=tool_name, input={"x": 1})],
    )


def test_finish_recording_skips_when_no_cacheable_steps() -> None:
    """A run with no cacheable steps must not write (or overwrite) a cache file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CacheManager()
        manager.start_recording(
            cache_dir=temp_dir,
            file_name="out.json",
            cache_writer_settings=CacheWritingSettings(
                visual_verification_method="none"
            ),
        )
        # No assistant tool_use messages -> empty trajectory.
        result = manager.finish_recording([MessageParam(role="user", content="hi")])

        assert "no cacheable steps" in result.lower()
        assert not (Path(temp_dir) / "out.json").exists()


def test_finish_recording_skips_when_only_non_cacheable_steps() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        toolbox = ToolCollection(tools=[_CacheableTool(cacheable=False)])
        tool_name = next(iter(toolbox.tool_map.keys()))

        manager = CacheManager()
        manager.start_recording(
            cache_dir=temp_dir,
            file_name="out.json",
            toolbox=toolbox,
            cache_writer_settings=CacheWritingSettings(
                visual_verification_method="none"
            ),
        )
        result = manager.finish_recording([_assistant_tool_use(tool_name)])

        assert "no cacheable steps" in result.lower()
        assert not (Path(temp_dir) / "out.json").exists()


def test_finish_recording_writes_when_cacheable_step_present() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        toolbox = ToolCollection(tools=[_CacheableTool(cacheable=True)])
        tool_name = next(iter(toolbox.tool_map.keys()))

        manager = CacheManager()
        manager.start_recording(
            cache_dir=temp_dir,
            file_name="out.json",
            toolbox=toolbox,
            cache_writer_settings=CacheWritingSettings(
                visual_verification_method="none"
            ),
        )
        result = manager.finish_recording([_assistant_tool_use(tool_name)])

        out = Path(temp_dir) / "out.json"
        assert out.exists()
        assert "Cache file written" in result
        written = CacheManager.read_cache_file(out)
        assert written.metadata.version == "0.3"
        assert len(written.trajectory) == 1
