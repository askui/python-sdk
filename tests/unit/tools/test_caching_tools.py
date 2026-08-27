"""Unit tests for caching tools."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from askui.models.shared.settings import CacheFile, CacheMetadata
from askui.speaker.cache_executor import CacheExecutor
from askui.tools.caching_tools import (
    InspectCacheMetadata,
    VerifyCacheExecution,
)
from askui.utils.caching.cache_manager import CacheManager


def _write_cache_file(path: Path, is_valid: bool = True) -> None:
    """Create a valid cache file with required metadata structure."""
    cache_data = {
        "metadata": {
            "version": "0.3",
            "created_at": "2025-01-01T00:00:00Z",
            "is_valid": is_valid,
            "execution_attempts": 0,
            "failures": [],
        },
        "trajectory": [],
        "cache_parameters": {},
    }
    path.write_text(json.dumps(cache_data), encoding="utf-8")


def _activated_cache_executor(cache_file_path: Path) -> CacheExecutor:
    """Return a CacheExecutor with an activated (loaded) cache file."""
    executor = CacheExecutor()
    executor._cache_file = CacheManager.read_cache_file(cache_file_path)
    executor._cache_file_path = str(cache_file_path)
    return executor


def test_verify_cache_execution_initializes_correctly() -> None:
    """Test that VerifyCacheExecution initializes correctly."""
    tool = VerifyCacheExecution()
    assert tool.name.startswith("verify_cache_execution")
    assert "success" in tool.input_schema["properties"]
    assert "verification_notes" in tool.input_schema["properties"]
    assert tool.is_cacheable is False


def test_verify_cache_execution_reports_success() -> None:
    """Test that VerifyCacheExecution reports success correctly."""
    tool = VerifyCacheExecution()
    result = tool(success=True, verification_notes="UI state matches expected")

    assert "success=True" in result
    assert "UI state matches expected" in result


def test_verify_cache_execution_reports_failure() -> None:
    """Test that VerifyCacheExecution reports failure correctly."""
    tool = VerifyCacheExecution()
    result = tool(success=False, verification_notes="Button was not clicked")

    assert "success=False" in result
    assert "Button was not clicked" in result


def test_verify_cache_execution_success_updates_metadata() -> None:
    """A successful verification records the execution attempt on disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "trajectory.json"
        _write_cache_file(cache_path, is_valid=True)

        executor = _activated_cache_executor(cache_path)
        tool = VerifyCacheExecution(
            cache_executor=executor, cache_manager=CacheManager()
        )
        tool(success=True, verification_notes="all good")

        persisted = CacheManager.read_cache_file(cache_path)
        assert persisted.metadata.is_valid is True
        assert persisted.metadata.execution_attempts == 1
        assert persisted.metadata.last_executed_at is not None


def test_verify_cache_execution_failure_invalidates_cache() -> None:
    """An unsuccessful verification invalidates the cache on disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "trajectory.json"
        _write_cache_file(cache_path, is_valid=True)

        executor = _activated_cache_executor(cache_path)
        tool = VerifyCacheExecution(
            cache_executor=executor, cache_manager=CacheManager()
        )
        tool(success=False, verification_notes="needed manual corrections")

        persisted = CacheManager.read_cache_file(cache_path)
        assert persisted.metadata.is_valid is False
        assert persisted.metadata.invalidation_reason is not None
        assert "needed manual corrections" in persisted.metadata.invalidation_reason


def test_verify_cache_execution_without_wiring_is_noop() -> None:
    """Without a wired executor/manager the tool only reports (no crash)."""
    tool = VerifyCacheExecution()
    # Should not raise even though there is nothing to persist.
    result = tool(success=False, verification_notes="no active execution")
    assert "success=False" in result


def test_inspect_cache_metadata_initializes_correctly() -> None:
    """Test that InspectCacheMetadata initializes correctly."""
    tool = InspectCacheMetadata()
    assert tool.name.startswith("inspect_cache_metadata_tool")
    assert "trajectory_file" in tool.input_schema["properties"]


def test_inspect_cache_metadata_returns_error_when_file_not_found() -> None:
    """Test that InspectCacheMetadata returns error if file doesn't exist."""
    tool = InspectCacheMetadata()

    result = tool(trajectory_file="/non/existent/file.json")

    assert "Trajectory file not found" in result


def test_inspect_cache_metadata_returns_metadata() -> None:
    """Test that InspectCacheMetadata returns formatted metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_cache.json"
        cache_data = {
            "metadata": {
                "version": "0.3",
                "created_at": "2025-01-01T00:00:00Z",
                "is_valid": True,
                "execution_attempts": 5,
                "failures": [],
            },
            "trajectory": [
                {"id": "1", "name": "click", "input": {}, "type": "tool_use"}
            ],
            "cache_parameters": {"url": "test"},
        }
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        tool = InspectCacheMetadata()
        result = tool(trajectory_file=str(cache_file))

        assert "=== Cache Metadata ===" in result
        assert "Version: 0.3" in result
        assert "Is Valid: True" in result
        assert "Total Execution Attempts: 5" in result
        assert "Total Steps: 1" in result
        assert "url" in result


def test_cache_manager_generates_version_0_3() -> None:
    """New cache files are written with the current 0.3 metadata version."""
    assert CacheMetadata(created_at=datetime.now(tz=timezone.utc)).version == "0.3"
    # Sanity: CacheFile round-trips with the new version.
    cache_file = CacheFile(
        metadata=CacheMetadata(created_at=datetime.now(tz=timezone.utc)),
        trajectory=[],
    )
    assert cache_file.metadata.version == "0.3"
