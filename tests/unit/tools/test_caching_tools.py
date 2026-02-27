"""Unit tests for caching tools."""

import json
import tempfile
from pathlib import Path

import pytest

from askui.tools.caching_tools import (
    ExecuteCachedTrajectory,
    InspectCacheMetadata,
    RetrieveCachedTestExecutions,
    VerifyCacheExecution,
)


def _create_valid_cache_file(path: Path, is_valid: bool = True) -> None:
    """Create a valid cache file with required metadata structure."""
    cache_data = {
        "metadata": {
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "is_valid": is_valid,
            "execution_attempts": 0,
            "failures": [],
        },
        "trajectory": [],
        "cache_parameters": {},
    }
    path.write_text(json.dumps(cache_data), encoding="utf-8")


def test_retrieve_cached_test_executions_lists_json_files() -> None:
    """Test that RetrieveCachedTestExecutions lists all JSON files in cache dir."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create valid cache files
        _create_valid_cache_file(cache_dir / "cache1.json")
        _create_valid_cache_file(cache_dir / "cache2.json")
        (cache_dir / "not_cache.txt").write_text("text", encoding="utf-8")

        tool = RetrieveCachedTestExecutions(cache_dir=str(cache_dir))
        result = tool()

        assert len(result) == 2
        assert any("cache1.json" in path for path in result)
        assert any("cache2.json" in path for path in result)
        assert not any("not_cache.txt" in path for path in result)


def test_retrieve_cached_test_executions_returns_empty_list_when_no_files() -> None:
    """Test that RetrieveCachedTestExecutions returns empty list when no files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        tool = RetrieveCachedTestExecutions(cache_dir=str(cache_dir))
        result = tool()

        assert result == []


def test_retrieve_cached_test_executions_raises_error_when_dir_not_found() -> None:
    """Test that RetrieveCachedTestExecutions raises error if directory doesn't exist"""
    tool = RetrieveCachedTestExecutions(cache_dir="/non/existent/directory")

    with pytest.raises(FileNotFoundError, match="Trajectories directory not found"):
        tool()


def test_retrieve_cached_test_executions_respects_custom_format() -> None:
    """Test that RetrieveCachedTestExecutions respects custom file format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create files with different extensions
        _create_valid_cache_file(cache_dir / "cache1.json")
        _create_valid_cache_file(cache_dir / "cache2.traj")

        # Default format (.json)
        tool_json = RetrieveCachedTestExecutions(
            cache_dir=str(cache_dir), trajectories_format=".json"
        )
        result_json = tool_json()
        assert len(result_json) == 1
        assert "cache1.json" in result_json[0]

        # Custom format (.traj)
        tool_traj = RetrieveCachedTestExecutions(
            cache_dir=str(cache_dir), trajectories_format=".traj"
        )
        result_traj = tool_traj()
        assert len(result_traj) == 1
        assert "cache2.traj" in result_traj[0]


def test_retrieve_cached_test_executions_filters_invalid_by_default() -> None:
    """Test that invalid caches are filtered out by default."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create valid and invalid cache files
        _create_valid_cache_file(cache_dir / "valid.json", is_valid=True)
        _create_valid_cache_file(cache_dir / "invalid.json", is_valid=False)

        tool = RetrieveCachedTestExecutions(cache_dir=str(cache_dir))
        result = tool(include_invalid=False)

        assert len(result) == 1
        assert any("valid.json" in path for path in result)
        assert not any("invalid.json" in path for path in result)


def test_retrieve_cached_test_executions_includes_invalid_when_requested() -> None:
    """Test that invalid caches are included when include_invalid=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create valid and invalid cache files
        _create_valid_cache_file(cache_dir / "valid.json", is_valid=True)
        _create_valid_cache_file(cache_dir / "invalid.json", is_valid=False)

        tool = RetrieveCachedTestExecutions(cache_dir=str(cache_dir))
        result = tool(include_invalid=True)

        assert len(result) == 2
        assert any("valid.json" in path for path in result)
        assert any("invalid.json" in path for path in result)


def test_retrieve_cached_test_executions_returns_parameter_info() -> None:
    """Test that cache parameter info is included in the result."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create cache file with parameters
        cache_data = {
            "metadata": {
                "version": "1.0",
                "created_at": "2025-01-01T00:00:00Z",
                "is_valid": True,
                "execution_attempts": 0,
                "failures": [],
            },
            "trajectory": [],
            "cache_parameters": {"target_url": "placeholder", "user_id": "123"},
        }
        cache_file = cache_dir / "with_params.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        tool = RetrieveCachedTestExecutions(cache_dir=str(cache_dir))
        result = tool()

        assert len(result) == 1
        assert "parameters:" in result[0]
        assert "target_url" in result[0]


def test_execute_cached_trajectory_initializes_correctly() -> None:
    """Test that ExecuteCachedTrajectory initializes correctly."""
    tool = ExecuteCachedTrajectory()
    assert tool.name.startswith("execute_cached_executions_tool")
    assert "trajectory_file" in tool.input_schema["properties"]
    assert "start_from_step_index" in tool.input_schema["properties"]
    assert "parameter_values" in tool.input_schema["properties"]


def test_execute_cached_trajectory_returns_error_when_file_not_found() -> None:
    """Test that ExecuteCachedTrajectory returns error if file doesn't exist."""
    tool = ExecuteCachedTrajectory()

    result = tool(trajectory_file="/non/existent/file.json")

    assert "Trajectory file not found" in result
    assert "retrieve_available_trajectories_tool" in result


def test_execute_cached_trajectory_returns_success_when_file_exists() -> None:
    """Test that ExecuteCachedTrajectory returns success when file exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_trajectory.json"
        _create_valid_cache_file(cache_file)

        tool = ExecuteCachedTrajectory()
        result = tool(trajectory_file=str(cache_file))

        assert "Requesting cache execution" in result
        assert "test_trajectory.json" in result


def test_execute_cached_trajectory_accepts_start_from_step_index() -> None:
    """Test that ExecuteCachedTrajectory accepts start_from_step_index parameter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_trajectory.json"
        _create_valid_cache_file(cache_file)

        tool = ExecuteCachedTrajectory()
        result = tool(trajectory_file=str(cache_file), start_from_step_index=5)

        assert "Requesting cache execution" in result


def test_execute_cached_trajectory_accepts_parameter_values() -> None:
    """Test that ExecuteCachedTrajectory accepts parameter_values parameter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_trajectory.json"
        _create_valid_cache_file(cache_file)

        tool = ExecuteCachedTrajectory()
        result = tool(
            trajectory_file=str(cache_file),
            parameter_values={"target_url": "https://example.com"},
        )

        assert "Requesting cache execution" in result


def test_execute_cached_trajectory_does_not_execute_directly() -> None:
    """Test that ExecuteCachedTrajectory does NOT directly execute the trajectory.

    The tool should only validate the file exists and return a success message.
    Actual execution is handled by the CacheExecutor speaker.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_trajectory.json"
        _create_valid_cache_file(cache_file)

        tool = ExecuteCachedTrajectory()

        # Should succeed without any toolbox being set
        result = tool(trajectory_file=str(cache_file))

        # Should return a "requesting" message, not "successfully executed"
        assert "Requesting cache execution" in result
        assert "Successfully executed" not in result


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
                "version": "1.0",
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
        assert "Version: 1.0" in result
        assert "Is Valid: True" in result
        assert "Total Execution Attempts: 5" in result
        assert "Total Steps: 1" in result
        assert "url" in result
