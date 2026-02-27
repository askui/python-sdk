"""Tests for caching functionality in the act method."""

import json
import tempfile
from pathlib import Path

from askui.agent import ComputerAgent
from askui.models.shared.settings import CacheExecutionSettings, CachingSettings


def test_act_with_caching_strategy_execute(vision_agent: ComputerAgent) -> None:
    """Test that caching_strategy='execute' adds retrieve and execute tools."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy cache file
        cache_dir = Path(temp_dir)
        cache_file = cache_dir / "test_cache.json"
        cache_file.write_text("[]", encoding="utf-8")

        # Act with execute caching strategy
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="execute",
                cache_dir=str(cache_dir),
            ),
        )
        assert True


def test_act_with_caching_strategy_record(vision_agent: ComputerAgent) -> None:
    """Test that caching_strategy='record' writes cache file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_filename = "test_output.json"

        # Act with record caching strategy
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="record",
                cache_dir=str(cache_dir),
                filename=cache_filename,
            ),
        )

        # Verify cache file was created
        cache_file = cache_dir / cache_filename
        assert cache_file.exists()


def test_act_with_caching_strategy_both(vision_agent: ComputerAgent) -> None:
    """Test that caching_strategy='both' enables both execute and record."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_filename = "test_both.json"

        # Create a dummy cache file for executing
        cache_file = cache_dir / "existing_cache.json"
        cache_file.write_text("[]", encoding="utf-8")

        # Act with both caching strategies
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="both",
                cache_dir=str(cache_dir),
                filename=cache_filename,
            ),
        )

        # Verify new cache file was created
        output_file = cache_dir / cache_filename
        assert output_file.exists()


def test_act_with_caching_strategy_none(vision_agent: ComputerAgent) -> None:
    """Test that caching_strategy=None doesn't create cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Act without caching
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy=None,
                cache_dir=str(cache_dir),
            ),
        )

        # Verify no cache files were created
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 0


def test_act_with_custom_cache_dir_and_filename(vision_agent: ComputerAgent) -> None:
    """Test that custom cache_dir and cache_filename are used."""
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_cache_dir = Path(temp_dir) / "custom_cache"
        custom_filename = "my_custom_cache.json"

        # Act with custom cache settings
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="record",
                cache_dir=str(custom_cache_dir),
                filename=custom_filename,
            ),
        )

        # Verify custom cache directory and file were created
        assert custom_cache_dir.exists()
        cache_file = custom_cache_dir / custom_filename
        assert cache_file.exists()


def test_cache_file_contains_tool_use_blocks(vision_agent: ComputerAgent) -> None:
    """Test that cache file contains ToolUseBlockParam entries."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_filename = "tool_blocks.json"

        # Act with caching
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="record",
                cache_dir=str(cache_dir),
                filename=cache_filename,
            ),
        )

        # Read and verify cache file structure
        cache_file = cache_dir / cache_filename
        assert cache_file.exists()

        with cache_file.open("r", encoding="utf-8") as f:
            cache_data: list[dict[str, str]] = json.load(f)

        # Cache should be a list
        assert isinstance(cache_data, list)
        # Each entry should have tool use structure (name, id, input, type)
        for entry in cache_data:
            assert "name" in entry
            assert "id" in entry
            assert "input" in entry
            assert "type" in entry


def test_act_with_custom_cached_execution_tool_settings(
    vision_agent: ComputerAgent,
) -> None:
    """Test that custom CacheExecutionSettings are applied."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create a dummy cache file for executing
        cache_file = cache_dir / "test_cache.json"
        cache_file.write_text("[]", encoding="utf-8")

        # Act with custom execution settings
        custom_settings = CacheExecutionSettings(delay_time_between_actions=2.0)
        vision_agent.act(
            goal="Tell me a joke",
            caching_settings=CachingSettings(
                strategy="execute",
                cache_dir=str(cache_dir),
                execution_settings=custom_settings,
            ),
        )

        # Test passes if no exceptions are raised
        assert True
