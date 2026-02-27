"""Cache manager for handling cache metadata, validation, and recording."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from askui.models.shared.agent_message_param import (
    MessageParam,
    ToolUseBlockParam,
    UsageParam,
)
from askui.models.shared.settings import (
    CacheFailure,
    CacheFile,
    CacheMetadata,
    CacheWritingSettings,
    VisualValidationMetadata,
)
from askui.models.shared.tools import ToolCollection
from askui.utils.caching.cache_parameter_handler import CacheParameterHandler
from askui.utils.caching.cache_validator import (
    CacheValidator,
    CompositeCacheValidator,
    StaleCacheValidator,
    StepFailureCountValidator,
    TotalFailureRateValidator,
)
from askui.utils.visual_validation import (
    compute_ahash,
    compute_phash,
    extract_region,
    find_recent_screenshot,
    get_validation_coordinate,
)

if TYPE_CHECKING:
    from askui.model_providers.vlm_provider import VlmProvider

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache metadata, validation, updates, and recording.

    This class provides high-level operations for cache management including:
    - Reading cache files from disk
    - Writing cache files to disk
    - Recording trajectories during execution (write mode)
    - Recording execution attempts and failures
    - Validating caches using pluggable validation strategies
    - Invalidating caches when they fail validation
    - Updating metadata on disk
    """

    def __init__(self, validators: list[CacheValidator] | None = None) -> None:
        """Initialize cache manager.

        Args:
            validators: Optional list of cache validators. If None, uses default
                validators (StepFailureCount, TotalFailureRate, StaleCache).
        """
        # Validation
        if validators is None:
            # Use default validators
            self.validators = CompositeCacheValidator(
                [
                    StepFailureCountValidator(max_failures_per_step=3),
                    TotalFailureRateValidator(min_attempts=10, max_failure_rate=0.5),
                    StaleCacheValidator(max_age_days=30),
                ]
            )
        else:
            self.validators = CompositeCacheValidator(validators)

        # Recording state (for write mode)
        self._recording = False
        self._tool_blocks: list[ToolUseBlockParam] = []
        self._cache_dir: Path | None = None
        self._file_name: str = ""
        self._goal: str | None = None
        self._toolbox: ToolCollection | None = None
        self._accumulated_usage = UsageParam()
        self._was_cached_execution = False
        self._cache_writer_settings = CacheWritingSettings()
        self._vlm_provider: "VlmProvider | None" = None

    def set_toolbox(self, toolbox: ToolCollection) -> None:
        """Set the toolbox for checking which tools are cacheable.

        Args:
            toolbox: ToolCollection to use for cacheable tool detection
        """
        self._toolbox = toolbox

    def record_execution_attempt(
        self,
        cache_file: CacheFile,
        success: bool,
        failure_info: CacheFailure | None = None,
    ) -> None:
        """Record an execution attempt in cache metadata.

        Args:
            cache_file: The cache file to update
            success: Whether the execution was successful
            failure_info: Optional failure information (required if success=False)
        """
        cache_file.metadata.execution_attempts += 1
        cache_file.metadata.last_executed_at = datetime.now(tz=timezone.utc)

        if not success and failure_info:
            cache_file.metadata.failures.append(failure_info)
            logger.debug(
                "Recorded failure at step %d: %s",
                failure_info.step_index,
                failure_info.error_message,
            )

    def record_step_failure(
        self, cache_file: CacheFile, step_index: int, error_message: str
    ) -> None:
        """Record a step failure in cache metadata.

        Args:
            cache_file: The cache file to update
            step_index: The step index where failure occurred
            error_message: Error message describing the failure
        """
        # Count existing failures at this step
        failures_at_step = sum(
            1 for f in cache_file.metadata.failures if f.step_index == step_index
        )

        failure = CacheFailure(
            timestamp=datetime.now(tz=timezone.utc),
            step_index=step_index,
            error_message=error_message,
            failure_count_at_step=failures_at_step + 1,
        )
        cache_file.metadata.failures.append(failure)
        logger.debug("Recorded failure at step %d", step_index)

    def should_invalidate(
        self, cache_file: CacheFile, step_index: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if cache should be invalidated.

        Args:
            cache_file: The cache file to validate
            step_index: Optional step index where failure occurred

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
        """
        return self.validators.should_invalidate(cache_file, step_index)

    def invalidate_cache(self, cache_file: CacheFile, reason: str) -> None:
        """Invalidate a cache file.

        Args:
            cache_file: The cache file to invalidate
            reason: Reason for invalidation
        """
        cache_file.metadata.is_valid = False
        cache_file.metadata.invalidation_reason = reason
        logger.warning("Cache invalidated: %s", reason)

    def mark_cache_valid(self, cache_file: CacheFile) -> None:
        """Mark a cache file as valid.

        This resets the is_valid flag to True and clears the invalidation reason.
        Useful for manual revalidation of caches that were previously invalidated.

        Args:
            cache_file: The cache file to mark as valid
        """
        cache_file.metadata.is_valid = True
        cache_file.metadata.invalidation_reason = None
        logger.info("Cache marked as valid")

    def get_failure_count_for_step(self, cache_file: CacheFile, step_index: int) -> int:
        """Get the total number of failures for a specific step.

        Args:
            cache_file: The cache file to check
            step_index: The step index to get failure count for

        Returns:
            Number of failures recorded for the given step index
        """
        return sum(
            1 for f in cache_file.metadata.failures if f.step_index == step_index
        )

    def update_metadata_on_failure(
        self,
        cache_file: CacheFile,
        cache_file_path: str,
        step_index: int,
        error_message: str,
    ) -> None:
        """Update cache metadata after execution failure and write to disk.

        This is a convenience method that combines recording the failure,
        checking validation, potentially invalidating, and writing to disk.

        Args:
            cache_file: The cache file to update
            cache_file_path: Path to write the updated cache file
            step_index: The step index where failure occurred
            error_message: Error message describing the failure
        """
        try:
            # Record the attempt and failure
            self.record_execution_attempt(cache_file, success=False)
            self.record_step_failure(
                cache_file, step_index=step_index, error_message=error_message
            )

            # Check if cache should be invalidated
            should_inv, reason = self.should_invalidate(
                cache_file, step_index=step_index
            )
            if should_inv and reason:
                self.invalidate_cache(cache_file, reason=reason)

            # Write updated metadata back to disk
            self._write_cache_file(cache_file, cache_file_path)
            logger.debug(
                "Updated cache metadata after failure: %s", Path(cache_file_path).name
            )
        except Exception:
            logger.exception("Failed to update cache metadata")

    def update_metadata_on_completion(
        self,
        cache_file: CacheFile,
        cache_file_path: str,
        success: bool,
    ) -> None:
        """Update cache metadata after execution completion and write to disk.

        Args:
            cache_file: The cache file to update
            cache_file_path: Path to write the updated cache file
            success: Whether the execution was successful
        """
        try:
            self.record_execution_attempt(cache_file, success=success)

            # Write updated metadata back to disk
            self._write_cache_file(cache_file, cache_file_path)
            logger.info("Updated cache metadata: %s", Path(cache_file_path).name)
        except Exception:
            logger.exception("Failed to update cache metadata")

    def _write_cache_file(self, cache_file: CacheFile, cache_file_path: str) -> None:
        """Write cache file to disk.

        Args:
            cache_file: The cache file to write
            cache_file_path: Path to write the cache file
        """
        cache_path = Path(cache_file_path)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(
                cache_file.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    @staticmethod
    def read_cache_file(cache_file_path: Path) -> CacheFile:
        """Read cache file with backward compatibility for legacy format.

        Supports two formats:
        1. Legacy format: Just a list of ToolUseBlockParam dicts
        2. New format: CacheFile with metadata and trajectory

        Args:
            cache_file_path: Path to the cache file

        Returns:
            CacheFile object with metadata and trajectory
        """
        logger.debug("Reading cache file: %s", cache_file_path)
        with cache_file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Handle legacy format (just a list of tool blocks)
        if isinstance(raw_data, list):
            logger.info("Detected legacy cache format, converting to CacheFile")
            trajectory = [ToolUseBlockParam(**step) for step in raw_data]
            cache_file = CacheFile(
                metadata=CacheMetadata(
                    version="0.0",
                    created_at=datetime.now(tz=timezone.utc),
                ),
                trajectory=trajectory,
            )
        else:
            cache_file = CacheFile(**raw_data)

        logger.info(
            "Successfully loaded cache: %s steps, %s parameters",
            len(cache_file.trajectory),
            len(cache_file.cache_parameters),
        )
        if cache_file.metadata.goal:
            logger.debug("Cache goal: %s", cache_file.metadata.goal)
        return cache_file

    def start_recording(
        self,
        cache_dir: str | Path,
        file_name: str = "",
        goal: str | None = None,
        toolbox: ToolCollection | None = None,
        cache_writer_settings: CacheWritingSettings | None = None,
        vlm_provider: "VlmProvider | None" = None,
    ) -> None:
        """Start recording a new trajectory.

        Args:
            cache_dir: Directory to store cache files
            file_name: Filename for cache file (auto-generated if not provided)
            goal: Goal string for this execution
            toolbox: ToolCollection to check which tools are cacheable
            cache_writer_settings: Settings for cache recording
            vlm_provider: VlmProvider instance to use for parameter identification
        """
        self._recording = True
        self._tool_blocks = []
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._file_name = (
            file_name
            if file_name.endswith(".json") or not file_name
            else f"{file_name}.json"
        )
        self._goal = goal
        self._toolbox = toolbox
        self._accumulated_usage = UsageParam()
        self._was_cached_execution = False
        self._cache_writer_settings = cache_writer_settings or CacheWritingSettings()
        self._vlm_provider = vlm_provider or self._vlm_provider

        logger.info(
            "Started recording trajectory to %s",
            self._cache_dir / (self._file_name or "[auto-generated]"),
        )

    def finish_recording(self, messages: list[MessageParam]) -> str:
        """Finish recording and write cache file to disk.

        Extracts tool blocks and usage from the message history.

        Args:
            messages: Complete message history from the conversation

        Returns:
            Success message with cache file path
        """
        if not self._recording:
            return "No recording in progress"

        # Extract tool blocks and usage from message history
        self._extract_from_messages(messages)

        if self._was_cached_execution:
            logger.info("Will not write cache file as this was a cached execution")
            self._reset_recording_state()
            return "Skipped writing cache (was cached execution)"

        # Blank non-cacheable tool inputs BEFORE parameterization
        # (so they don't get sent to LLM for parameter identification)
        if self._toolbox is not None:
            self._tool_blocks = self._blank_non_cacheable_tool_inputs(self._tool_blocks)
        else:
            logger.info("No toolbox set, skipping non-cacheable tool input blanking")

        # Auto-generate filename if not provided
        if not self._file_name:
            self._file_name = (
                f"cached_trajectory_{datetime.now(tz=timezone.utc):%Y%m%d%H%M%S%f}.json"
            )

        assert isinstance(self._cache_dir, Path)
        cache_file_path = self._cache_dir / self._file_name

        # Parameterize trajectory (this creates NEW tool blocks)
        goal_to_save, trajectory_to_save, parameters_dict = (
            self._parameterize_trajectory()
        )

        # Add visual validation hashes to trajectory AFTER parameterization
        # (so visual_representation fields don't get lost during parameterization)
        self._add_visual_validation_to_trajectory(trajectory_to_save, messages)

        # Generate cache file
        self._generate_cache_file(
            goal_to_save, trajectory_to_save, parameters_dict, cache_file_path
        )

        # Reset recording state
        self._reset_recording_state()

        return f"Cache file written: {cache_file_path}"

    def _reset_recording_state(self) -> None:
        """Reset all recording state variables."""
        self._recording = False
        self._tool_blocks = []
        self._file_name = ""
        self._was_cached_execution = False
        self._accumulated_usage = UsageParam()

    def _extract_from_messages(self, messages: list[MessageParam]) -> None:
        """Extract tool blocks and usage from message history.

        Args:
            messages: Complete message history from the conversation
        """
        for message in messages:
            if message.role == "assistant":
                contents = message.content
                if isinstance(contents, list):
                    for content in contents:
                        if isinstance(content, ToolUseBlockParam):
                            self._tool_blocks.append(content)
                            # Check if this was a cached execution
                            if content.name == "execute_cached_executions_tool":
                                self._was_cached_execution = True

                # Accumulate usage from assistant messages
                if message.usage:
                    self._accumulate_usage(message.usage)

    def _parameterize_trajectory(
        self,
    ) -> tuple[str | None, list[ToolUseBlockParam], dict[str, str]]:
        """Identify parameters and return parameterized trajectory + goal."""
        return CacheParameterHandler.identify_and_parameterize(
            trajectory=self._tool_blocks,
            goal=self._goal,
            identification_strategy=self._cache_writer_settings.parameter_identification_strategy,
            vlm_provider=self._vlm_provider,
        )

    def _blank_non_cacheable_tool_inputs(
        self, trajectory: list[ToolUseBlockParam]
    ) -> list[ToolUseBlockParam]:
        """Blank out input fields for non-cacheable tools to save space.

        For tools marked as is_cacheable=False, we replace their input with an
        empty dict since we won't be executing them from cache anyway.

        Args:
            trajectory: The trajectory to process

        Returns:
            New trajectory with non-cacheable tool inputs blanked out
        """
        if self._toolbox is None:
            return trajectory

        blanked_count = 0
        result: list[ToolUseBlockParam] = []
        tools = self._toolbox.tool_map
        for tool_block in trajectory:
            # Check if this tool is cacheable
            tool = tools.get(tool_block.name)

            # If tool is not cacheable, blank out its input
            if tool is not None and not tool.is_cacheable:
                logger.debug(
                    "Blanking input for non-cacheable tool: %s", tool_block.name
                )
                blanked_count += 1
                result.append(
                    ToolUseBlockParam(
                        id=tool_block.id,
                        name=tool_block.name,
                        input={},  # Blank out the input
                        type=tool_block.type,
                        cache_control=tool_block.cache_control,
                    )
                )
            else:
                # Keep the tool block as-is
                result.append(tool_block)

        if blanked_count > 0:
            logger.info(
                "Blanked inputs for %s non-cacheable tool(s) to save space",
                blanked_count,
            )

        return result

    def _add_visual_validation_to_trajectory(  # noqa: C901
        self, trajectory: list[ToolUseBlockParam], messages: list[MessageParam]
    ) -> None:
        """Add visual validation hashes to tool use blocks in the trajectory.

        This method processes the complete message history to find screenshots
        and compute visual hashes for actions that require validation.
        The hashes are stored in the visual_representation field of each
        ToolUseBlockParam in the provided trajectory.

        Args:
            trajectory: The parameterized trajectory to add validation to
            messages: Complete message history from the conversation
        """
        if self._cache_writer_settings.visual_verification_method == "none":
            logger.info("Visual validation disabled, skipping hash computation")
            return

        # Build a mapping from tool_use_id to tool_block in the trajectory
        # This allows us to update the correct tool block in the trajectory
        tool_block_map: dict[str, ToolUseBlockParam] = {
            block.id: block for block in trajectory
        }

        # Iterate through messages to find tool uses and their context
        validated_count = 0
        for i, message in enumerate(messages):
            if message.role != "assistant":
                continue

            if isinstance(message.content, str):
                continue

            # Process tool use blocks in this message
            for block in message.content:
                if block.type != "tool_use":
                    continue

                # Find the corresponding block in the trajectory
                trajectory_block = tool_block_map.get(block.id)
                if not trajectory_block:
                    # This tool use is not in the trajectory (might be non-cacheable)
                    continue

                # Check if this tool has coordinates for visual validation
                tool_input: dict[str, Any] = (
                    block.input if isinstance(block.input, dict) else {}
                )
                coordinate = get_validation_coordinate(tool_input)
                if coordinate is None:
                    # Tools without coordinates don't need visual validation
                    trajectory_block.visual_representation = None
                    continue

                # Find most recent screenshot BEFORE this tool use
                screenshot = find_recent_screenshot(messages, from_index=i - 1)
                if not screenshot:
                    logger.warning(
                        "No screenshot found before tool_id=%s, "
                        "skipping visual validation",
                        block.id,
                    )
                    trajectory_block.visual_representation = None
                    continue

                # Extract region and compute hash
                try:
                    # Pass coordinate in the format extract_region expects
                    region = extract_region(
                        screenshot,
                        {"coordinate": list(coordinate)},
                        region_size=self._cache_writer_settings.visual_validation_region_size,
                    )
                    visual_hash = self._compute_visual_hash(
                        region, self._cache_writer_settings.visual_verification_method
                    )
                    trajectory_block.visual_representation = visual_hash
                    validated_count += 1
                    logger.debug(
                        "Added visual validation hash for tool_id=%s",
                        block.id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to compute visual hash for tool_id=%s", block.id
                    )
                    trajectory_block.visual_representation = None

        if validated_count > 0:
            logger.info(
                "Added visual validation to %d action(s) in trajectory",
                validated_count,
            )

    def _compute_visual_hash(self, image: Image.Image, method: str) -> str:
        """Compute visual hash using specified method.

        Args:
            image: PIL Image to hash
            method: Hash method ("phash", "ahash", or "none")

        Returns:
            String representation of the hash

        Raises:
            ValueError: If method is not supported
        """
        if method == "phash":
            return compute_phash(image, hash_size=8)
        if method == "ahash":
            return compute_ahash(image, hash_size=8)
        if method == "none":
            return ""
        msg = f"Unsupported visual verification method: {method}"
        raise ValueError(msg)

    def _generate_cache_file(
        self,
        goal_to_save: str | None,
        trajectory_to_save: list[ToolUseBlockParam],
        parameters_dict: dict[str, str],
        cache_file_path: Path,
    ) -> None:
        """Write cache file to disk with metadata.

        Args:
            goal_to_save: Goal string (may be parameterized)
            trajectory_to_save: Trajectory (parameterized and blanked)
            parameters_dict: Cache parameters dictionary
            cache_file_path: Path to write cache file
        """
        # Prepare visual validation metadata
        visual_validation_metadata: VisualValidationMetadata | None = None
        if self._cache_writer_settings.visual_verification_method != "none":
            visual_validation_metadata = VisualValidationMetadata(
                enabled=True,
                method=self._cache_writer_settings.visual_verification_method,
                region_size=self._cache_writer_settings.visual_validation_region_size,
            )

        cache_file = CacheFile(
            metadata=CacheMetadata(
                version="0.2",
                created_at=datetime.now(tz=timezone.utc),
                goal=goal_to_save,
                token_usage=self._accumulated_usage,
                visual_validation=visual_validation_metadata,
            ),
            trajectory=trajectory_to_save,
            cache_parameters=parameters_dict,
        )

        with cache_file_path.open("w", encoding="utf-8") as f:
            json.dump(cache_file.model_dump(mode="json"), f, indent=4)
        logger.info("Cache file successfully written: %s", cache_file_path)

    def _accumulate_usage(self, step_usage: UsageParam) -> None:
        """Accumulate usage statistics from a single API call.

        Args:
            step_usage: Usage from a single message
        """
        self._accumulated_usage.input_tokens = (
            self._accumulated_usage.input_tokens or 0
        ) + (step_usage.input_tokens or 0)
        self._accumulated_usage.output_tokens = (
            self._accumulated_usage.output_tokens or 0
        ) + (step_usage.output_tokens or 0)
        self._accumulated_usage.cache_creation_input_tokens = (
            self._accumulated_usage.cache_creation_input_tokens or 0
        ) + (step_usage.cache_creation_input_tokens or 0)
        self._accumulated_usage.cache_read_input_tokens = (
            self._accumulated_usage.cache_read_input_tokens or 0
        ) + (step_usage.cache_read_input_tokens or 0)
