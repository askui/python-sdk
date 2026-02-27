import json
import logging
from pathlib import Path

from pydantic import ValidationError, validate_call
from typing_extensions import override

from ..models.shared.tools import Tool
from ..utils.caching.cache_manager import CacheManager

logger = logging.getLogger()


class RetrieveCachedTestExecutions(Tool):
    """
    List all available trajectory files that can be used for fast-forward execution
    """

    def __init__(self, cache_dir: str, trajectories_format: str = ".json") -> None:
        super().__init__(
            name="retrieve_available_trajectories_tool",
            description=(
                "Use this tool to list all available pre-recorded trajectory "
                "files in the trajectories directory. These trajectories "
                "represent successful UI interaction sequences that can be "
                "replayed using the execute_trajectory_tool. Call this tool "
                "first to see which trajectories are available before "
                "executing one. The tool returns a list of file paths to "
                "available trajectory files.\n\n"
                "By default, only valid (non-invalidated) caches are returned. "
                "Set include_invalid=True to see all caches including those "
                "marked as invalid due to repeated failures."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "include_invalid": {
                        "type": "boolean",
                        "description": (
                            "Whether to include invalid/invalidated caches in "
                            "the results. Default is False (only show valid "
                            "caches)."
                        ),
                        "default": False,
                    },
                },
                "required": [],
            },
        )
        self._cache_dir = Path(cache_dir)
        self._trajectories_format = trajectories_format
        self.is_cacheable = True

    @override
    @validate_call
    def __call__(self, include_invalid: bool = False) -> list[str]:  # type: ignore
        """Retrieve available cached trajectories.

        Args:
            include_invalid: Whether to include invalid caches

        Returns:
            List of strings with filename and parameters info.
        """
        logger.info(
            "Retrieving cached trajectories from %s (include_invalid=%s)",
            self._cache_dir,
            include_invalid,
        )

        if not Path.is_dir(self._cache_dir):
            error_msg = f"Trajectories directory not found: {self._cache_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_files = [
            f
            for f in self._cache_dir.iterdir()
            if str(f).endswith(self._trajectories_format)
        ]
        logger.debug("Found %d total cache files", len(all_files))

        available: list[str] = []
        invalid_count = 0
        unreadable_count = 0

        for f in all_files:
            try:
                cache_file = CacheManager.read_cache_file(f)

                # Check if we should include this cache
                if not include_invalid and not cache_file.metadata.is_valid:
                    invalid_count += 1
                    logger.debug(
                        "Excluding invalid cache: %s (reason: %s)",
                        f.name,
                        cache_file.metadata.invalidation_reason,
                    )
                    continue

                # Add cache info with filename and parameters
                available.append(
                    f"filename: {f!s} (parameters: {cache_file.cache_parameters})"
                )

            except Exception:  # noqa: PERF203
                unreadable_count += 1
                logger.exception("Failed to read cache file %s", f.name)
                continue

        logger.info(
            "Found %d cache(s), excluded %d invalid, %d unreadable",
            len(available),
            invalid_count,
            unreadable_count,
        )

        if not available:
            if include_invalid:
                warning_msg = f"Warning: No trajectory files found in {self._cache_dir}"
            else:
                warning_msg = (
                    f"Warning: No valid trajectory files found in "
                    f"{self._cache_dir}. "
                    "Try include_invalid=True to see all caches."
                )
            logger.warning(warning_msg)

        return available


class ExecuteCachedTrajectory(Tool):
    """
    Activate cache execution mode to replay a pre-recorded trajectory.

    This tool performs minimal validation (file exists) and signals that cache
    execution should start. The actual execution with visual validation is
    handled by the CacheExecutor speaker.
    """

    def __init__(self) -> None:
        super().__init__(
            name="execute_cached_executions_tool",
            description=(
                "Activate cache execution mode to replay a pre-recorded "
                "trajectory. This tool sets up the agent to execute cached UI "
                "interactions step-by-step.\n\n"
                "Before using this tool:\n"
                "1. Use retrieve_available_trajectories_tool to see which "
                "trajectory files are available\n"
                "2. Select the appropriate trajectory file path from the "
                "returned list\n"
                "3. If the trajectory contains parameters (e.g., {{target_url}}), "
                "provide values for them in the parameter_values parameter\n"
                "4. Pass the full file path to this tool\n\n"
                "Cache parameters allow dynamic values to be injected during "
                "execution. For example, if a trajectory types '{{target_url}}', "
                "you must provide parameter_values={'target_url': 'https://...'}.\n\n"
                "To continue from a specific step (e.g., after manually "
                "handling a non-cacheable step), use the start_from_step_index "
                "parameter. By default, execution starts from the beginning "
                "(step 0).\n\n"
                "Once activated, the agent will execute cached steps "
                "automatically. If a non-cacheable step is encountered, the "
                "agent will be asked to handle it manually before resuming "
                "cache execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "trajectory_file": {
                        "type": "string",
                        "description": (
                            "Full path to the trajectory file (use "
                            "retrieve_available_trajectories_tool to find "
                            "available files)"
                        ),
                    },
                    "start_from_step_index": {
                        "type": "integer",
                        "description": (
                            "Optional: The step index to start or resume "
                            "execution from (0-based). Use 0 (default) to start "
                            "from the beginning. Use a higher index to continue "
                            "from a specific step, e.g., after manually handling "
                            "a non-cacheable step."
                        ),
                        "default": 0,
                    },
                    "parameter_values": {
                        "type": "object",
                        "description": (
                            "Optional dictionary mapping parameter names to "
                            "their values. Required if the trajectory contains "
                            "parameters like {{variable}}. Example: "
                            "{'target_url': 'https://example.com'}"
                        ),
                        "additionalProperties": {"type": "string"},
                        "default": {},
                    },
                },
                "required": ["trajectory_file"],
            },
        )

    @override
    @validate_call
    def __call__(
        self,
        trajectory_file: str,
        start_from_step_index: int = 0,
        parameter_values: dict[str, str] | None = None,
    ) -> str:
        """Request cache execution.

        This tool performs minimal validation (file exists) and requests
        cache execution. Full validation and execution happens in CacheExecutor.

        Args:
            trajectory_file: Path to the trajectory file
            start_from_step_index: Step index to start from (default: 0)
            parameter_values: Parameter values for the trajectory

        Returns:
            Success message if file exists, error message otherwise
        """
        if parameter_values is None:
            parameter_values = {}

        logger.info(
            "Requesting cache execution: %s (start_from_step=%d)",
            Path(trajectory_file).name,
            start_from_step_index,
        )

        # Validate file exists
        if not Path(trajectory_file).is_file():
            error_msg = (
                f"Trajectory file not found: {trajectory_file}\n"
                "Use retrieve_available_trajectories_tool to see available files."
            )
            logger.error(error_msg)
            return error_msg

        # Validate file structure using CacheFile pydantic model
        try:
            CacheManager.read_cache_file(Path(trajectory_file))
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = (
                f"Invalid cache file format: {trajectory_file}\n"
                f"Error: {e}\n"
                "The cache file may be corrupted or in an old format."
            )
            logger.exception(error_msg)
            return error_msg

        # Return success - CacheExecutor will handle full execution
        return f"Requesting cache execution for {Path(trajectory_file).name}"


class VerifyCacheExecution(Tool):
    """Tool for agent to explicitly report cache execution verification results."""

    def __init__(self) -> None:
        super().__init__(
            name="verify_cache_execution",
            description=(
                "IMPORTANT: Call this tool immediately after reviewing a "
                "cached trajectory execution.\n\n"
                "Report whether the cached execution successfully achieved "
                "the target system state. You MUST call this tool to complete "
                "the cache verification process.\n\n"
                "Set success=True if:\n"
                "- The cached execution correctly achieved the intended goal\n"
                "- The final state matches what was expected\n"
                "- No corrections or additional actions were needed\n\n"
                "Set success=False if:\n"
                "- The execution did not achieve the target state\n"
                "- You had to make corrections or perform additional actions\n"
                "- The final state is incorrect or incomplete"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": (
                            "True if cached execution correctly "
                            "achieved target state, "
                            "False if execution was incorrect or "
                            "corrections were needed"
                        ),
                    },
                    "verification_notes": {
                        "type": "string",
                        "description": (
                            "Brief explanation of what you verified. "
                            "If success=False, describe what was "
                            "wrong and what corrections you made."
                        ),
                    },
                },
                "required": ["success", "verification_notes"],
            },
        )
        self.is_cacheable = False  # Verification is not cacheable

    @override
    @validate_call
    def __call__(self, success: bool, verification_notes: str) -> str:
        """Record cache verification result.

        Args:
            success: Whether cache execution achieved target state
            verification_notes: Explanation of verification result

        Returns:
            Confirmation message
        """
        message = (
            f"Cache verification reported: success={success}, "
            f"notes={verification_notes}"
        )
        logger.info(
            "Cache verification reported: success=%s, notes=%s",
            success,
            verification_notes,
        )
        return message


class InspectCacheMetadata(Tool):
    """
    Inspect detailed metadata for a cached trajectory file
    """

    def __init__(self) -> None:
        super().__init__(
            name="inspect_cache_metadata_tool",
            description=(
                "Inspect and display detailed metadata for a cached trajectory "
                "file. This tool shows information about:\n"
                "- Cache version and creation timestamp\n"
                "- Execution statistics (attempts, last execution time)\n"
                "- Validity status and invalidation reason (if invalid)\n"
                "- Failure history with timestamps and error messages\n"
                "- Parameters and trajectory step count\n\n"
                "Use this tool to debug cache issues or understand why a cache "
                "might be failing or invalidated."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "trajectory_file": {
                        "type": "string",
                        "description": (
                            "Full path to the trajectory file to inspect. "
                            "Use retrieve_available_trajectories_tool to "
                            "find available files."
                        ),
                    },
                },
                "required": ["trajectory_file"],
            },
        )

    @override
    @validate_call
    def __call__(self, trajectory_file: str) -> str:
        """Inspect cache metadata.

        Args:
            trajectory_file: Path to the trajectory file

        Returns:
            Formatted metadata string
        """
        logger.info("Inspecting cache metadata: %s", Path(trajectory_file).name)

        if not Path(trajectory_file).is_file():
            error_msg = (
                f"Trajectory file not found: {trajectory_file}\n"
                "Use retrieve_available_trajectories_tool to see available files."
            )
            logger.error(error_msg)
            return error_msg

        try:
            cache_file = CacheManager.read_cache_file(Path(trajectory_file))
        except Exception:
            error_msg = f"Failed to read cache file {Path(trajectory_file).name}"
            logger.exception(error_msg)
            return error_msg

        metadata = cache_file.metadata
        logger.debug(
            "Metadata loaded: version=%s, valid=%s, attempts=%d, failures=%d",
            metadata.version,
            metadata.is_valid,
            metadata.execution_attempts,
            len(metadata.failures),
        )

        # Format the metadata into a readable string
        lines = [
            "=== Cache Metadata ===",
            f"File: {trajectory_file}",
            "",
            "--- Basic Info ---",
            f"Version: {metadata.version}",
            f"Created: {metadata.created_at}",
            f"Last Executed: {metadata.last_executed_at or 'Never'}",
            "",
            "--- Execution Statistics ---",
            f"Total Execution Attempts: {metadata.execution_attempts}",
            f"Total Failures: {len(metadata.failures)}",
            "",
            "--- Validity Status ---",
            f"Is Valid: {metadata.is_valid}",
        ]

        if not metadata.is_valid:
            lines.append(f"Invalidation Reason: {metadata.invalidation_reason}")

        lines.append("")
        lines.append("--- Trajectory Info ---")
        lines.append(f"Total Steps: {len(cache_file.trajectory)}")
        lines.append(f"Parameters: {len(cache_file.cache_parameters)}")
        if cache_file.cache_parameters:
            lines.append(
                f"Parameter Names: {', '.join(cache_file.cache_parameters.keys())}"
            )

        if metadata.failures:
            lines.append("")
            lines.append("--- Failure History ---")
            for i, failure in enumerate(metadata.failures, 1):
                lines.append(f"Failure {i}:")
                lines.append(f"  Timestamp: {failure.timestamp}")
                lines.append(f"  Step Index: {failure.step_index}")
                lines.append(
                    f"  Failure Count at Step: {failure.failure_count_at_step}"
                )
                lines.append(f"  Error: {failure.error_message}")

        return "\n".join(lines)
