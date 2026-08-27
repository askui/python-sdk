import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import validate_call
from typing_extensions import override

from ..models.shared.tools import Tool
from ..utils.caching.cache_manager import CacheManager
from ..utils.caching.reporting_utils import report_cache_event

if TYPE_CHECKING:
    from ..reporting import Reporter
    from ..speaker.cache_executor import CacheExecutor

logger = logging.getLogger(__name__)


class VerifyCacheExecution(Tool):
    """Tool for the agent to report cache execution verification results.

    When wired with the active `CacheExecutor` and `CacheManager`, this tool
    also persists the outcome to the trajectory's metadata: a successful
    verification records the execution attempt, while an unsuccessful one
    additionally invalidates the cache so it is not reused.

    Args:
        cache_executor: The active `CacheExecutor`, used to resolve which
            trajectory was replayed. If `None`, the tool only reports the result.
        cache_manager: The active `CacheManager`, used to persist metadata. If
            `None`, the tool only reports the result.
        reporter: Optional reporter used to surface the verification outcome
            (and its reason) to the user in addition to the logs.
    """

    def __init__(
        self,
        cache_executor: "CacheExecutor | None" = None,
        cache_manager: "CacheManager | None" = None,
        reporter: "Reporter | None" = None,
    ) -> None:
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
                "- The final state is incorrect or incomplete\n\n"
                "Reporting success=False invalidates the cache so it is not "
                "reused until it is re-recorded."
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
        self._cache_executor = cache_executor
        self._cache_manager = cache_manager
        self._reporter = reporter
        self.is_cacheable = False  # Verification is not cacheable

    @override
    @validate_call
    def __call__(self, success: bool, verification_notes: str) -> str:
        """Record cache verification result and persist it to metadata.

        Args:
            success: Whether cache execution achieved target state
            verification_notes: Explanation of verification result

        Returns:
            Confirmation message
        """
        cache_name = self._current_cache_name()
        if success:
            report_cache_event(
                self._reporter,
                f"Cache verification PASSED for {cache_name}: {verification_notes}",
                log=logger,
                level=logging.INFO,
            )
        else:
            report_cache_event(
                self._reporter,
                f"Cache verification FAILED for {cache_name} - the replay did not "
                f"achieve the expected result. Agent's reason: {verification_notes}. "
                "The cache will be invalidated so it is not reused.",
                log=logger,
                level=logging.WARNING,
            )

        self._persist_verification(success, verification_notes)
        return (
            f"Cache verification reported: success={success}, "
            f"notes={verification_notes}"
        )

    def _current_cache_name(self) -> str:
        """Human-readable name of the trajectory being verified."""
        if self._cache_executor is not None:
            path = self._cache_executor.current_cache_file_path
            if path:
                return f"'{Path(path).name}'"
        return "the cached trajectory"

    def _persist_verification(self, success: bool, verification_notes: str) -> None:
        """Persist the verification outcome to the trajectory metadata, if wired."""
        if self._cache_executor is None or self._cache_manager is None:
            return

        cache_file = self._cache_executor.current_cache_file
        cache_file_path = self._cache_executor.current_cache_file_path
        if cache_file is None or cache_file_path is None:
            logger.debug("No active cache execution to persist verification result for")
            return

        if success:
            self._cache_manager.update_metadata_on_completion(
                cache_file=cache_file,
                cache_file_path=cache_file_path,
                success=True,
            )
        else:
            reason = (
                f"Agent reported unsuccessful cache execution: {verification_notes}"
            )
            self._cache_manager.mark_execution_unsuccessful(
                cache_file=cache_file,
                cache_file_path=cache_file_path,
                reason=reason,
            )


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
                        "description": ("Full path to the trajectory file to inspect."),
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
            error_msg = f"Trajectory file not found: {trajectory_file}"
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
