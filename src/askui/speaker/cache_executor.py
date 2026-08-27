"""Cache Executor speaker for executing cached trajectories."""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import BaseModel, Field
from typing_extensions import Literal, override

from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.settings import CacheExecutionSettings
from askui.utils.caching.cache_manager import CacheManager
from askui.utils.caching.cache_parameter_handler import CacheParameterHandler
from askui.utils.caching.reporting_utils import report_cache_event
from askui.utils.visual_validation import (
    compute_ahash,
    compute_hamming_distance,
    compute_phash,
    extract_region,
    find_recent_screenshot,
    get_validation_coordinate,
)

from .speaker import Speaker, SpeakerResult

if TYPE_CHECKING:
    from askui.models.shared.conversation import Conversation
    from askui.models.shared.settings import CacheFile
    from askui.models.shared.tools import ToolCollection
    from askui.reporting import Reporter

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    """Result of executing a single step in a trajectory.

    Attributes:
        status: Execution status (SUCCESS, FAILED, NEEDS_AGENT, COMPLETED)
        step_index: Index of the step that was executed
        tool_result: The tool result or tool use block for reference
        error_message: Error message if execution failed
        screenshots_taken: List of screenshots captured during this step
        message_history: List of MessageParam representing the conversation history
    """

    status: Literal["SUCCESS", "FAILED", "NEEDS_AGENT", "COMPLETED"]
    step_index: int
    tool_result: Any | None = None
    error_message: str | None = None
    screenshots_taken: list[Any] = Field(default_factory=list)
    message_history: list[MessageParam] = Field(default_factory=list)


class CacheExecutor(Speaker):
    """Speaker that handles cached trajectory playback.

    This speaker generates messages from a cached trajectory:
    1. Get next step from cached trajectory (as tool use message)
    2. Track progress through the trajectory
    3. Pause for non-cacheable tools (switch to agent)
    4. Handle completion (switch to agent for verification)
    5. Handle failures (update metadata, switch to agent)

    Tool execution is handled by the Conversation class, not by this speaker.
    """

    _DEFAULT_DESCRIPTION: str = (
        "Replays a pre-recorded UI interaction trajectory from a cache "
        "file. Use this speaker to fast-forward through previously "
        "recorded action sequences instead of executing each step from "
        "scratch.\n"
        "Expected context keys:\n"
        "  - trajectory_file (str, required): Full path to the "
        "trajectory file\n"
        "  - start_from_step_index (int, optional, default=0): Step "
        "index to start from\n"
        "  - parameter_values (dict[str, str], optional, default={}): "
        "Dynamic parameter values for the trajectory"
    )

    def __init__(
        self,
        execution_settings: CacheExecutionSettings | None = None,
        *,
        name: str = "CacheExecutor",
        description: str = _DEFAULT_DESCRIPTION,
    ) -> None:
        """Initialize Cache Executor speaker.

        Args:
            execution_settings: Settings for cache execution including delay time,
                visual validation threshold, etc. If None, default settings are used.
            name: Speaker name for identification.
            description: Human-readable description of this speaker.
        """
        super().__init__(name=name, description=description)
        _settings = execution_settings or CacheExecutionSettings()

        # Cache execution state
        self._executing_from_cache: bool = False
        self._cache_verification_pending: bool = False
        self._cache_file: "CacheFile | None" = None
        self._cache_file_path: str | None = None

        # Cache Execution Settings
        self._skip_visual_validation: bool = _settings.skip_visual_validation
        self._visual_validation_threshold: int = _settings.visual_validation_threshold
        self._delay_time_between_actions: float = _settings.delay_time_between_actions

        self._trajectory: list[ToolUseBlockParam] = []
        self._toolbox: "ToolCollection | None" = None
        self._parameter_values: dict[str, str] = {}
        self._visual_validation_enabled: bool = False
        self._visual_validation_method: str = "phash"
        self._visual_validation_region_size: int = 100

        self._current_step_index: int = 0
        self._message_history: list[MessageParam] = []

        # Activation context received via on_activate()
        self._activation_context: dict[str, Any] = {}

        # Reporter for surfacing replay progress (set on activation).
        self._reporter: "Reporter | None" = None

    @property
    def current_cache_file(self) -> "CacheFile | None":
        """The cache file of the most recently activated trajectory, if any."""
        return self._cache_file

    @property
    def current_cache_file_path(self) -> str | None:
        """Path of the most recently activated trajectory, if any."""
        return self._cache_file_path

    @override
    def can_handle(self, conversation: "Conversation") -> bool:  # noqa: ARG002
        """Check if cache execution is active or should be activated.

        Args:
            conversation: The conversation instance.

        Returns:
            True if in cache execution mode or if activation context is set.
        """
        can_handle_step = self._executing_from_cache or bool(self._activation_context)
        if not can_handle_step:
            logger.warning("CacheExecutor can't handle the next step")
        return can_handle_step

    @override
    def handle_step(
        self, conversation: "Conversation", cache_manager: "CacheManager | None"
    ) -> SpeakerResult:
        """Get next cached step message.

        This speaker only generates messages (tool use blocks from cache).
        Tool execution is handled by the Conversation class.

        Args:
            conversation: The conversation instance with current state
            cache_manager: Cache manager for recording/playback

        Returns:
            SpeakerResult with the next cached tool use message
        """
        if cache_manager is None:
            error_msg = "CacheManager must be provided if executing from Cache"
            raise RuntimeError(error_msg)

        # Check if we need to activate cache execution from internal context
        if not self._executing_from_cache and self._activation_context:
            try:
                # Augment context with toolbox and reporter from conversation
                activation_context = {
                    **self._activation_context,
                    "toolbox": conversation.tools,
                    "reporter": conversation._reporter,  # noqa: SLF001
                }
                self._activate_from_context(activation_context, cache_manager)
                self._activation_context = {}
                conversation._executed_from_cache = True  # noqa: SLF001
            except Exception as e:
                # Validation or loading failed - report error and switch back
                logger.exception("Failed to activate cache execution")
                self._activation_context = {}

                error_message = MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(
                            type="text",
                            text=f"Cache execution failed: {e}",
                        )
                    ],
                )

                return SpeakerResult(
                    status="switch_speaker",
                    next_speaker="AgentSpeaker",
                    messages_to_add=[error_message],
                )

        messages = conversation.get_messages()

        # Check if last message was a tool result - if so, move to next step
        if messages and messages[-1].role == "user":
            # Last message is tool result, check if it's from our current step
            if self._message_history and len(self._message_history) > 0:
                # Tool was executed, move to next step
                self._current_step_index += 1
                # Add delay between actions
                if self._current_step_index < len(self._trajectory):
                    time.sleep(self._delay_time_between_actions)

        # Require a toolbox to execute. An empty trajectory is allowed: it flows
        # into `_get_next_step()`'s COMPLETED path (which requests verification)
        # rather than silently bouncing back to the agent.
        if self._toolbox is None:
            logger.error("Cache executor called but no toolbox available")
            return SpeakerResult(
                status="switch_speaker",
                next_speaker="AgentSpeaker",
            )

        # Get next step from cache (doesn't execute, just prepares the message)
        logger.debug("Getting next step from cache")
        result: ExecutionResult = self._get_next_step(conversation_messages=messages)

        # Handle result based on status
        return self._handle_result(result, cache_manager)

    @override
    def on_activate(self, context: dict[str, Any]) -> None:
        """Store activation context for use in the next `handle_step()` call.

        Args:
            context: Dict with trajectory_file, start_from_step_index,
                parameter_values.
        """
        self._activation_context = context

    def _handle_result(
        self, result: ExecutionResult, cache_manager: "CacheManager"
    ) -> SpeakerResult:
        """Handle execution result and return appropriate SpeakerResult."""
        if result.status == "SUCCESS":
            return self._handle_success(result)
        if result.status == "NEEDS_AGENT":
            return self._handle_needs_agent(result)
        if result.status == "COMPLETED":
            return self._handle_completed(result)
        # FAILED
        return self._handle_failed(cache_manager, result)

    def _handle_success(self, result: ExecutionResult) -> SpeakerResult:
        """Handle successful preparation of next cache step."""
        if not result.message_history:
            return SpeakerResult(
                status="switch_speaker",
                next_speaker="AgentSpeaker",
            )

        # Get assistant message (tool use)
        assistant_msg = result.message_history[-1]

        # Store this message for tracking
        self._message_history.append(assistant_msg)

        # Continue with cache execution
        return SpeakerResult(
            status="continue",
            messages_to_add=[assistant_msg],
        )

    def _handle_needs_agent(self, result: ExecutionResult) -> SpeakerResult:
        """Handle cache execution pausing for non-cacheable tool."""
        tool_name = getattr(result.tool_result, "name", "unknown")
        report_cache_event(
            self._reporter,
            f"Paused replay at step {result.step_index}: the '{tool_name}' tool "
            "cannot be replayed from cache; the agent will perform this step.",
            log=logger,
        )
        self._executing_from_cache = False

        tool_to_execute = result.tool_result

        if tool_to_execute:
            resume_index = result.step_index + 1
            more_steps_remain = self._has_executable_steps_from(resume_index)

            if more_steps_remain:
                resume_instruction = (
                    "Execute this tool with the necessary parameters. To replay the "
                    "remaining cached steps afterwards, switch back to the "
                    "CacheExecutor with "
                    f"start_from_step_index={resume_index}."
                )
            else:
                resume_instruction = (
                    "This is the FINAL step of the trajectory. Execute this tool "
                    "with the necessary parameters, then verify the outcome with the "
                    "verify_cache_execution tool. Do NOT switch back to the "
                    "CacheExecutor - there are no further cached steps to replay."
                )

            instruction_message = MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text",
                        text=(
                            f"Cache execution paused at step {result.step_index}. "
                            "The previous steps were executed successfully "
                            f"from cache. The next step requires the "
                            f"'{tool_to_execute.name}' tool, which cannot be "
                            f"executed from cache. {resume_instruction}"
                        ),
                    )
                ],
            )

            return SpeakerResult(
                status="switch_speaker",
                next_speaker="AgentSpeaker",
                messages_to_add=[instruction_message],
            )

        return SpeakerResult(
            status="switch_speaker",
            next_speaker="AgentSpeaker",
        )

    def _handle_completed(
        self,
        result: ExecutionResult,  # noqa: ARG002
    ) -> SpeakerResult:
        """Handle cache execution completion."""
        report_cache_event(
            self._reporter,
            f"Finished replaying {len(self._trajectory)} cached step(s); "
            "asking the agent to verify the result.",
            log=logger,
        )
        self._executing_from_cache = False
        self._cache_verification_pending = True

        verification_request = MessageParam(
            role="user",
            content=[
                TextBlockParam(
                    type="text",
                    text=(
                        "[CACHE EXECUTION COMPLETED]\n\n"
                        "The CacheExecutor has automatically executed"
                        f" {len(self._trajectory)} steps from the cached trajectory"
                        f" '{self._cache_file_path}'. All previous tool calls in this"
                        " conversation were replayed from cache, not performed by the"
                        " agent.\n\n Please verify if the cached execution correctly"
                        " achieved the target system state using the"
                        " verify_cache_execution tool."
                    ),
                )
            ],
        )

        return SpeakerResult(
            status="switch_speaker",
            next_speaker="AgentSpeaker",
            messages_to_add=[verification_request],
        )

    def _handle_failed(
        self, cache_manager: CacheManager, result: ExecutionResult
    ) -> SpeakerResult:
        """Handle cache execution failure."""
        report_cache_event(
            self._reporter,
            f"Cache replay failed at step {result.step_index}: "
            f"{result.error_message or 'unknown error'}. "
            "The agent will complete the task manually.",
            log=logger,
            level=logging.ERROR,
        )
        self._executing_from_cache = False

        # Update cache metadata
        if self._cache_file and self._cache_file_path:
            cache_manager.update_metadata_on_failure(
                cache_file=self._cache_file,
                cache_file_path=self._cache_file_path,
                step_index=result.step_index,
                error_message=result.error_message or "Unknown error",
            )

        # Add failure message to inform the agent about what happened
        failure_message = MessageParam(
            role="user",
            content=[
                TextBlockParam(
                    type="text",
                    text=(
                        "[CACHE EXECUTION FAILED]\n\n"
                        f"The CacheExecutor failed to execute the cached trajectory "
                        f"'{self._cache_file_path}' at step {result.step_index}.\n\n"
                        f"Error: {result.error_message}\n\n"
                        "The cache file is potentially invalid. "
                        "Please complete the remaining steps manually. After that, use "
                        "the verify_cache_execution tool with success=False to "
                        "potentially invalidate the cache file."
                    ),
                )
            ],
        )

        return SpeakerResult(
            status="switch_speaker",
            next_speaker="AgentSpeaker",
            messages_to_add=[failure_message],
        )

    def _activate_from_context(
        self, context: dict[str, Any], cache_manager: "CacheManager"
    ) -> None:
        """Activate cache execution from conversation context.

        Args:
            context: Dict containing cache execution parameters
            cache_manager: Cache manager for loading cache files

        Raises:
            FileNotFoundError: If cache file doesn't exist
            ValueError: If validation fails
        """
        # Extract parameters
        trajectory_file: str = context["trajectory_file"]
        start_from_step_index: int = context.get("start_from_step_index", 0)
        parameter_values: dict[str, Any] = context.get("parameter_values", {})
        toolbox: "ToolCollection" = context["toolbox"]

        logger.debug("Activating cache execution from: %s", trajectory_file)

        # Load and validate cache file
        if not self._cache_file_path or self._cache_file_path != trajectory_file:
            self._cache_file_path = trajectory_file
            self._cache_file = cache_manager.read_cache_file(Path(trajectory_file))
        else:
            if not self._cache_file:
                self._cache_file = cache_manager.read_cache_file(Path(trajectory_file))

        # Validate step index. `start_from_step_index == len(trajectory)` is
        # allowed and means "there is nothing left to replay" - this happens when
        # the agent resumes after handling the trajectory's last step (e.g. a
        # non-cacheable final step). It is treated as an immediate completion by
        # `_get_next_step()` instead of being rejected as out of range.
        trajectory_len = len(self._cache_file.trajectory)
        if start_from_step_index < 0 or start_from_step_index > trajectory_len:
            valid_range = (
                f"0-{trajectory_len}" if trajectory_len > 0 else "0 (empty trajectory)"
            )
            error_msg = (
                f"Invalid start_from_step_index: {start_from_step_index}. "
                f"Trajectory has {trajectory_len} steps "
                f"(valid indices: {valid_range})."
            )
            raise ValueError(error_msg)

        # Validate parameters
        is_valid, missing = CacheParameterHandler.validate_parameters(
            self._cache_file.trajectory, parameter_values
        )
        if not is_valid:
            error_msg = (
                f"Missing required parameter values: {', '.join(missing)}. "
                f"The trajectory contains the following parameters: "
                f"{', '.join(self._cache_file.cache_parameters.keys())}. "
                "Please provide values for all parameters."
            )
            raise ValueError(error_msg)

        # Warn if cache is invalid
        if not self._cache_file.metadata.is_valid:
            logger.warning(
                "Using invalid cache from %s. Reason: %s. "
                "This cache may not work correctly.",
                Path(trajectory_file).name,
                self._cache_file.metadata.invalidation_reason,
            )

        # Set up execution state
        self._trajectory = self._cache_file.trajectory
        self._toolbox = toolbox
        self._parameter_values = parameter_values
        self._current_step_index = start_from_step_index
        self._message_history = []
        self._executing_from_cache = True

        # Configure visual validation
        visual_validation_config = self._cache_file.metadata.visual_validation

        if self._skip_visual_validation:
            self._visual_validation_enabled = False
            logger.info("Visual validation disabled by execution settings")
        elif visual_validation_config and visual_validation_config.enabled:
            self._visual_validation_enabled = True
            self._visual_validation_method = visual_validation_config.method
            self._visual_validation_region_size = visual_validation_config.region_size
            logger.info(
                "Visual validation enabled (method=%s, threshold=%d)",
                self._visual_validation_method,
                self._visual_validation_threshold,
            )
        else:
            self._visual_validation_enabled = False
            logger.debug("Visual validation disabled or not configured")

        # Reporter for surfacing replay progress/outcomes to the user.
        reporter: Reporter | None = context.get("reporter")
        self._reporter = reporter

        step_count = len(self._cache_file.trajectory)
        from_suffix = (
            "" if start_from_step_index == 0 else f" from step {start_from_step_index}"
        )
        report_cache_event(
            self._reporter,
            f"Replaying cached trajectory '{Path(trajectory_file).name}' "
            f"({step_count} steps){from_suffix}.",
            log=logger,
        )

        # Report cache execution statistics to the reporter
        if reporter and self._cache_file.metadata.token_usage:
            reporter.add_cache_execution_statistics(
                self._cache_file.metadata.token_usage.model_dump()
            )

    def reset_state(self) -> None:
        """Reset cache execution state."""
        self._executing_from_cache = False
        self._cache_verification_pending = False
        self._cache_file = None
        self._cache_file_path = None
        self._trajectory = []
        self._toolbox = None
        self._parameter_values = {}
        self._current_step_index = 0
        self._message_history = []
        self._activation_context = {}
        self._reporter = None

    def _get_next_step(
        self, conversation_messages: list[MessageParam] | None = None
    ) -> ExecutionResult:
        """Get the next step message from the trajectory.

        This method does NOT execute tools - it only prepares the message.
        Tool execution is handled by the Conversation class.

        Args:
            conversation_messages: Optional conversation messages for visual validation

        Returns:
            ExecutionResult with status and the prepared message
        """
        # Check if we've completed all steps
        if self._current_step_index >= len(self._trajectory):
            return ExecutionResult(
                status="COMPLETED",
                step_index=max(self._current_step_index - 1, 0),
                message_history=self._message_history,
            )

        step = self._trajectory[self._current_step_index]
        step_index = self._current_step_index

        # Check if step should be skipped
        if self._should_skip_step(step):
            logger.debug("Skipping step %d: %s", step_index, step.name)
            self._current_step_index += 1
            return self._get_next_step(conversation_messages=conversation_messages)

        # Check if step needs agent intervention (non-cacheable)
        if self._should_pause_for_agent(step):
            logger.info(
                "Pausing at step %d: %s (non-cacheable tool)",
                step_index,
                step.name,
            )
            return ExecutionResult(
                status="NEEDS_AGENT",
                step_index=step_index,
                message_history=self._message_history.copy(),
                tool_result=step,
            )

        # Visual validation
        if self._visual_validation_enabled:
            current_screenshot = None
            if conversation_messages:
                current_screenshot = find_recent_screenshot(conversation_messages)

            is_valid, error_msg = self._validate_step_visually(
                step, current_screenshot=current_screenshot
            )
            if not is_valid:
                return ExecutionResult(
                    status="FAILED",
                    step_index=step_index,
                    error_message=error_msg,
                    message_history=self._message_history.copy(),
                )

        # Substitute parameters
        substituted_step = CacheParameterHandler.substitute_parameters(
            step, self._parameter_values
        )

        # Create assistant message (tool use) - DON'T execute yet
        logger.debug("Preparing step %d: %s", step_index, step.name)

        assistant_message = MessageParam(
            role="assistant",
            content=[substituted_step],
        )

        return ExecutionResult(
            status="SUCCESS",
            step_index=step_index,
            tool_result=None,
            message_history=[assistant_message],
        )

    def _has_executable_steps_from(self, index: int) -> bool:
        """Return whether any step at or after `index` would still be replayed.

        Skippable steps (e.g. `switch_speaker`, verbosity tools) are ignored.
        Non-cacheable steps count as executable because resuming would replay up
        to them and pause again. Used to decide whether the agent should resume
        cache execution after handling a non-cacheable step, or whether that step
        was the trajectory's last and no resume is needed.
        """
        return any(
            not self._should_skip_step(self._trajectory[i])
            for i in range(index, len(self._trajectory))
        )

    def _should_pause_for_agent(self, step: ToolUseBlockParam) -> bool:
        """Check if execution should pause for agent intervention."""
        if not self._toolbox:
            return False

        # Try exact match first, then prefix match (for tools with UUID suffixes)
        tool = self._toolbox.tool_map.get(step.name)
        if tool is None:
            tool = self._toolbox.find_tool_by_prefix(step.name)

        if tool is None:
            # Tool not found - should pause for agent to handle
            return True

        return not tool.is_cacheable

    def _should_skip_step(self, step: ToolUseBlockParam) -> bool:
        """Check if a step should be skipped during execution."""
        # Use startswith() to handle tool names with UUID suffixes
        tools_to_skip: list[str] = [
            "retrieve_available_trajectories_tool",
            "switch_speaker",
            "execute_cached_executions_tool",  # backward compat for old caches
            "print_to_console",  # verbosity is not needed during cached executions
        ]
        return any(step.name.startswith(prefix) for prefix in tools_to_skip)

    def _validate_step_visually(
        self, step: ToolUseBlockParam, current_screenshot: Image.Image | None = None
    ) -> tuple[bool, str | None]:
        """Validate cached step using visual hash comparison."""
        if not self._visual_validation_enabled:
            return True, None

        if not step.visual_representation:
            return True, None

        if current_screenshot is None:
            current_screenshot = find_recent_screenshot(self._message_history)
            if not current_screenshot:
                logger.warning("No screenshot found for visual validation, skipping")
                return True, None

        try:
            # Extract coordinate using the same logic as cache_manager
            tool_input: dict[str, Any] = (
                step.input if isinstance(step.input, dict) else {}
            )
            coordinate = get_validation_coordinate(tool_input)

            if coordinate is None:
                # No coordinate found - skip visual validation for this step
                logger.info(
                    "No coordinate found in step input, skipping visual validation"
                )
                return True, None

            # Pass coordinate in the format extract_region expects
            region = extract_region(
                current_screenshot,
                {"coordinate": list(coordinate)},
                region_size=self._visual_validation_region_size,
            )

            if self._visual_validation_method == "phash":
                current_hash = compute_phash(region, hash_size=8)
            else:
                current_hash = compute_ahash(region, hash_size=8)

            distance = compute_hamming_distance(
                step.visual_representation, current_hash
            )

            if distance > self._visual_validation_threshold:
                error_msg = (
                    f"Visual validation failed: UI has changed significantly. "
                    f"Hamming distance: {distance} > threshold: "
                    f"{self._visual_validation_threshold}."
                )
                return False, error_msg

            logger.debug(
                "Visual validation passed (distance=%d, threshold=%d)",
                distance,
                self._visual_validation_threshold,
            )
            return True, None  # noqa: TRY300

        except Exception as e:
            logger.exception("Failed to perform visual validation")
            return True, f"Visual validation skipped due to error: {e}"
