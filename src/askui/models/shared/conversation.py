"""Conversation class for managing speaker-based agent interactions."""

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from askui.model_providers.detection_provider import DetectionProvider
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import (
    MessageParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UsageParam,
)
from askui.models.shared.settings import ActSettings
from askui.models.shared.tools import ToolCollection
from askui.models.shared.truncation_strategies import (
    SimpleTruncationStrategyFactory,
    TruncationStrategy,
    TruncationStrategyFactory,
)
from askui.reporting import NULL_REPORTER, Reporter
from askui.speaker.speaker import SpeakerResult, Speakers

if TYPE_CHECKING:
    from askui.models.shared.conversation_callback import ConversationCallback
    from askui.utils.caching.cache_manager import CacheManager

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ConversationException(Exception):
    """Exception raised during conversation execution."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class Conversation:
    """Manages conversation state and delegates execution to speakers.

    The Conversation holds all model providers (`VlmProvider`, `ImageQAProvider`,
    `DetectionProvider`), message history, truncation strategy, token usage,
    and current speaker. It orchestrates the conversation by delegating each
    step to the appropriate speaker.

    Speakers access the model providers via the conversation instance
    (e.g., `conversation.vlm_provider`).

    Args:
        speakers: Collection of speakers to use
        vlm_provider: VLM provider for LLM API calls
        image_qa_provider: Image Q&A provider (optional)
        detection_provider: Detection provider (optional)
        reporter: Reporter for logging messages and actions
        cache_manager: Cache manager for recording/playback (optional)
        truncation_strategy_factory: Factory for creating truncation strategies
        callbacks: List of callbacks for conversation lifecycle hooks (optional)
    """

    def __init__(
        self,
        speakers: Speakers,
        vlm_provider: VlmProvider,
        image_qa_provider: ImageQAProvider | None = None,
        detection_provider: DetectionProvider | None = None,
        reporter: Reporter = NULL_REPORTER,
        cache_manager: "CacheManager | None" = None,
        truncation_strategy_factory: TruncationStrategyFactory | None = None,
        callbacks: "list[ConversationCallback] | None" = None,
    ) -> None:
        """Initialize conversation with speakers and model providers."""
        if not speakers:
            msg = "At least one speaker must be provided"
            raise ValueError(msg)

        # Speakers and current state
        self.speakers = speakers
        self.current_speaker = speakers[speakers.default_speaker]
        self.accumulated_usage = UsageParam()

        # Model providers - accessible by speakers via conversation instance
        self.vlm_provider = vlm_provider
        self.image_qa_provider = image_qa_provider
        self.detection_provider = detection_provider

        # Infrastructure
        self._reporter = reporter
        self.cache_manager = cache_manager
        self._truncation_strategy_factory = (
            truncation_strategy_factory or SimpleTruncationStrategyFactory()
        )
        self._truncation_strategy: TruncationStrategy | None = None
        self._callbacks: "list[ConversationCallback]" = callbacks or []

        # State for current execution (set in start())
        self.settings: ActSettings = ActSettings()
        self.tools: ToolCollection = ToolCollection()
        self._reporters: list[Reporter] = []
        self._step_index: int = 0

        # Cache execution context (for communication between tools and CacheExecutor)
        self.cache_execution_context: dict[str, Any] = {}

        # Track if cache execution was used (to prevent recording during playback)
        self._executed_from_cache: bool = False

    def _call_callbacks(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Call a method on all registered callbacks.

        Args:
            method_name: Name of the callback method to call
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
        """
        for callback in self._callbacks:
            method = getattr(callback, method_name, None)
            if method and callable(method):
                method(self, *args, **kwargs)

    @tracer.start_as_current_span("conversation")
    def execute_conversation(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        settings: ActSettings | None = None,
        reporters: list[Reporter] | None = None,
    ) -> None:
        """Setup conversation state and start control loop.

        Model providers are accessed via self.vlm_provider, etc.
        Speakers can access them via conversation.vlm_provider.

        Args:
            messages: Initial message history
            tools: Available tools
            settings: Agent settings
            reporters: Optional list of additional reporters for this conversation
        """
        msg = f"Starting conversation with speaker: {self.current_speaker.get_name()}"
        logger.info(msg)

        self._setup_control_loop(messages, tools, settings, reporters)

        self._call_callbacks("on_conversation_start")
        self._execute_control_loop()
        self._call_callbacks("on_conversation_end")

        self._conclude_control_loop()

    @tracer.start_as_current_span("setup_control_loop")
    def _setup_control_loop(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        settings: ActSettings | None = None,
        reporters: list[Reporter] | None = None,
    ) -> None:
        # Reset state
        self.accumulated_usage = UsageParam()
        self.cache_execution_context = {}
        self._executed_from_cache = False
        self.speakers.reset_state()

        # Store execution parameters
        self.settings = settings or ActSettings()
        self.tools = tools or ToolCollection()
        self._reporters = reporters or []

        # Initialize truncation strategy
        self._truncation_strategy = (
            self._truncation_strategy_factory.create_truncation_strategy(
                tools=self.tools.to_params(),
                system=self.settings.messages.system,
                messages=messages,
                model=self.vlm_provider.model_id,
            )
        )

    @tracer.start_as_current_span("control_loop")
    def _execute_control_loop(self) -> None:
        self._call_callbacks("on_control_loop_start")
        self._step_index = 0
        continue_execution = True
        while continue_execution:
            continue_execution = self._execute_step()
        self._call_callbacks("on_control_loop_end")

    @tracer.start_as_current_span("finish_control_loop")
    def _conclude_control_loop(self) -> None:
        # Finish recording if cache_manager is active and not executing from cache
        if self.cache_manager is not None and not self._executed_from_cache:
            self.cache_manager.finish_recording(self.get_messages())

        # Report final usage
        self._reporter.add_usage_summary(self.accumulated_usage.model_dump())

    @tracer.start_as_current_span("step")
    def _execute_step(self) -> bool:
        """Execute one step of the conversation loop with speakers.

        Each step includes:
        1. Infer next speaker
        2. Get message(s) from active speaker and add to history
        3. Execute tool calls if applicable and add result to history
        4. Check if conversation should continue and switch speaker if necessary
        5. Collect Statistics

        Returns:
            True if loop should continue, False if done
        """
        self._call_callbacks("on_step_start", self._step_index)

        # 1. Infer next speaker
        speaker = self.current_speaker
        if not speaker.can_handle(self):
            logger.debug(
                "Speaker %s cannot handle current state, switching to default",
                self.current_speaker.get_name(),
            )
            self.switch_speaker(self.speakers.default_speaker)
            speaker = self.speakers[self.current_speaker.get_name()]

        # 2. Get next message(s) from speaker and add to history
        logger.debug("Executing step with speaker: %s", speaker.get_name())
        result: SpeakerResult = speaker.handle_step(self, self.cache_manager)
        for message in result.messages_to_add:
            self._add_message(message)

        # 3. Execute tool calls if applicable
        continue_loop = False
        if result.messages_to_add:
            last_message = result.messages_to_add[-1]
            tool_result_message = self._execute_tools_if_present(last_message)
            if tool_result_message:
                self._add_message(tool_result_message)

                # Handle side effects of tool execution (e.g., speaker switches)
                self._handle_tool_results(last_message, tool_result_message)

                continue_loop = True  # we always continue after a tool was called

        # 4. Check if conversation should continue and switch speaker if necessary
        continue_loop = continue_loop or self._handle_result_status(result)

        # 5. Collect Statistics
        if result.usage:
            self._accumulate_usage(result.usage)

        self._call_callbacks("on_step_end", self._step_index, result)
        self._step_index += 1

        return continue_loop

    @tracer.start_as_current_span("execute_tool_call")
    def _execute_tools_if_present(self, message: MessageParam) -> MessageParam | None:
        """Execute tools if the message contains tool use blocks.

        Args:
            message: Message to check for tool calls

        Returns:
            MessageParam with tool results, or None if no tools to execute
        """
        # Only process assistant messages
        if message.role != "assistant":
            return None

        # Check if content is a list (could contain tool use blocks)
        if isinstance(message.content, str):
            return None

        # Find tool use blocks
        tool_use_blocks = [
            block for block in message.content if block.type == "tool_use"
        ]

        if not tool_use_blocks:
            return None

        # Execute tools
        tool_names = [block.name for block in tool_use_blocks]
        logger.debug("Executing %d tool(s)", len(tool_use_blocks))
        self._call_callbacks("on_tool_execution_start", tool_names)
        tool_results = self.tools.run(tool_use_blocks)
        self._call_callbacks("on_tool_execution_end", tool_names)

        if not tool_results:
            return None

        # Return tool results as a user message
        return MessageParam(content=tool_results, role="user")

    @tracer.start_as_current_span("handle_tool_result")
    def _handle_tool_results(
        self,
        assistant_message: MessageParam,
        tool_result_message: MessageParam,
    ) -> None:
        """Handle side effects of tool execution.

        Extracts tool use blocks and tool results from messages, then checks
        if specific tools require speaker switches or other actions.

        Currently handles:
        - ExecuteCachedTrajectory: Switches to CacheExecutor if successful

        Args:
            assistant_message: The assistant message containing tool use blocks
            tool_result_message: The user message containing tool results
        """
        # Extract tool use blocks from assistant message
        if isinstance(assistant_message.content, str):
            return

        tool_use_blocks: list[ToolUseBlockParam] = [
            block for block in assistant_message.content if block.type == "tool_use"
        ]

        if isinstance(tool_result_message.content, str):
            return

        tool_results: list[ToolResultBlockParam] = tool_result_message.content  # type: ignore[assignment]

        # Handle side effects for each tool
        for tool_use_block, tool_result in zip(
            tool_use_blocks, tool_results, strict=False
        ):
            # Check if ExecuteCachedTrajectory was called successfully
            if (
                tool_use_block.name.startswith("execute_cached_executions_tool")
                and not tool_result.is_error
            ):
                # Extract parameters from tool call (input is dict at runtime)
                trajectory_file: str = tool_use_block.input["trajectory_file"]  # type: ignore[index]
                start_from_step_index: int = tool_use_block.input.get(  # type: ignore[attr-defined]
                    "start_from_step_index", 0
                )
                parameter_values: dict[str, str] = tool_use_block.input.get(  # type: ignore[attr-defined]
                    "parameter_values", {}
                )

                # Prepare cache execution context for CacheExecutor
                # CacheExecutor will validate and load the cache file
                self.cache_execution_context = {
                    "trajectory_file": trajectory_file,
                    "start_from_step_index": start_from_step_index,
                    "parameter_values": parameter_values,
                    "toolbox": self.tools,
                    "reporter": self._reporter,
                }
                self._executed_from_cache = True
                self.switch_speaker("CacheExecutor")

    def _add_message(self, message: MessageParam) -> None:
        """Add message to conversation history.

        Args:
            message: Message to add
        """
        if not self._truncation_strategy:
            logger.error("No truncation strategy, cannot add message")
            return

        # Add to truncation strategy
        self._truncation_strategy.append_message(message)

        # Report to reporter
        self._reporter.add_message(
            self.current_speaker.get_name(), message.model_dump(mode="json")
        )

    @tracer.start_as_current_span("handle_result_status")
    def _handle_result_status(self, result: SpeakerResult) -> bool:
        """Handle speaker result status and determine if loop should continue.

        Args:
            result: Result from speaker

        Returns:
            True if loop should continue, False if done
        """
        if result.status == "done":
            logger.info("Conversation completed successfully")
            return False
        if result.status == "failed":
            logger.error("Conversation failed")
            return False
        if result.status == "switch_speaker":
            if result.next_speaker:
                self.switch_speaker(result.next_speaker)
            return True
        # status == "continue"
        return True

    @tracer.start_as_current_span("switch_speaker")
    def switch_speaker(self, speaker_name: str) -> None:
        """Switch to a different speaker.

        Args:
            speaker_name: Name of the speaker to switch to
        """
        old_speaker = self.current_speaker
        self.current_speaker = self.speakers[speaker_name]
        logger.info(
            "Switched speaker: %s => %s",
            old_speaker.get_name(),
            self.current_speaker.get_name(),
        )

    def get_messages(self) -> list[MessageParam]:
        """Get current message history from truncation strategy.

        Returns:
            List of messages in current conversation
        """
        return self._truncation_strategy.messages if self._truncation_strategy else []

    def get_truncation_strategy(self) -> TruncationStrategy | None:
        """Get current truncation strategy.

        Returns:
            Current truncation strategy or None if not initialized
        """
        return self._truncation_strategy

    def _accumulate_usage(self, step_usage: UsageParam) -> None:
        """Accumulate token usage statistics.

        Args:
            step_usage: Usage from a single step
        """
        self.accumulated_usage.input_tokens = (
            self.accumulated_usage.input_tokens or 0
        ) + (step_usage.input_tokens or 0)
        self.accumulated_usage.output_tokens = (
            self.accumulated_usage.output_tokens or 0
        ) + (step_usage.output_tokens or 0)
        self.accumulated_usage.cache_creation_input_tokens = (
            self.accumulated_usage.cache_creation_input_tokens or 0
        ) + (step_usage.cache_creation_input_tokens or 0)
        self.accumulated_usage.cache_read_input_tokens = (
            self.accumulated_usage.cache_read_input_tokens or 0
        ) + (step_usage.cache_read_input_tokens or 0)

        current_span = trace.get_current_span()
        current_span.set_attributes(
            {
                "input_tokens": step_usage.input_tokens or 0,
                "output_tokens": step_usage.output_tokens or 0,
                "cache_creation_input_tokens": (
                    step_usage.cache_creation_input_tokens or 0
                ),
                "cache_read_input_tokens": step_usage.cache_read_input_tokens or 0,
            }
        )
