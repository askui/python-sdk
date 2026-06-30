"""Conversation class for managing speaker-based agent interactions."""

import logging
import uuid
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from askui.model_providers.detection_provider import DetectionProvider
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import MessageParam
from askui.models.shared.secrets import SecretVault
from askui.models.shared.settings import ActSettings
from askui.models.shared.tools import ToolCollection
from askui.models.shared.truncation_strategies import (
    SummarizingTruncationStrategy,
    TruncationStrategy,
)
from askui.reporting import NULL_REPORTER, Reporter
from askui.speaker.speaker import SpeakerResult, Speakers
from askui.tools.switch_speaker_tool import SwitchSpeakerTool

if TYPE_CHECKING:
    from askui.callbacks import ConversationCallback
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
        truncation_strategy: truncation strategies (optional)
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
        truncation_strategy: TruncationStrategy | None = None,
        callbacks: "list[ConversationCallback] | None" = None,
    ) -> None:
        """Initialize conversation with speakers and model providers."""
        if not speakers:
            msg = "At least one speaker must be provided"
            raise ValueError(msg)

        # Identity
        self.conversation_id: str = str(uuid.uuid4())

        # Speakers and current state
        self.speakers = speakers
        self.current_speaker = speakers[speakers.default_speaker]

        # Model providers - accessible by speakers via conversation instance
        self.vlm_provider = vlm_provider
        self.image_qa_provider = image_qa_provider
        self.detection_provider = detection_provider

        # Infrastructure
        self._reporter = reporter
        self.cache_manager = cache_manager
        self._callbacks: "list[ConversationCallback]" = callbacks or []

        self.secret_vault: SecretVault = SecretVault()

        # State for current execution (set in start())
        self.settings: ActSettings = ActSettings()
        self.tools: ToolCollection = ToolCollection()
        self._reporters: list[Reporter] = []
        self._step_index: int = 0

        # Truncation strategy. Conversation-owned dependencies are
        # auto-injected so users can pass a custom strategy with only
        # strategy-specific config (e.g. n_messages_to_keep) without
        # needing access to vlm_provider/reporter/callbacks/conversation
        # at construction time. ``vlm_provider`` is only injected when
        # not pre-set, allowing callers to override the summarization
        # VLM (e.g. with a cheaper model).
        self._truncation_strategy: TruncationStrategy = (
            truncation_strategy or SummarizingTruncationStrategy()
        )
        if self._truncation_strategy.vlm_provider is None:
            self._truncation_strategy.vlm_provider = vlm_provider
        self._truncation_strategy.reporter = reporter
        self._truncation_strategy.callbacks = self._callbacks
        self._truncation_strategy.conversation = self

        # Track if cache execution was used (to prevent recording during playback)
        self._executed_from_cache: bool = False

    def _on_conversation_start(self) -> None:
        for callback in self._callbacks:
            callback.on_conversation_start(self)

    def _on_conversation_end(self) -> None:
        for callback in self._callbacks:
            callback.on_conversation_end(self)

    def _on_control_loop_start(self) -> None:
        for callback in self._callbacks:
            callback.on_control_loop_start(self)

    def _on_control_loop_end(self) -> None:
        for callback in self._callbacks:
            callback.on_control_loop_end(self)

    def _on_step_start(self, step_index: int) -> None:
        for callback in self._callbacks:
            callback.on_step_start(self, step_index)

    def _on_step_end(self, step_index: int, result: SpeakerResult) -> None:
        for callback in self._callbacks:
            callback.on_step_end(self, step_index, result)

    def _on_speaker_switch(self, from_speaker: str, to_speaker: str) -> None:
        for callback in self._callbacks:
            callback.on_speaker_switch(self, from_speaker, to_speaker)

    def _on_tool_execution_start(self, tool_names: list[str]) -> None:
        for callback in self._callbacks:
            callback.on_tool_execution_start(self, tool_names)

    def _on_tool_execution_end(self, tool_names: list[str]) -> None:
        for callback in self._callbacks:
            callback.on_tool_execution_end(self, tool_names)

    @tracer.start_as_current_span("execute_conversation")
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
        msg = f"Starting conversation with speaker: {self.current_speaker.name}"
        logger.info(msg)

        self._setup_control_loop(messages, tools, settings, reporters)

        self._on_conversation_start()
        try:
            self._execute_control_loop()
        finally:
            self._on_conversation_end()
            self._teardown_control_loop()

    @tracer.start_as_current_span("_setup_control_loop")
    def _setup_control_loop(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        settings: ActSettings | None = None,
        reporters: list[Reporter] | None = None,
    ) -> None:
        # Reset state
        self._truncation_strategy.reset(messages)
        self._executed_from_cache = False
        self.speakers.reset_state()

        # Store execution parameters. Deep-copy settings so per-call mutations
        # (speaker handoff + secret sections appended to the system prompt) do not
        # accumulate on the Agent's persistent, reused settings object across calls.
        self.settings = (settings or ActSettings()).model_copy(deep=True)
        self.tools = tools or ToolCollection()
        self.tools.secret_vault = self.secret_vault
        self._reporters = reporters or []

        # Auto-populate speaker descriptions and switch_speaker tool
        self._setup_speaker_handoff()
        # Advertise available secret placeholders to the model
        self._setup_secrets()

    @tracer.start_as_current_span("_execute_control_loop")
    def _execute_control_loop(self) -> None:
        self._on_control_loop_start()
        self._step_index = 0
        continue_execution = True
        while continue_execution:
            continue_execution = self._execute_step()
            if self._is_max_steps_reached():
                continue_execution = False
        self._on_control_loop_end()

    def _is_max_steps_reached(self) -> bool:
        if self.settings.max_steps is None:
            return False
        if self._step_index >= self.settings.max_steps:
            msg = (
                f"Reached max_steps limit {self.settings.max_steps}, stopping execution"
            )
            logger.exception(msg)
            return True
        return False

    @tracer.start_as_current_span("_teardown_control_loop")
    def _teardown_control_loop(self) -> None:
        # Finish recording if cache_manager is active and not executing from cache
        if self.cache_manager is not None and not self._executed_from_cache:
            self.cache_manager.finish_recording(self.get_messages())

    def _setup_speaker_handoff(self) -> None:
        """Set up speaker handoff infrastructure.

        If there are speakers with descriptions (handoff targets), this method:
        1. Appends an ``<AVAILABLE_SPEAKERS>`` section to ``system_capabilities``
        2. Adds a ``SwitchSpeakerTool`` to the tool collection
        """
        speaker_descriptions = self._build_speaker_descriptions()
        if not speaker_descriptions:
            return

        # Append speaker descriptions to system_capabilities
        if self.settings.messages.system is not None:
            has_capabilities = self.settings.messages.system.system_capabilities
            separator = "\n\n" if has_capabilities else ""
            self.settings.messages.system.system_capabilities += (
                f"{separator}<AVAILABLE_SPEAKERS>\n"
                "The following specialized speakers are available in this "
                "conversation. Use the switch_speaker tool to hand off to "
                "them when appropriate.\n\n"
                f"{speaker_descriptions}\n"
                "</AVAILABLE_SPEAKERS>"
            )

        # Create switch_speaker tool with valid speaker names
        handoff_speakers = [speaker.name for speaker in self.speakers]
        switch_tool = SwitchSpeakerTool(speaker_names=handoff_speakers)
        self.tools.append_tool(switch_tool)

    def _setup_secrets(self) -> None:
        """Advertise available secret placeholders to the model.

        Appends an ``<AVAILABLE_SECRETS>`` section to ``system_capabilities`` so the
        model knows which ``<|secret|>NAME<|secret|>`` placeholders it may use. No-op
        when no
        secrets are registered or no system prompt is set. Mirrors
        ``_setup_speaker_handoff``'s ``<AVAILABLE_SPEAKERS>`` injection.
        """
        if not self.secret_vault:
            return
        section = self.secret_vault.system_prompt_section()
        if not section or self.settings.messages.system is None:
            return
        has_capabilities = self.settings.messages.system.system_capabilities
        separator = "\n\n" if has_capabilities else ""
        self.settings.messages.system.system_capabilities += f"{separator}{section}"

    def _build_speaker_descriptions(self) -> str:
        """Build formatted speaker descriptions for the system prompt.

        Returns:
            Formatted string with speaker names and descriptions,
            or empty string if no speakers have descriptions.
        """
        return "\n\n".join(
            f"### {s.name}\n{s.description}"
            for s in self.speakers
            if s.name != self.speakers.default_speaker
        )

    @tracer.start_as_current_span("_execute_step")
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
        self._on_step_start(self._step_index)

        # 1. Infer next speaker
        self._switch_speaker_if_needed()

        # 2. Get next message(s) from speaker and add to history
        result = self._get_next_message()

        # 3. Execute tool calls if applicable
        continue_loop = False
        if result.messages_to_add:
            last_message = result.messages_to_add[-1]
            tool_result_message = self._execute_tools_if_present(last_message)
            if tool_result_message:
                self._add_message(tool_result_message)
                continue_loop = True  # we always continue after a tool was called

        # 4. Check if conversation should continue and switch speaker if necessary
        # Note:_handle_continue_conversation must always be called (not short-circuited)
        # because it has side effects (e.g., triggering speaker switches).
        status_continue = self._handle_continue_conversation(result)
        continue_loop = continue_loop or status_continue

        self._on_step_end(self._step_index, result)
        self._step_index += 1

        return continue_loop

    @tracer.start_as_current_span("_get_next_message")
    def _get_next_message(self) -> SpeakerResult:
        logger.debug("Executing step with speaker: %s", self.current_speaker.name)
        result: SpeakerResult = self.current_speaker.handle_step(
            self, self.cache_manager
        )
        for message in result.messages_to_add:
            self._add_message(message)
        return result

    @tracer.start_as_current_span("_execute_tools_if_present")
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
        self._on_tool_execution_start(tool_names)
        tool_results = self.tools.run(tool_use_blocks)
        self._on_tool_execution_end(tool_names)

        if not tool_results:
            return None

        # Return tool results as a user message
        return MessageParam(content=tool_results, role="user")

    def _add_message(self, message: MessageParam) -> None:
        """Add message to conversation history.

        Redacts literal secret values from the message before it reaches the reporter,
        truncation strategy (LLM history) or cache recording. This is the single point
        every speaker-produced message and tool result flows through.

        Args:
            message: Message to add
        """
        # Defense-in-depth: scrub any literal secret value that slipped into content
        # (e.g. a tool echoing typed text) before it leaves the trusted boundary.
        message = self.secret_vault.redact_message(message)

        # Report to reporter
        self._reporter.add_message(
            self.current_speaker.name, message.model_dump(mode="json")
        )

        if not self._truncation_strategy:
            logger.error("No truncation strategy, cannot add message")
            return

        # Add to truncation strategy
        self._truncation_strategy.append_message(message)

    @tracer.start_as_current_span("_handle_continue_conversation")
    def _handle_continue_conversation(self, result: SpeakerResult) -> bool:
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
                self.switch_speaker(
                    result.next_speaker,
                    speaker_context=result.speaker_context,
                )
            return True
        # status == "continue"
        return True

    def _switch_speaker_if_needed(self) -> None:
        """Switch to default speaker if current one cannot handle."""
        if not self.current_speaker.can_handle(self):
            logger.debug(
                "Speaker %s cannot handle current state, switching to default",
                self.current_speaker.name,
            )
            self.switch_speaker(self.speakers.default_speaker)

    @tracer.start_as_current_span("switch_speaker")
    def switch_speaker(
        self,
        speaker_name: str,
        speaker_context: dict[str, Any] | None = None,
    ) -> None:
        """Switch to a different speaker, optionally passing activation context.

        Args:
            speaker_name: Name of the speaker to switch to.
            speaker_context: Optional activation context to pass to the
                target speaker via ``on_activate()``.
        """
        old_speaker = self.current_speaker
        self.current_speaker = self.speakers[speaker_name]
        logger.info(
            "Switched speaker: %s => %s",
            old_speaker.name,
            self.current_speaker.name,
        )
        self._on_speaker_switch(
            old_speaker.name,
            self.current_speaker.name,
        )
        if speaker_context is not None:
            self.current_speaker.on_activate(speaker_context)

    def get_messages(self) -> list[MessageParam]:
        """Get current message history from truncation strategy.

        Returns:
            List of messages in current conversation
        """
        return (
            self._truncation_strategy.full_messages if self._truncation_strategy else []
        )

    def get_truncation_strategy(self) -> TruncationStrategy | None:
        """Get current truncation strategy.

        Returns:
            Current truncation strategy or None if not initialized
        """
        return self._truncation_strategy
