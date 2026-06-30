"""Agent speaker for normal LLM API interactions."""

import logging
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from askui.models.exceptions import MaxTokensExceededError, ModelRefusalError
from askui.models.shared.agent_message_param import MessageParam

from .speaker import Speaker, SpeakerResult

if TYPE_CHECKING:
    from askui.models.shared.conversation import Conversation
    from askui.utils.caching.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class AgentSpeaker(Speaker):
    """Speaker that handles normal agent API calls.

    This speaker generates messages from the LLM by:
    1. Making API calls to get agent responses via VlmProvider
    2. Handling stop reasons (max_tokens, refusal)
    3. Returning messages for the Conversation to process

    Tool execution is handled by the Conversation class, not by this speaker.

    The VlmProvider is accessed from the Conversation instance.
    """

    def __init__(self) -> None:
        self.name = "AgentSpeaker"
        self.description = (
            "Default speaker that handles LLM API calls and "
            "coordinates conversation flow."
        )

    @override
    def can_handle(self, conversation: "Conversation") -> bool:  # noqa: ARG002
        """AgentSpeaker can always handle normal conversation flow.

        Args:
            conversation: The conversation instance

        Returns:
            Always True - this is the default speaker
        """
        return True

    @override
    def handle_step(
        self,
        conversation: "Conversation",
        cache_manager: "CacheManager | None",  # noqa: ARG002
    ) -> SpeakerResult:
        """Get next message from the agent API.

        This speaker only generates messages from the LLM. Tool execution
        is handled by the Conversation class.

        Args:
            conversation: The conversation instance with current state
            cache_manager: Optional cache manager (not used by this speaker)

        Returns:
            SpeakerResult with the agent's message
        """
        messages = conversation.get_messages()
        truncation_strategy = conversation.get_truncation_strategy()

        if not truncation_strategy:
            logger.error("No truncation strategy available")
            return SpeakerResult(status="failed")

        # Only call agent if last message is from user
        if not messages or messages[-1].role != "user":
            logger.debug("Last message not from user, nothing to do")
            return SpeakerResult(status="done")

        # Make API call to get agent response using VlmProvider
        response = conversation.vlm_provider.create_message(
            messages=truncation_strategy.truncated_messages,
            tools=conversation.tools,
            max_tokens=conversation.settings.messages.max_tokens,
            system=conversation.settings.messages.system,
            thinking=conversation.settings.messages.thinking,
            tool_choice=conversation.settings.messages.tool_choice,
            temperature=conversation.settings.messages.temperature,
            provider_options=conversation.settings.messages.provider_options,
        )

        # Log response
        if logger.isEnabledFor(logging.DEBUG):  # avoid costly model_dump if possible
            logger.debug("Agent response: %s", response.model_dump(mode="json"))

        # Handle stop reason
        try:
            self._handle_stop_reason(
                response, conversation.settings.messages.max_tokens
            )
        except (MaxTokensExceededError, ModelRefusalError):
            logger.exception("Agent stopped with error")
            return SpeakerResult(status="failed", messages_to_add=[response])

        # Check for tool_use stop_reason with no actual tool calls
        if response.stop_reason == "tool_use" and not self._has_tool_calls(response):
            logger.warning(
                "Model returned stop_reason 'tool_use' but content "
                "contains no tool_use blocks"
            )
            messages_to_add: list[MessageParam] = []
            # Ensure the response has valid content for conversation history
            # (empty content would violate API alternating-role requirements)
            if isinstance(response.content, list) and not response.content:
                response = MessageParam(
                    role="assistant",
                    content="I need to use a tool for this task.",
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                )
            messages_to_add.append(response)
            feedback_msg = (
                "No tool name was provided in your response. "
                "Please provide a valid tool call."
            )
            messages_to_add.append(MessageParam(role="user", content=feedback_msg))
            return SpeakerResult(
                status="continue",
                messages_to_add=messages_to_add,
                usage=response.usage,
            )

        # Check for switch_speaker tool call
        switch_info = self._extract_switch_speaker(response)
        if switch_info:
            speaker_name, speaker_context = switch_info
            return SpeakerResult(
                status="switch_speaker",
                next_speaker=speaker_name,
                speaker_context=speaker_context,
                messages_to_add=[response],
                usage=response.usage,
            )

        # "continue" if tool calls present (conversation needs to execute them
        # and feed results back), "done" if the LLM produced a final text response.
        status = "continue" if self._has_tool_calls(response) else "done"
        return SpeakerResult(
            status=status,
            messages_to_add=[response],
            usage=response.usage,
        )

    def _extract_switch_speaker(
        self, message: MessageParam
    ) -> tuple[str, dict[str, Any]] | None:
        """Extract switch_speaker tool call from message if present.

        Args:
            message: The assistant message to inspect.

        Returns:
            Tuple of (speaker_name, speaker_context) if found, None otherwise.
        """
        if isinstance(message.content, str):
            return None

        for block in message.content:
            if block.type == "tool_use" and block.name.startswith("switch_speaker"):
                input_data: dict[str, Any] = (
                    dict(block.input) if isinstance(block.input, dict) else {}
                )
                speaker_name = str(input_data.get("speaker_name", ""))
                speaker_context: dict[str, Any] = (
                    input_data.get("speaker_context", {}) or {}
                )
                return speaker_name, speaker_context

        return None

    def _has_tool_calls(self, message: MessageParam) -> bool:
        """Check if the message contains tool use blocks."""
        if isinstance(message.content, str):
            return False
        return any(block.type == "tool_use" for block in message.content)

    def _handle_stop_reason(self, message: MessageParam, max_tokens: int) -> None:
        """Handle agent stop reasons.

        Args:
            message: Message to check stop reason
            max_tokens: Maximum tokens configured

        Raises:
            MaxTokensExceededError: If agent stopped due to max tokens
            ModelRefusalError: If agent refused the request
        """
        if message.stop_reason == "max_tokens":
            raise MaxTokensExceededError(max_tokens)
        if message.stop_reason == "refusal":
            raise ModelRefusalError
