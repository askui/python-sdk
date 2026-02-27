"""Agent speaker for normal LLM API interactions."""

import logging
from typing import TYPE_CHECKING

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
        try:
            response = conversation.vlm_provider.create_message(
                messages=truncation_strategy.messages,
                tools=conversation.tools,
                max_tokens=conversation.settings.messages.max_tokens,
                system=conversation.settings.messages.system,
                thinking=conversation.settings.messages.thinking,
                tool_choice=conversation.settings.messages.tool_choice,
                temperature=conversation.settings.messages.temperature,
                provider_options=conversation.settings.messages.provider_options,
            )

            # Log response
            logger.debug("Agent response: %s", response.model_dump(mode="json"))

        except Exception:
            logger.exception("Error calling agent API")
            return SpeakerResult(status="failed")

        # Handle stop reason
        try:
            self._handle_stop_reason(
                response, conversation.settings.messages.max_tokens
            )
        except (MaxTokensExceededError, ModelRefusalError):
            logger.exception("Agent stopped with error")
            return SpeakerResult(status="failed", messages_to_add=[response])

        # Determine status based on whether there are tool calls
        # If there are tool calls, conversation will execute them and loop back
        # If no tool calls, conversation is done
        has_tool_calls = self._has_tool_calls(response)
        status = "continue" if has_tool_calls else "done"

        return SpeakerResult(
            status=status,
            messages_to_add=[response],
            usage=response.usage,
        )

    @override
    def get_name(self) -> str:
        """Return speaker name.

        Returns:
            "AgentSpeaker"
        """
        return "AgentSpeaker"

    def _has_tool_calls(self, message: MessageParam) -> bool:
        """Check if message contains tool use blocks.

        Args:
            message: The message to check

        Returns:
            True if message contains tool calls, False otherwise
        """
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
