"""Unit tests for AgentSpeaker."""

from unittest.mock import MagicMock

from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    UsageParam,
)
from askui.speaker.agent_speaker import AgentSpeaker


def _make_conversation(response: MessageParam) -> MagicMock:
    """Create a mock Conversation that returns the given response."""
    conversation = MagicMock()
    conversation.get_messages.return_value = [
        MessageParam(role="user", content="do something")
    ]
    truncation_strategy = MagicMock()
    truncation_strategy.truncated_messages = [
        MessageParam(role="user", content="do something")
    ]
    conversation.get_truncation_strategy.return_value = truncation_strategy
    conversation.vlm_provider.create_message.return_value = response
    conversation.tools = []
    conversation.settings.messages.max_tokens = 4096
    conversation.settings.messages.system = None
    conversation.settings.messages.thinking = None
    conversation.settings.messages.tool_choice = None
    conversation.settings.messages.temperature = None
    conversation.settings.messages.provider_options = None
    return conversation


class TestEmptyToolUseContent:
    """Tests for handling stop_reason='tool_use' with no tool_use blocks."""

    def test_empty_content_returns_continue(self) -> None:
        """When content is an empty list, status should be 'continue'."""
        response = MessageParam(
            role="assistant",
            content=[],
            stop_reason="tool_use",
            usage=UsageParam(input_tokens=10, output_tokens=5),
        )
        conversation = _make_conversation(response)
        speaker = AgentSpeaker()

        result = speaker.handle_step(conversation, cache_manager=None)

        assert result.status == "continue"

    def test_empty_content_adds_assistant_and_user_messages(self) -> None:
        """Messages should contain a synthetic assistant msg and a user feedback msg."""
        response = MessageParam(
            role="assistant",
            content=[],
            stop_reason="tool_use",
            usage=UsageParam(input_tokens=10, output_tokens=5),
        )
        conversation = _make_conversation(response)
        speaker = AgentSpeaker()

        result = speaker.handle_step(conversation, cache_manager=None)

        assert len(result.messages_to_add) == 2
        assert result.messages_to_add[0].role == "assistant"
        assert (
            result.messages_to_add[0].content == "I need to use a tool for this task."
        )
        assert result.messages_to_add[1].role == "user"
        assert "No tool name was provided" in str(result.messages_to_add[1].content)

    def test_text_only_content_returns_continue(self) -> None:
        """When content has text blocks but no tool_use blocks, status is 'continue'."""
        response = MessageParam(
            role="assistant",
            content=[TextBlockParam(text="Let me think about this...")],
            stop_reason="tool_use",
            usage=UsageParam(input_tokens=10, output_tokens=5),
        )
        conversation = _make_conversation(response)
        speaker = AgentSpeaker()

        result = speaker.handle_step(conversation, cache_manager=None)

        assert result.status == "continue"

    def test_text_only_content_preserves_original_response(self) -> None:
        """When content has text (non-empty), the original response is preserved."""
        response = MessageParam(
            role="assistant",
            content=[TextBlockParam(text="Let me think about this...")],
            stop_reason="tool_use",
            usage=UsageParam(input_tokens=10, output_tokens=5),
        )
        conversation = _make_conversation(response)
        speaker = AgentSpeaker()

        result = speaker.handle_step(conversation, cache_manager=None)

        assert len(result.messages_to_add) == 2
        # Original response is kept as-is (not replaced)
        assert result.messages_to_add[0].content == [
            TextBlockParam(text="Let me think about this...")
        ]
        assert result.messages_to_add[1].role == "user"

    def test_usage_is_propagated(self) -> None:
        """Usage from the response should be on the SpeakerResult."""
        usage = UsageParam(input_tokens=100, output_tokens=50)
        response = MessageParam(
            role="assistant",
            content=[],
            stop_reason="tool_use",
            usage=usage,
        )
        conversation = _make_conversation(response)
        speaker = AgentSpeaker()

        result = speaker.handle_step(conversation, cache_manager=None)

        assert result.usage == usage
