"""Unit tests for OpenAIVlmProvider."""

from unittest.mock import MagicMock

from openai import OpenAI

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider
from askui.models.shared.agent_message_param import MessageParam


class TestOpenAIVlmProvider:
    def test_model_id(self) -> None:
        provider = OpenAIVlmProvider(model_id="gpt-4o", api_key="sk-test")
        assert provider.model_id == "gpt-4o"

    def test_pricing_returns_none(self) -> None:
        provider = OpenAIVlmProvider(model_id="gpt-4o", api_key="sk-test")
        assert provider.pricing is None

    def test_injected_client_used(self) -> None:
        mock_client = MagicMock(spec=OpenAI)
        provider = OpenAIVlmProvider(model_id="gpt-4o", client=mock_client)
        assert provider._client is mock_client

    def test_create_message_delegates_to_messages_api(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    finish_reason="stop",
                    message=MagicMock(content="done", tool_calls=None),
                )
            ],
            usage=MagicMock(prompt_tokens=5, completion_tokens=10),
        )

        provider = OpenAIVlmProvider(model_id="gpt-4o", client=mock_client)
        result = provider.create_message(
            messages=[MessageParam(role="user", content="hi")],
        )

        mock_client.chat.completions.create.assert_called_once()
        assert result.role == "assistant"
