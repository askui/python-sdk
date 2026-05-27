"""Unit tests for OllamaVlmProvider."""

from unittest.mock import MagicMock

from openai import OpenAI

from askui.model_providers.ollama_vlm_provider import OllamaVlmProvider
from askui.models.shared.agent_message_param import MessageParam


class TestOllamaVlmProvider:
    def test_default_model_id(self) -> None:
        provider = OllamaVlmProvider()
        assert provider.model_id == "qwen3.5"

    def test_custom_model_id(self) -> None:
        provider = OllamaVlmProvider(model_id="llava")
        assert provider.model_id == "llava"

    def test_pricing_returns_none(self) -> None:
        provider = OllamaVlmProvider()
        assert provider.pricing is None

    def test_injected_client_used(self) -> None:
        mock_client = MagicMock(spec=OpenAI)
        provider = OllamaVlmProvider(client=mock_client)
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

        provider = OllamaVlmProvider(
            model_id="test-model",
            client=mock_client,
        )
        result = provider.create_message(
            messages=[MessageParam(role="user", content="hi")],
        )

        mock_client.chat.completions.create.assert_called_once()
        assert result.role == "assistant"
