"""Unit tests for OllamaImageQAProvider."""

from unittest.mock import MagicMock

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from askui.model_providers.ollama_image_qa_provider import OllamaImageQAProvider
from askui.models.shared.settings import GetSettings
from askui.utils.image_utils import ImageSource


def _make_completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=1234567890,
        model="qwen3.5",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


class TestOllamaImageQAProvider:
    def test_default_model_id(self) -> None:
        provider = OllamaImageQAProvider()
        assert provider._model_id == "qwen3.5"

    def test_custom_model_id(self) -> None:
        provider = OllamaImageQAProvider(model_id="llava")
        assert provider._model_id == "llava"

    def test_injected_client_used(self) -> None:
        mock_client = MagicMock(spec=OpenAI)
        provider = OllamaImageQAProvider(client=mock_client)
        assert provider._client is mock_client

    def test_query_delegates_to_get_model(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            "The button"
        )

        source = MagicMock(spec=ImageSource)
        source.to_data_url.return_value = "data:image/png;base64,abc"

        provider = OllamaImageQAProvider(model_id="qwen2.5vl", client=mock_client)
        result = provider.query(
            query="What is this?",
            source=source,
            response_schema=None,
            get_settings=GetSettings(),
        )

        assert result == "The button"
        mock_client.chat.completions.create.assert_called_once()
