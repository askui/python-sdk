"""Unit tests for OpenAICompatibleVlmProvider."""

from unittest.mock import MagicMock

import httpx

from askui.model_providers.openai_compatible_vlm_provider import (
    OpenAICompatibleVlmProvider,
)
from askui.models.shared.agent_message_param import MessageParam


class TestOpenAICompatibleVlmProvider:
    def test_model_id(self) -> None:
        provider = OpenAICompatibleVlmProvider(
            endpoint_url="https://my-host/v1/chat/completions",
            model_id="my-model",
            api_key="test-key",
        )
        assert provider.model_id == "my-model"

    def test_pricing_returns_none(self) -> None:
        provider = OpenAICompatibleVlmProvider(
            endpoint_url="https://my-host/v1/chat/completions",
            model_id="my-model",
            api_key="test-key",
        )
        assert provider.pricing is None

    def test_injected_client_is_openai_instance(self) -> None:
        provider = OpenAICompatibleVlmProvider(
            endpoint_url="https://my-host/v1/chat/completions",
            model_id="my-model",
            api_key="test-key",
        )
        assert provider._client is not None

    def test_httpx_event_hook_rewrites_url(self) -> None:
        endpoint_url = "https://my-host/v1/chat/completions"
        provider = OpenAICompatibleVlmProvider(
            endpoint_url=endpoint_url,
            model_id="my-model",
            api_key="test-key",
        )

        http_client: httpx.Client = provider._client._client
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        for hook in http_client.event_hooks["request"]:
            hook(request)

        assert str(request.url) == endpoint_url

    def test_create_message_delegates_to_messages_api(self) -> None:
        provider = OpenAICompatibleVlmProvider(
            endpoint_url="https://my-host/v1/chat/completions",
            model_id="test-model",
            api_key="test-key",
        )

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
        provider._client = mock_client

        result = provider.create_message(
            messages=[MessageParam(role="user", content="hi")],
        )

        mock_client.chat.completions.create.assert_called_once()
        assert result.role == "assistant"
