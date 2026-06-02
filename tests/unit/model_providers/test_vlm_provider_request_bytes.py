"""Tests for the ``max_request_bytes`` limit advertised by VLM providers."""

from unittest.mock import MagicMock

from askui.model_providers.anthropic_vlm_provider import AnthropicVlmProvider
from askui.model_providers.askui_vlm_provider import AskUIVlmProvider
from askui.models.shared.request_size import ANTHROPIC_MAX_REQUEST_BYTES


class TestProviderMaxRequestBytes:
    def test_anthropic_provider_reports_anthropic_limit(self) -> None:
        provider = AnthropicVlmProvider(client=MagicMock())
        assert provider.max_request_bytes == ANTHROPIC_MAX_REQUEST_BYTES

    def test_askui_provider_reports_anthropic_limit(self) -> None:
        # Pass mock settings so construction does not validate env credentials.
        provider = AskUIVlmProvider(askui_settings=MagicMock(), client=MagicMock())
        assert provider.max_request_bytes == ANTHROPIC_MAX_REQUEST_BYTES
