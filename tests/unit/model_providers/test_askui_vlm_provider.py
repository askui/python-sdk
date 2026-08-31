"""Unit tests for AskUIVlmProvider proxy routing."""

from unittest.mock import MagicMock

import pytest
from anthropic import Anthropic
from openai import OpenAI

from askui.model_providers.askui_vlm_provider import (
    AskUIVlmProvider,
    _with_google_thinking,
)
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.askui.inference_api_settings import AskUiInferenceApiSettings
from askui.models.openai.messages_api import OpenAIMessagesApi
from askui.models.shared.coordinate_space import (
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
)
from askui.models.shared.prompts import SystemPrompt

# Placeholder workspace id: all zeros except the mandatory UUIDv4
# version (4) and variant (8) nibbles required by pydantic validation.
_WORKSPACE_ID = "00000000-0000-4000-8000-000000000000"


@pytest.fixture
def askui_settings() -> AskUiInferenceApiSettings:
    return AskUiInferenceApiSettings(
        workspace_id=_WORKSPACE_ID,
        token="secret-token",
    )


class TestAskUIVlmProviderRouting:
    def test_claude_model_uses_anthropic_messages_api(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="claude-sonnet-4-6",
        )
        assert isinstance(provider._messages_api, AnthropicMessagesApi)

    def test_gemini_model_uses_openai_messages_api(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="gemini-2.5-pro",
        )
        assert isinstance(provider._messages_api, OpenAIMessagesApi)

    def test_vendor_prefixed_gemini_uses_openai_messages_api(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="google/gemini-3.5-flash",
        )
        assert isinstance(provider._messages_api, OpenAIMessagesApi)

    def test_unknown_model_raises(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="mystery-model-1",
        )
        with pytest.raises(ValueError, match="Cannot infer a backend"):
            _ = provider._messages_api

    def test_gemini_client_targets_openai_proxy(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="gemini-2.5-pro",
        )
        api = provider._messages_api
        assert isinstance(api, OpenAIMessagesApi)
        assert str(api._client.base_url).rstrip("/").endswith("/proxy/openai/v1")

    def test_injected_anthropic_client_used_for_claude(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        mock_client = MagicMock(spec=Anthropic)
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="claude-sonnet-4-6",
            client=mock_client,
        )
        api = provider._messages_api
        assert isinstance(api, AnthropicMessagesApi)
        assert api._client is mock_client

    def test_injected_openai_client_used_for_gemini(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        mock_client = MagicMock(spec=OpenAI)
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="gemini-2.5-pro",
            client=mock_client,
        )
        api = provider._messages_api
        assert isinstance(api, OpenAIMessagesApi)
        assert api._client is mock_client


class TestAskUIVlmProviderSystemPrompt:
    def test_claude_prompt_unchanged(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="claude-sonnet-4-6",
        )
        system = SystemPrompt(prompt="Base prompt.")
        assert provider.augment_system_prompt(system) is system

    def test_gemini_prompt_augmented_with_coordinates(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="gemini-2.5-pro",
        )
        system = SystemPrompt(prompt="Base prompt.")
        rendered = str(provider.augment_system_prompt(system))
        assert "Base prompt." in rendered
        assert "1000x1000 normalised grid" in rendered


class TestAskUIVlmProviderCoordinateSpace:
    def test_claude_uses_pixel_coordinate_space(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="claude-sonnet-4-6",
        )
        assert provider.coordinate_space == PixelCoordinateSpace()

    def test_gemini_uses_scaled_coordinate_space(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="google/gemini-3.5-flash",
        )
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )


class TestWithGoogleThinking:
    def test_injects_include_thoughts_when_absent(self) -> None:
        options = _with_google_thinking(None)
        assert options == {
            "extra_body": {"google": {"thinking_config": {"include_thoughts": True}}}
        }

    def test_preserves_other_provider_options(self) -> None:
        options = _with_google_thinking({"temperature": 0.2})
        assert options["temperature"] == 0.2
        assert options["extra_body"]["google"]["thinking_config"]["include_thoughts"]

    def test_caller_extra_body_wins(self) -> None:
        caller = {"extra_body": {"google": {"foo": "bar"}}}
        options = _with_google_thinking(caller)
        assert options["extra_body"] == {"google": {"foo": "bar"}}


class TestAskUIVlmProviderThinkingRequest:
    def test_gemini_requests_thoughts(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="gemini-2.5-pro",
        )
        stub = MagicMock()
        provider.__dict__["_messages_api"] = stub
        provider.create_message(messages=[])
        forwarded = stub.create_message.call_args.kwargs["provider_options"]
        assert forwarded["extra_body"]["google"]["thinking_config"]["include_thoughts"]

    def test_claude_does_not_request_thoughts(
        self, askui_settings: AskUiInferenceApiSettings
    ) -> None:
        provider = AskUIVlmProvider(
            askui_settings=askui_settings,
            model_id="claude-sonnet-4-6",
        )
        stub = MagicMock()
        provider.__dict__["_messages_api"] = stub
        provider.create_message(messages=[])
        assert stub.create_message.call_args.kwargs["provider_options"] is None
