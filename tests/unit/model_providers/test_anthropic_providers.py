"""Unit tests for the direct-Anthropic providers' defaults."""

from askui.model_providers.anthropic_image_qa_provider import (
    AnthropicImageQAProvider,
)
from askui.model_providers.anthropic_vlm_provider import AnthropicVlmProvider


class TestAnthropicVlmProviderDefaults:
    def test_default_model_is_haiku_5_5(self) -> None:
        provider = AnthropicVlmProvider(api_key="sk-ant-test")
        assert provider.model_id == "claude-haiku-5-5"

    def test_default_model_is_priced(self) -> None:
        pricing = AnthropicVlmProvider(api_key="sk-ant-test").pricing
        assert pricing is not None
        assert pricing.input_cost_per_million_tokens == 0.10
        assert pricing.output_cost_per_million_tokens == 0.50


class TestAnthropicImageQAProviderDefaults:
    def test_default_model_is_haiku_5_5(self) -> None:
        provider = AnthropicImageQAProvider(api_key="sk-ant-test")
        assert provider._get_model._model_id == "claude-haiku-5-5"
