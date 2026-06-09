"""Unit tests for OllamaVlmProvider."""

from unittest.mock import MagicMock

from openai import OpenAI

from askui.model_providers.ollama_vlm_provider import OllamaVlmProvider
from askui.models.shared.agent_message_param import MessageParam
from askui.models.shared.coordinate_space import (
    NormalizedCoordinateSpace,
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
)


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

    def test_coordinate_space_auto_detects_qwen(self) -> None:
        provider = OllamaVlmProvider(model_id="qwen3.5")
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )

    def test_coordinate_space_auto_detects_qwen_case_insensitive(self) -> None:
        provider = OllamaVlmProvider(model_id="Qwen2-VL")
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )

    def test_coordinate_space_auto_detects_kimi(self) -> None:
        provider = OllamaVlmProvider(model_id="kimi-vl")
        assert provider.coordinate_space == NormalizedCoordinateSpace()

    def test_coordinate_space_auto_detects_kimi_case_insensitive(self) -> None:
        provider = OllamaVlmProvider(model_id="Kimi-VL-A3B")
        assert provider.coordinate_space == NormalizedCoordinateSpace()

    def test_coordinate_space_default_for_non_qwen(self) -> None:
        provider = OllamaVlmProvider(model_id="llava")
        assert provider.coordinate_space == PixelCoordinateSpace()

    def test_coordinate_space_explicit_override(self) -> None:
        provider = OllamaVlmProvider(
            model_id="llava",
            coordinate_space=ScaledCoordinateSpace(width=500, height=500),
        )
        assert provider.coordinate_space == ScaledCoordinateSpace(width=500, height=500)

    def test_coordinate_space_explicit_override_takes_precedence(self) -> None:
        provider = OllamaVlmProvider(
            model_id="qwen3.5",
            coordinate_space=ScaledCoordinateSpace(width=2000, height=2000),
        )
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=2000, height=2000
        )

    def test_coordinate_space_explicit_pixel_overrides_qwen_auto_detect(self) -> None:
        provider = OllamaVlmProvider(
            model_id="qwen3.5",
            coordinate_space=PixelCoordinateSpace(),
        )
        assert provider.coordinate_space == PixelCoordinateSpace()

    def test_coordinate_space_auto_detects_holo(self) -> None:
        provider = OllamaVlmProvider(model_id="holo3.1-35b-a3b")
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )

    def test_coordinate_space_auto_detects_holo_case_insensitive(self) -> None:
        provider = OllamaVlmProvider(model_id="Holo-3.1-4B")
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )
