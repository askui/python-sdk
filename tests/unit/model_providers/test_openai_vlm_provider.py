"""Unit tests for OpenAIVlmProvider."""

from unittest.mock import MagicMock

from openai import OpenAI
from PIL import Image

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider
from askui.models.shared.agent_message_param import MessageParam
from askui.models.shared.coordinate_space import (
    NormalizedCoordinateSpace,
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
)
from askui.models.shared.image_scaler import ImageScaler
from askui.models.shared.prompts import SystemPrompt


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

    def test_coordinate_space_defaults_to_pixel(self) -> None:
        provider = OpenAIVlmProvider(model_id="gpt-4o", api_key="sk-test")
        assert provider.coordinate_space == PixelCoordinateSpace()

    def test_coordinate_space_passthrough(self) -> None:
        provider = OpenAIVlmProvider(
            model_id="gpt-4o",
            api_key="sk-test",
            coordinate_space=ScaledCoordinateSpace(width=1000, height=1000),
        )
        assert provider.coordinate_space == ScaledCoordinateSpace(
            width=1000, height=1000
        )

    def test_augment_system_prompt_scaled_coordinate_space(self) -> None:
        provider = OpenAIVlmProvider(
            model_id="gpt-4o",
            api_key="sk-test",
            coordinate_space=ScaledCoordinateSpace(width=1000, height=1000),
        )
        system = SystemPrompt(prompt="You are a helpful assistant.")
        augmented = provider.augment_system_prompt(system)

        rendered = str(augmented)
        assert "You are a helpful assistant." in rendered
        assert "1000x1000 normalised grid" in rendered

    def test_augment_system_prompt_pixel_coordinate_space(self) -> None:
        provider = OpenAIVlmProvider(model_id="gpt-4o", api_key="sk-test")
        system = SystemPrompt(prompt="Base prompt.")
        augmented = provider.augment_system_prompt(system)

        rendered = str(augmented)
        assert "normalised grid" not in rendered
        assert "pixel space matching the screenshot dimensions" in rendered


class TestImageScaler:
    def test_default_scaler_returns_valid_image(self) -> None:
        provider = OpenAIVlmProvider(model_id="gpt-4o", api_key="sk-test")
        img = Image.new("RGB", (1920, 1080))
        scaled = provider.image_scaler(img)
        assert scaled.width <= 2048
        assert scaled.height <= 2048

    def test_custom_scaler_override(self) -> None:
        class _FixedSizeScaler(ImageScaler):
            def __call__(self, image: Image.Image) -> Image.Image:
                return image.resize((100, 100))

        provider = OpenAIVlmProvider(
            model_id="gpt-4o",
            api_key="sk-test",
            image_scaler=_FixedSizeScaler(),
        )
        img = Image.new("RGB", (1920, 1080))
        scaled = provider.image_scaler(img)
        assert scaled.size == (100, 100)


class TestPixelCoordinateSpacePrompt:
    def test_shows_pixel_space_description(self) -> None:
        cs = PixelCoordinateSpace()
        result = cs.build_prompt_section()
        assert "pixel space matching the screenshot dimensions" in result
        assert "normalised grid" not in result

    def test_includes_origin_info(self) -> None:
        cs = PixelCoordinateSpace()
        result = cs.build_prompt_section()
        assert "top-left" in result


class TestScaledCoordinateSpacePrompt:
    def test_shows_normalised_grid(self) -> None:
        cs = ScaledCoordinateSpace(width=1000, height=1000)
        result = cs.build_prompt_section()
        assert "1000x1000 normalised grid" in result
        assert "0 <= x < 1000" in result
        assert "0 <= y < 1000" in result

    def test_includes_origin_info(self) -> None:
        cs = ScaledCoordinateSpace(width=1000, height=1000)
        result = cs.build_prompt_section()
        assert "top-left" in result


class TestNormalizedCoordinateSpacePrompt:
    def test_shows_normalised_floats(self) -> None:
        cs = NormalizedCoordinateSpace()
        result = cs.build_prompt_section()
        assert "0.0 <= x <= 1.0" in result
        assert "0.0 <= y <= 1.0" in result
        assert "normalised floats" in result

    def test_includes_origin_info(self) -> None:
        cs = NormalizedCoordinateSpace()
        result = cs.build_prompt_section()
        assert "top-left" in result


class TestMapsToScreenshotPixels:
    def test_pixel_returns_true(self) -> None:
        assert PixelCoordinateSpace().maps_to_screenshot_pixels is True

    def test_scaled_returns_false(self) -> None:
        assert (
            ScaledCoordinateSpace(width=1000, height=1000).maps_to_screenshot_pixels
            is False
        )

    def test_normalized_returns_false(self) -> None:
        assert NormalizedCoordinateSpace().maps_to_screenshot_pixels is False


class TestMapToTarget:
    def test_pixel_identity(self) -> None:
        cs = PixelCoordinateSpace()
        assert cs.map_to_target(512, 384, (1024, 768)) == (512, 384)

    def test_pixel_truncates_floats(self) -> None:
        cs = PixelCoordinateSpace()
        assert cs.map_to_target(512.7, 384.3, (1024, 768)) == (512, 384)

    def test_scaled_maps_correctly(self) -> None:
        cs = ScaledCoordinateSpace(width=1000, height=1000)
        assert cs.map_to_target(500, 500, (1024, 768)) == (512, 384)

    def test_scaled_zero(self) -> None:
        cs = ScaledCoordinateSpace(width=1000, height=1000)
        assert cs.map_to_target(0, 0, (1024, 768)) == (0, 0)

    def test_normalized_maps_correctly(self) -> None:
        cs = NormalizedCoordinateSpace()
        assert cs.map_to_target(0.5, 0.5, (1024, 768)) == (512, 384)

    def test_normalized_zero(self) -> None:
        cs = NormalizedCoordinateSpace()
        assert cs.map_to_target(0.0, 0.0, (1024, 768)) == (0, 0)

    def test_normalized_one(self) -> None:
        cs = NormalizedCoordinateSpace()
        assert cs.map_to_target(1.0, 1.0, (1024, 768)) == (1024, 768)


class TestBackendRouting:
    def test_default_base_url_uses_responses_api(self) -> None:
        from askui.models.openai.responses_api import OpenAIResponsesApi

        provider = OpenAIVlmProvider(model_id="gpt-5.6-terra", api_key="sk-x")
        assert isinstance(provider._messages_api, OpenAIResponsesApi)

    def test_custom_base_url_uses_chat_completions(self) -> None:
        from askui.models.openai.messages_api import OpenAIMessagesApi

        provider = OpenAIVlmProvider(
            model_id="qwen2.5vl",
            api_key="sk-x",
            base_url="http://localhost:11434/v1",
        )
        assert isinstance(provider._messages_api, OpenAIMessagesApi)
