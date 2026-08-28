"""Unit tests for Anthropic messages API output_config / thinking handling."""

import inspect
from typing import Any
from unittest.mock import MagicMock

import anthropic
from anthropic import omit

from askui.models.anthropic.messages_api import (
    AnthropicMessagesApi,
    _parse_to_anthropic_types,
    from_content_block,
    from_message_param,
)
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)


def _assert_no_nulls(value: Any, path: str = "") -> None:
    """Recursively assert that a serialized block contains no `None` values."""
    if isinstance(value, dict):
        for key, sub in value.items():
            assert sub is not None, f"unexpected null at {path}.{key}"
            _assert_no_nulls(sub, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _assert_no_nulls(item, f"{path}[{i}]")


class TestSerializationOmitsNulls:
    """Content blocks must not serialize optional fields as explicit `null`.

    Real Anthropic tolerates `cache_control: null`, but stricter
    Anthropic-compatible endpoints (e.g. OpenRouter) reject it.
    """

    def test_text_block_has_no_nulls(self) -> None:
        dumped = from_content_block(TextBlockParam(text="hi"))
        assert "cache_control" not in dumped
        assert "citations" not in dumped
        _assert_no_nulls(dumped)

    def test_image_block_has_no_nulls(self) -> None:
        block = ImageBlockParam(
            source=Base64ImageSourceParam(data="AAAA", media_type="image/png")
        )
        _assert_no_nulls(from_content_block(block))

    def test_tool_result_with_image_has_no_nested_nulls(self) -> None:
        block = ToolResultBlockParam(
            tool_use_id="t1",
            content=[
                TextBlockParam(text="hi"),
                ImageBlockParam(
                    source=Base64ImageSourceParam(data="AAAA", media_type="image/png")
                ),
            ],
        )
        _assert_no_nulls(from_content_block(block))

    def test_tool_use_block_has_no_nulls_and_drops_internal_fields(self) -> None:
        block = ToolUseBlockParam(id="1", name="click", input={"x": 1})
        dumped = from_content_block(block)
        assert "visual_representation" not in dumped
        assert "extra_content" not in dumped
        _assert_no_nulls(dumped)

    def test_message_with_block_content_has_no_nulls(self) -> None:
        message = MessageParam(role="user", content=[TextBlockParam(text="hi")])
        _assert_no_nulls(from_message_param(message))


class TestParseToAnthropicTypes:
    """`output_config` (carrying `effort`) is forwarded, or omitted when absent."""

    def test_output_config_passed_through(self) -> None:
        result = _parse_to_anthropic_types(tools=None, output_config={"effort": "high"})
        assert result[5] == {"effort": "high"}

    def test_no_output_config_is_omitted(self) -> None:
        result = _parse_to_anthropic_types(tools=None, output_config=None)
        assert result[5] is omit

    def test_adaptive_thinking_passed_through(self) -> None:
        result = _parse_to_anthropic_types(tools=None, thinking={"type": "adaptive"})
        assert result[4] == {"type": "adaptive"}


class TestCreateMessage:
    """`create_message` reads output_config from provider_options."""

    def _make_api(self) -> tuple[AnthropicMessagesApi, MagicMock]:
        client = MagicMock()
        response = MagicMock()
        response.model_dump.return_value = {
            "role": "assistant",
            "content": "hello",
        }
        client.beta.messages.create.return_value = response
        return AnthropicMessagesApi(client=client), client

    def test_effort_from_provider_options_forwarded(self) -> None:
        api, client = self._make_api()

        result = api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-5",
            thinking={"type": "adaptive"},
            provider_options={"output_config": {"effort": "medium"}},
        )

        assert isinstance(result, MessageParam)
        kwargs = client.beta.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "medium"}

    def test_no_output_config_omits_it(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-4-5-20250929",
            thinking={"type": "enabled", "budget_tokens": 2048},
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        assert kwargs["output_config"] is omit
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    # A legacy budget-thinking model (accepts sampling) vs. an adaptive model
    # (rejects sampling / deprecated temperature).
    _SAMPLING_MODEL = "claude-sonnet-4-5"
    _NO_SAMPLING_MODEL = "claude-sonnet-5"

    def test_temperature_sent_via_extra_body_for_sampling_model(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id=self._SAMPLING_MODEL,
            temperature=0.3,
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        # Routed through the request body (never as the typed `temperature=`
        # kwarg, which newer clients removed).
        assert "temperature" not in kwargs
        assert kwargs["extra_body"] == {"temperature": 0.3}

    def test_temperature_zero_sent_for_sampling_model(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id=self._SAMPLING_MODEL,
            temperature=0.0,
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        assert kwargs["extra_body"] == {"temperature": 0.0}

    def test_temperature_dropped_for_non_sampling_model(self) -> None:
        api, client = self._make_api()

        for temperature in (0.0, 0.5, 1.0):
            api.create_message(
                messages=[MessageParam(role="user", content="hi")],
                model_id=self._NO_SAMPLING_MODEL,
                temperature=temperature,
            )
            kwargs = client.beta.messages.create.call_args.kwargs
            assert "temperature" not in kwargs
            assert "extra_body" not in kwargs

    def test_temperature_never_in_body_when_unset(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id=self._SAMPLING_MODEL,
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert "extra_body" not in kwargs

    def test_warns_once_per_model_for_non_sampling_temperature(
        self, caplog: Any
    ) -> None:
        import logging

        api, _ = self._make_api()

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                api.create_message(
                    messages=[MessageParam(role="user", content="hi")],
                    model_id=self._NO_SAMPLING_MODEL,
                    temperature=0.2,
                )
        warnings = [
            rec for rec in caplog.records if "sampling parameters" in rec.message
        ]
        assert len(warnings) == 1  # once per model, not per call
        assert self._NO_SAMPLING_MODEL in warnings[0].message

    def test_no_warning_for_sampling_model_or_unset(self, caplog: Any) -> None:
        import logging

        api, _ = self._make_api()

        with caplog.at_level(logging.WARNING):
            api.create_message(
                messages=[MessageParam(role="user", content="hi")],
                model_id=self._SAMPLING_MODEL,
                temperature=0.5,
            )
            api.create_message(
                messages=[MessageParam(role="user", content="hi")],
                model_id=self._NO_SAMPLING_MODEL,
            )
        assert not any("sampling parameters" in rec.message for rec in caplog.records)

    def test_kwargs_accepted_by_real_client_signature(self) -> None:
        """Integration guard: every kwarg the SDK sends - for a sampling model
        (temperature in extra_body) and a non-sampling model (no temperature) -
        must be accepted by the REAL installed `anthropic` client signature
        (bound with no network). Catches the SDK forwarding an unsupported
        parameter on whatever anthropic CI resolves."""
        real_client = anthropic.Anthropic(api_key="dummy")
        real_signature = inspect.signature(real_client.beta.messages.create)

        def spy(**kwargs: Any) -> MagicMock:
            real_signature.bind(**kwargs)  # raises on an unsupported keyword
            response = MagicMock()
            response.model_dump.return_value = {"role": "assistant", "content": "hi"}
            return response

        real_client.beta.messages.create = spy  # type: ignore[method-assign]
        api = AnthropicMessagesApi(client=real_client)

        for model_id in (self._SAMPLING_MODEL, self._NO_SAMPLING_MODEL):
            for temperature in (None, 0.0, 0.7):
                result = api.create_message(
                    messages=[MessageParam(role="user", content="hi")],
                    model_id=model_id,
                    temperature=temperature,
                )
                assert isinstance(result, MessageParam)
