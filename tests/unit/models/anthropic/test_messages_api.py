"""Unit tests for Anthropic messages API output_config / thinking handling."""

from unittest.mock import MagicMock

from anthropic import omit

from askui.models.anthropic.messages_api import (
    AnthropicMessagesApi,
    _parse_to_anthropic_types,
)
from askui.models.shared.agent_message_param import MessageParam


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
