"""Unit tests for Anthropic messages API output_config / thinking handling."""

import inspect
from typing import Any
from unittest.mock import MagicMock

import anthropic
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

    def test_no_temperature_is_omitted(self) -> None:
        result = _parse_to_anthropic_types(tools=None, temperature=None)
        assert result[7] is omit

    def test_temperature_passed_through(self) -> None:
        result = _parse_to_anthropic_types(tools=None, temperature=0.7)
        assert result[7] == 0.7

    def test_temperature_zero_is_preserved(self) -> None:
        # 0.0 is a valid deterministic value and must not be treated as unset.
        result = _parse_to_anthropic_types(tools=None, temperature=0.0)
        assert result[7] == 0.0


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

    def test_temperature_not_in_body_when_unset(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-5",
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        # Never sent as a typed keyword, and no temperature in the body.
        assert "temperature" not in kwargs
        assert kwargs["extra_body"] is omit

    def test_temperature_sent_via_extra_body_when_set(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-5",
            temperature=0.3,
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        # Routed through the request body (not the typed `temperature=` kwarg,
        # which newer clients removed).
        assert "temperature" not in kwargs
        assert kwargs["extra_body"] == {"temperature": 0.3}

    def test_temperature_zero_sent_via_extra_body(self) -> None:
        api, client = self._make_api()

        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-5",
            temperature=0.0,
        )

        kwargs = client.beta.messages.create.call_args.kwargs
        assert kwargs["extra_body"] == {"temperature": 0.0}

    def test_succeeds_on_client_that_rejects_temperature_kwarg(self) -> None:
        """Regression: a client whose create() has no `temperature` parameter
        (and no **kwargs) must not receive it as a keyword - even when a
        temperature is requested. It goes into the request body instead."""

        def create(**kwargs: object) -> MagicMock:
            if "temperature" in kwargs:
                error_msg = "create() got an unexpected keyword argument 'temperature'"
                raise TypeError(error_msg)
            response = MagicMock()
            response.model_dump.return_value = {"role": "assistant", "content": "hi"}
            return response

        client = MagicMock()
        client.beta.messages.create = create
        api = AnthropicMessagesApi(client=client)

        result = api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="claude-sonnet-5",
            temperature=0.5,  # even when set, must not become a kwarg
        )
        assert isinstance(result, MessageParam)

    def test_kwargs_accepted_by_real_client_signature(self) -> None:
        """Integration guard: every kwarg the SDK sends - with and without a
        temperature - must be accepted by the REAL installed `anthropic` client's
        `beta.messages.create` signature (bound with no network). This catches
        the SDK forwarding a parameter the installed client version does not
        support, on whatever anthropic CI resolves."""
        real_client = anthropic.Anthropic(api_key="dummy")
        real_signature = inspect.signature(real_client.beta.messages.create)

        def spy(**kwargs: Any) -> MagicMock:
            # Raises TypeError if the SDK sends an unsupported keyword.
            real_signature.bind(**kwargs)
            response = MagicMock()
            response.model_dump.return_value = {"role": "assistant", "content": "hi"}
            return response

        real_client.beta.messages.create = spy  # type: ignore[method-assign]
        api = AnthropicMessagesApi(client=real_client)

        for temperature in (None, 0.0, 0.7):
            result = api.create_message(
                messages=[MessageParam(role="user", content="hi")],
                model_id="claude-sonnet-5",
                temperature=temperature,
            )
            assert isinstance(result, MessageParam)
