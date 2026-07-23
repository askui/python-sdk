"""Unit tests for the OpenAIMessagesApi message_transform hook."""

from typing import Any
from unittest.mock import MagicMock

from askui.models.openai.messages_api import OpenAIMessagesApi
from askui.models.shared.agent_message_param import MessageParam


def _fake_completion(content: str = "ok") -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


def _client_capturing(captured: dict[str, Any]) -> MagicMock:
    client = MagicMock()

    def create(**kwargs: Any) -> MagicMock:
        captured["messages"] = kwargs["messages"]
        return _fake_completion()

    client.chat.completions.create.side_effect = create
    return client


def test_message_transform_is_applied_before_send() -> None:
    captured: dict = {}
    client = _client_capturing(captured)

    def transform(messages: list[dict]) -> list[dict]:
        return [*messages, {"role": "user", "content": "SHIM"}]

    api = OpenAIMessagesApi(client=client, message_transform=transform)
    api.create_message(messages=[MessageParam(role="user", content="hi")], model_id="m")

    assert captured["messages"][-1] == {"role": "user", "content": "SHIM"}


def test_no_transform_sends_messages_unchanged() -> None:
    captured: dict = {}
    client = _client_capturing(captured)

    api = OpenAIMessagesApi(client=client)
    api.create_message(messages=[MessageParam(role="user", content="hi")], model_id="m")

    assert captured["messages"] == [{"role": "user", "content": "hi"}]
