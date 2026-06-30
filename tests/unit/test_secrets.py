"""Unit tests for the secret scope (`Secret` / `SecretVault`)."""

import logging

import pytest
from pydantic import SecretStr, ValidationError
from typing_extensions import override

from askui import Secret, SecretVault
from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import Tool, ToolCallResult, ToolCollection


class _EchoTool(Tool):
    """Tool that echoes its input back (to exercise output redaction)."""

    @override
    def __call__(self, text: str = "") -> ToolCallResult:
        return f"you typed: {text}"


class TestSecret:
    def test_accepts_plain_str_and_stores_as_secret_str(self) -> None:
        secret = Secret(name="password", value="hunter2value")
        assert isinstance(secret.value, SecretStr)
        assert secret.value.get_secret_value() == "hunter2value"

    def test_accepts_secret_str(self) -> None:
        secret = Secret(name="password", value=SecretStr("hunter2value"))
        assert isinstance(secret.value, SecretStr)
        assert secret.value.get_secret_value() == "hunter2value"

    def test_value_is_masked_in_repr_and_dump(self) -> None:
        secret = Secret(name="password", value="hunter2value")
        assert "hunter2value" not in repr(secret)
        assert "hunter2value" not in secret.model_dump_json()

    def test_placeholder(self) -> None:
        placeholder = "<|secret|>password<|secret|>"
        assert Secret(name="password", value="x").placeholder == placeholder

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            Secret(name="", value="x")

    @pytest.mark.parametrize(
        "name", ["password", "PIN_2", "token0", "with space", "with-dash"]
    )
    def test_name_has_no_charset_restriction(self, name: str) -> None:
        assert Secret(name=name, value="x").name == name


class TestSubstitute:
    def test_replaces_placeholder_in_str(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        assert vault.substitute("<|secret|>password<|secret|>") == "hunter2"

    def test_replaces_within_surrounding_text(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        assert (
            vault.substitute("pw is <|secret|>password<|secret|>!") == "pw is hunter2!"
        )

    def test_walks_nested_structures(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        result = vault.substitute(
            {
                "text": "<|secret|>password<|secret|>",
                "items": ["<|secret|>password<|secret|>", 1],
            }
        )
        assert result == {"text": "hunter2", "items": ["hunter2", 1]}

    def test_tolerates_whitespace_in_placeholder_name(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        assert vault.substitute("<|secret|> password <|secret|>") == "hunter2"

    def test_unknown_placeholder_left_intact(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        unknown = "<|secret|>unknown<|secret|>"
        assert vault.substitute(unknown) == unknown

    def test_empty_vault_is_noop_same_object(self) -> None:
        vault = SecretVault()
        obj = {"a": "b"}
        assert vault.substitute(obj) is obj

    def test_does_not_mutate_input(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        original = {"text": "<|secret|>password<|secret|>"}
        vault.substitute(original)
        assert original == {"text": "<|secret|>password<|secret|>"}


class TestRedact:
    def test_replaces_literal_value_with_placeholder(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        assert vault.redact("typed hunter2") == "typed <|secret|>password<|secret|>"

    def test_warns_when_redacting(self, caplog: pytest.LogCaptureFixture) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        with caplog.at_level(logging.WARNING):
            vault.redact("typed hunter2")
        assert "Redacted secret 'password'" in caplog.text
        # The value itself must never be logged.
        assert "hunter2" not in caplog.text

    def test_short_values_not_redacted(self) -> None:
        vault = SecretVault([Secret(name="pin", value="12")])
        assert vault.redact("code 12 shown") == "code 12 shown"

    def test_walks_nested_structures(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        result = vault.redact(["typed hunter2", {"k": "hunter2"}])
        assert result == [
            "typed <|secret|>password<|secret|>",
            {"k": "<|secret|>password<|secret|>"},
        ]


class TestRedactMessage:
    def test_redacts_str_content(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        msg = MessageParam(role="user", content="login with hunter2")
        assert (
            vault.redact_message(msg).content
            == "login with <|secret|>password<|secret|>"
        )

    def test_redacts_text_and_tool_result_blocks(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        msg = MessageParam(
            role="user",
            content=[
                TextBlockParam(text="value hunter2"),
                ToolResultBlockParam(
                    tool_use_id="t1",
                    content=[TextBlockParam(text="Typed hunter2")],
                ),
                ToolResultBlockParam(tool_use_id="t2", content="echo hunter2"),
            ],
        )
        redacted = vault.redact_message(msg)
        assert isinstance(redacted.content, list)
        text_block, tool_result_block, echo_block = redacted.content
        assert isinstance(text_block, TextBlockParam)
        assert text_block.text == "value <|secret|>password<|secret|>"
        assert isinstance(tool_result_block, ToolResultBlockParam)
        assert isinstance(tool_result_block.content, list)
        inner_block = tool_result_block.content[0]
        assert isinstance(inner_block, TextBlockParam)
        assert inner_block.text == "Typed <|secret|>password<|secret|>"
        assert isinstance(echo_block, ToolResultBlockParam)
        assert echo_block.content == "echo <|secret|>password<|secret|>"

    def test_redacts_tool_use_input(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        msg = MessageParam(
            role="assistant",
            content=[
                ToolUseBlockParam(id="t1", name="type", input={"text": "hunter2"})
            ],
        )
        redacted = vault.redact_message(msg)
        assert isinstance(redacted.content, list)
        tool_use_block = redacted.content[0]
        assert isinstance(tool_use_block, ToolUseBlockParam)
        assert tool_use_block.input == {"text": "<|secret|>password<|secret|>"}

    def test_does_not_mutate_original_message(self) -> None:
        vault = SecretVault([Secret(name="password", value="hunter2")])
        msg = MessageParam(role="user", content="hunter2 here")
        vault.redact_message(msg)
        assert msg.content == "hunter2 here"


class TestMergeAndSystemPrompt:
    def test_merge_precedence_and_names(self) -> None:
        base = SecretVault([Secret(name="password", value="old")])
        override = SecretVault(
            [Secret(name="password", value="new"), Secret(name="pin", value="1234")]
        )
        merged = base.merge(override)
        assert set(merged.names) == {"password", "pin"}
        assert merged.substitute("<|secret|>password<|secret|>") == "new"

    def test_system_prompt_section_lists_placeholders(self) -> None:
        vault = SecretVault(
            [Secret(name="password", value="hunter2", description="login password")]
        )
        section = vault.system_prompt_section()
        assert "<AVAILABLE_SECRETS>" in section
        assert "<|secret|>password<|secret|>" in section
        assert "login password" in section
        assert "Example" in section
        assert "hunter2" not in section

    def test_system_prompt_section_empty_vault(self) -> None:
        assert SecretVault().system_prompt_section() == ""

    def test_bool(self) -> None:
        assert not SecretVault()
        assert SecretVault([Secret(name="a", value="bbbb")])


class TestToolOutputRedaction:
    def _echo_tool(self) -> _EchoTool:
        return _EchoTool(
            name="echo",
            description="echo text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    def test_tool_output_echoing_secret_is_redacted(self) -> None:
        tool = self._echo_tool()
        tools = ToolCollection(
            tools=[tool],
            secret_vault=SecretVault([Secret(name="password", value="hunter2xyz")]),
        )
        block = ToolUseBlockParam(
            id="t1", name=tool.name, input={"text": "<|secret|>password<|secret|>"}
        )

        results = tools.run([block])

        # The tool received the decoded value (substitution) and echoed it, but the
        # output is redacted before it becomes a tool result.
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert isinstance(result.content, list)
        text_block = result.content[0]
        assert isinstance(text_block, TextBlockParam)
        assert "hunter2xyz" not in text_block.text
        assert text_block.text == "you typed: <|secret|>password<|secret|>"
