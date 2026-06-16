"""Integration tests proving secrets are usable but never leak to the LLM."""

import json
from typing import Any

from typing_extensions import override

from askui import ComputerAgent, Secret
from askui.agent_settings import AgentSettings
from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import (
    MessageParam,
    ToolUseBlockParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.reporting import Reporter
from askui.tools.toolbox import AgentToolbox

_SECRET_VALUE = "hunter2-Sup3rSecret"


class _RecordingReporter(Reporter):
    """Reporter that records everything for leak assertions."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []

    @override
    def add_message(self, role: str, content: Any, image: Any = None) -> None:
        self.messages.append((role, content))

    @override
    def add_usage_summary(self, usage: Any) -> None:
        pass

    @override
    def add_cache_execution_statistics(self, original_usage: Any) -> None:
        pass

    @override
    def generate(self) -> None:
        pass


class _TypeSecretVlmProvider(VlmProvider):
    """Fake VLM that types the password placeholder, then finishes.

    Records every ``messages``/``system`` it receives so tests can assert the literal
    secret value never reaches the model.
    """

    def __init__(self) -> None:
        self.received_messages: list[list[dict[str, Any]]] = []
        self.received_systems: list[str | None] = []
        self._call = 0

    @property
    @override
    def model_id(self) -> str:
        return "test-model"

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        max_tokens: int | None = None,
        system: SystemPrompt | None = None,
        thinking: Any = None,
        tool_choice: Any = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        self.received_messages.append([m.model_dump(mode="json") for m in messages])
        self.received_systems.append(str(system) if system is not None else None)
        self._call += 1
        if self._call == 1 and tools is not None:
            type_tool_name = next(
                name
                for name, tool in tools.tool_map.items()
                if getattr(tool, "base_name", None) == "type"
            )
            return MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tool_1",
                        name=type_tool_name,
                        input={"text": "<|secret|>password<|secret|>"},
                    )
                ],
                stop_reason="tool_use",
            )
        return MessageParam(role="assistant", content="done", stop_reason="end_turn")

    def all_text_sent_to_llm(self) -> str:
        return json.dumps(self.received_messages) + json.dumps(self.received_systems)


class TestSecretsDoNotLeakDuringAct:
    def test_secret_is_typed_but_never_reaches_llm_or_reporter(
        self, agent_os_mock: Any, agent_toolbox_mock: AgentToolbox
    ) -> None:
        vlm = _TypeSecretVlmProvider()
        reporter = _RecordingReporter()
        with ComputerAgent(
            tools=agent_toolbox_mock,
            reporters=[reporter],
            settings=AgentSettings(vlm_provider=vlm),
            secrets=[
                Secret(
                    name="password", value=_SECRET_VALUE, description="login password"
                )
            ],
        ) as agent:
            agent.act("Type the password into the field")

        # (a) The OS actually received the real secret value.
        typed = [
            (c.args[0] if c.args else c.kwargs.get("text"))
            for c in agent_os_mock.type.call_args_list
        ]
        assert _SECRET_VALUE in typed

        # (b) The literal value never reached the LLM (messages or system prompt)...
        assert _SECRET_VALUE not in vlm.all_text_sent_to_llm()
        # ...and the model was told about the placeholder.
        assert "<|secret|>password<|secret|>" in (vlm.received_systems[0] or "")
        assert "<AVAILABLE_SECRETS>" in (vlm.received_systems[0] or "")

        # (c) The literal value never reached the reporter.
        assert _SECRET_VALUE not in json.dumps(reporter.messages, default=str)

    def test_literal_secret_in_goal_is_redacted(
        self, agent_toolbox_mock: AgentToolbox
    ) -> None:
        vlm = _TypeSecretVlmProvider()
        with ComputerAgent(
            tools=agent_toolbox_mock,
            settings=AgentSettings(vlm_provider=vlm),
            secrets=[Secret(name="password", value=_SECRET_VALUE)],
        ) as agent:
            agent.act(f"Type {_SECRET_VALUE} into the field")

        sent = vlm.all_text_sent_to_llm()
        assert _SECRET_VALUE not in sent
        assert "<|secret|>password<|secret|>" in sent

    def test_per_call_secret_overrides_agent_secret(
        self, agent_os_mock: Any, agent_toolbox_mock: AgentToolbox
    ) -> None:
        vlm = _TypeSecretVlmProvider()
        with ComputerAgent(
            tools=agent_toolbox_mock,
            settings=AgentSettings(vlm_provider=vlm),
            secrets=[Secret(name="password", value="agent-level-value")],
        ) as agent:
            agent.act(
                "Type the password",
                secrets=[Secret(name="password", value=_SECRET_VALUE)],
            )

        typed = [
            (c.args[0] if c.args else c.kwargs.get("text"))
            for c in agent_os_mock.type.call_args_list
        ]
        assert _SECRET_VALUE in typed
        assert "agent-level-value" not in vlm.all_text_sent_to_llm()


class TestSecretsInDeterministicType:
    def test_type_resolves_placeholder_and_redacts_report(
        self, agent_os_mock: Any, agent_toolbox_mock: AgentToolbox
    ) -> None:
        reporter = _RecordingReporter()
        with ComputerAgent(
            tools=agent_toolbox_mock,
            reporters=[reporter],
            settings=AgentSettings(vlm_provider=_TypeSecretVlmProvider()),
            secrets=[Secret(name="password", value=_SECRET_VALUE)],
        ) as agent:
            agent.type("<|secret|>password<|secret|>")

        typed = [
            (c.args[0] if c.args else c.kwargs.get("text"))
            for c in agent_os_mock.type.call_args_list
        ]
        assert typed == [_SECRET_VALUE]
        assert _SECRET_VALUE not in json.dumps(reporter.messages, default=str)

    def test_type_literal_secret_is_redacted_in_report(
        self, agent_os_mock: Any, agent_toolbox_mock: AgentToolbox
    ) -> None:
        reporter = _RecordingReporter()
        with ComputerAgent(
            tools=agent_toolbox_mock,
            reporters=[reporter],
            settings=AgentSettings(vlm_provider=_TypeSecretVlmProvider()),
            secrets=[Secret(name="password", value=_SECRET_VALUE)],
        ) as agent:
            agent.type(_SECRET_VALUE)

        # OS still gets the real value; reporter shows the placeholder only.
        typed = [
            (c.args[0] if c.args else c.kwargs.get("text"))
            for c in agent_os_mock.type.call_args_list
        ]
        assert typed == [_SECRET_VALUE]
        dumped = json.dumps(reporter.messages, default=str)
        assert _SECRET_VALUE not in dumped
        assert "<|secret|>password<|secret|>" in dumped
