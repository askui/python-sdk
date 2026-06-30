from unittest.mock import MagicMock

import pytest

from askui.tools.agent_os import ComputerAgentOS
from askui.tools.askui.agent_os_target_computer import RemoteComputerTarget
from askui.tools.computer import (
    ComputerGetCurrentComputerTargetIdTool,
    ComputerListAgentOsTargetComputersTool,
    ComputerSwitchAgentOsTargetComputerTool,
)


@pytest.fixture
def fake_agent_os() -> MagicMock:
    """A MagicMock that passes `isinstance(x, ComputerAgentOS)` checks."""
    return MagicMock(spec=ComputerAgentOS)


class TestComputerListAgentOsTargetComputersTool:
    def test_tool_name(self, fake_agent_os: MagicMock) -> None:
        tool = ComputerListAgentOsTargetComputersTool(agent_os=fake_agent_os)
        assert tool.base_name == "list_agent_os_target_computers"

    def test_returns_newline_joined_reprs(self, fake_agent_os: MagicMock) -> None:
        a = RemoteComputerTarget(
            address="1.1.1.1:23000", description="a", computer_id="a"
        )
        b = RemoteComputerTarget(
            address="2.2.2.2:23000", description="b", computer_id="b"
        )
        fake_agent_os.describe_agent_os_target_computers.return_value = [
            repr(a),
            repr(b),
        ]
        tool = ComputerListAgentOsTargetComputersTool(agent_os=fake_agent_os)
        out = tool()
        assert out == f"{a!r}\n{b!r}"

    def test_empty_list_yields_empty_string(self, fake_agent_os: MagicMock) -> None:
        fake_agent_os.describe_agent_os_target_computers.return_value = []
        tool = ComputerListAgentOsTargetComputersTool(agent_os=fake_agent_os)
        assert tool() == ""


class TestComputerSwitchAgentOsTargetComputerTool:
    def test_tool_name(self, fake_agent_os: MagicMock) -> None:
        tool = ComputerSwitchAgentOsTargetComputerTool(agent_os=fake_agent_os)
        assert tool.base_name == "switch_agent_os_target_computer"

    def test_input_schema_requires_computer_id(self, fake_agent_os: MagicMock) -> None:
        tool = ComputerSwitchAgentOsTargetComputerTool(agent_os=fake_agent_os)
        schema = tool.input_schema
        assert "computer_id" in schema["properties"]
        assert schema["required"] == ["computer_id"]

    def test_call_delegates_to_switch_agent_os_target_computer(
        self, fake_agent_os: MagicMock
    ) -> None:
        switched = RemoteComputerTarget(
            address="1.1.1.1:23000", description="new", computer_id="new"
        )
        fake_agent_os.switch_agent_os_target_computer.return_value = switched
        tool = ComputerSwitchAgentOsTargetComputerTool(agent_os=fake_agent_os)
        out = tool(computer_id="new")
        fake_agent_os.switch_agent_os_target_computer.assert_called_once_with("new")
        assert out == repr(switched)


class TestComputerGetCurrentComputerTargetIdTool:
    def test_tool_name(self, fake_agent_os: MagicMock) -> None:
        tool = ComputerGetCurrentComputerTargetIdTool(agent_os=fake_agent_os)
        assert tool.base_name == "get_current_computer_target_id"

    def test_call_returns_current_computer_id(self, fake_agent_os: MagicMock) -> None:
        fake_agent_os.get_current_computer_target_id.return_value = "a"
        tool = ComputerGetCurrentComputerTargetIdTool(agent_os=fake_agent_os)
        out = tool()
        fake_agent_os.get_current_computer_target_id.assert_called_once_with()
        assert out == "a"
