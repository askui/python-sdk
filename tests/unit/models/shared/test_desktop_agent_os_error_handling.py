"""Tests that Desktop Agent OS errors are routed by recoverability.

The Desktop Agent OS raises two error types:

- `DesktopAgentOsException` for failures the agent can react to (e.g. reading a
  path that does not exist). The tool-calling loop catches it and surfaces it to
  the agent as a tool error result so the run can continue.
- `DesktopAgentOsError` for unfixable protocol violations. It derives from
  `AutomationError` and is re-raised by the tool-calling loop, terminating the
  run instead of being fed back to the agent.
"""

import pytest

from askui.models.exceptions import AutomationError
from askui.models.shared.agent_message_param import (
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import Tool, ToolCollection
from askui.tools.askui.askui_ui_controller_grpc.desktop_agent_os_error import (
    DesktopAgentOsError,
    DesktopAgentOsException,
)

_RECOVERABLE_MESSAGE = (
    "directory_iterator::directory_iterator: The system cannot find the "
    'path specified.: "FrontEnd\\Traces"'
)
_FATAL_MESSAGE = "unexpected response type: <broken>"


class _RaisingTool(Tool):
    """A tool whose `__call__` raises the exception it was constructed with."""

    _error: BaseException

    def __init__(self, error: BaseException) -> None:
        super().__init__(
            name="raising_tool",
            description="Raises a preconfigured Desktop Agent OS error.",
        )
        self._error = error

    def __call__(self) -> str:
        raise self._error


def _run(tool: Tool) -> list:
    collection = ToolCollection(tools=[tool])
    tool_use = ToolUseBlockParam(id="tool_use_1", input={}, name=tool.name)
    return collection.run([tool_use])


class TestDesktopAgentOsErrorHierarchy:
    def test_error_is_an_automation_error(self) -> None:
        assert issubclass(DesktopAgentOsError, AutomationError)

    def test_exception_is_a_plain_exception_not_automation_error(self) -> None:
        assert issubclass(DesktopAgentOsException, Exception)
        assert not issubclass(DesktopAgentOsException, AutomationError)


class TestDesktopAgentOsErrorHandling:
    def test_recoverable_exception_returns_error_result(self) -> None:
        results = _run(_RaisingTool(DesktopAgentOsException(_RECOVERABLE_MESSAGE)))

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert result.is_error is True
        assert result.tool_use_id == "tool_use_1"
        assert "FrontEnd\\Traces" in str(result.content)

    def test_fatal_error_propagates_and_terminates(self) -> None:
        with pytest.raises(DesktopAgentOsError):
            _run(_RaisingTool(DesktopAgentOsError(_FATAL_MESSAGE)))
