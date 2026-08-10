"""Tests that tool failures are surfaced to the agent instead of crashing.

When a tool raises, the tool-calling loop is expected to catch the error and
return a `ToolResultBlockParam` with `is_error=True` so the agent can react to
it. This only works if the raised exception derives from `Exception`; a
`BaseException` subclass would slip past the `except Exception` handler and
crash the run instead. `DesktopAgentOsError` (raised e.g. when reading a remote
file/directory that does not exist) must therefore behave like a regular
`Exception`.
"""

from askui.models.shared.agent_message_param import (
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import Tool, ToolCollection
from askui.tools.askui.askui_ui_controller_grpc.desktop_agent_os_error import (
    DesktopAgentOsError,
)


class _RaisingTool(Tool):
    """A tool whose `__call__` always raises a `DesktopAgentOsError`."""

    def __init__(self) -> None:
        super().__init__(
            name="raising_tool",
            description="Always raises a DesktopAgentOsError.",
        )

    def __call__(self) -> str:
        raise DesktopAgentOsError(self._error_message)

    _error_message = (
        "directory_iterator::directory_iterator: The system cannot find the "
        'path specified.: "FrontEnd\\Traces"'
    )


class TestDesktopAgentOsErrorHandling:
    def test_desktop_agent_os_error_is_an_exception(self) -> None:
        assert issubclass(DesktopAgentOsError, Exception)

    def test_raising_tool_returns_error_result_instead_of_crashing(self) -> None:
        tool = _RaisingTool()
        collection = ToolCollection(tools=[tool])
        tool_use = ToolUseBlockParam(
            id="tool_use_1",
            input={},
            name=tool.name,
        )

        results = collection.run([tool_use])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert result.is_error is True
        assert result.tool_use_id == "tool_use_1"
        assert "FrontEnd\\Traces" in str(result.content)
