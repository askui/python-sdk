"""Unit tests for `AndroidTapTool`.

Includes a regression test for models emitting integer tool arguments as JSON
strings, e.g. `{"x": "726", "y": "122", "repeat": "1", "repeat_delay_in_ms":
"50"}`. Calling the tool through `ToolCollection` must coerce these instead of
failing with `'<' not supported between instances of 'str' and 'int'`.
"""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from askui.models.shared.agent_message_param import (
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import ToolCollection
from askui.tools.android.agent_os_facade import AndroidAgentOsFacade
from askui.tools.android.tools import AndroidTapTool


@pytest.fixture
def agent_os() -> MagicMock:
    return MagicMock(spec=AndroidAgentOsFacade)


@pytest.fixture
def tap_tool(agent_os: MagicMock, mocker: MockerFixture) -> AndroidTapTool:
    mocker.patch("askui.tools.android.tools.time.sleep")
    return AndroidTapTool(agent_os=agent_os)


class TestAndroidTapTool:
    def test_taps_with_integer_arguments(
        self, tap_tool: AndroidTapTool, agent_os: MagicMock
    ) -> None:
        result = tap_tool(x=726, y=122, repeat=1, repeat_delay_in_ms=50)

        agent_os.tap.assert_called_once_with(726, 122)
        assert result == "Tapped at (726, 122)"

    def test_rejects_negative_delay(self, tap_tool: AndroidTapTool) -> None:
        with pytest.raises(ValueError, match="Delay between taps"):
            tap_tool(x=1, y=1, repeat_delay_in_ms=-1)

    def test_rejects_zero_repeat(self, tap_tool: AndroidTapTool) -> None:
        with pytest.raises(ValueError, match="Number of taps"):
            tap_tool(x=1, y=1, repeat=0)

    def test_string_arguments_from_model_are_coerced(
        self, tap_tool: AndroidTapTool, agent_os: MagicMock
    ) -> None:
        collection = ToolCollection(tools=[tap_tool])
        tool_use = ToolUseBlockParam(
            id="tool_use_1",
            input={"x": "726", "y": "122", "repeat": "2", "repeat_delay_in_ms": "50"},
            name=tap_tool.name,
        )

        results = collection.run([tool_use])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ToolResultBlockParam)
        assert not result.is_error
        assert "Tapped at (726, 122)" in str(result.content)
        assert agent_os.tap.call_count == 2
        agent_os.tap.assert_called_with(726, 122)
