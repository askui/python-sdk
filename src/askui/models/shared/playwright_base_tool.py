from typing import Any

from askui.models.shared.tool_tags import ToolTags
from askui.models.shared.tools import ToolWithAgentOS
from askui.tools.agent_os import ComputerAgentOS
from askui.tools.agent_os_type_error import AgentOsTypeError
from askui.tools.android.agent_os import AndroidAgentOs
from askui.tools.playwright.agent_os import PlaywrightAgentOs


class PlaywrightBaseTool(ToolWithAgentOS):
    """Tool base class that has a Playwright ComputerAgentOS available."""

    def __init__(
        self,
        agent_os: ComputerAgentOS | None = None,
        required_tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            required_tags=[ToolTags.PLAYWRIGHT.value] + (required_tags or []),
            agent_os=agent_os,
            **kwargs,
        )

    @property
    def agent_os(self) -> PlaywrightAgentOs:
        """Get the agent OS.

        Returns:
            PlaywrightAgentOs: The Playwright agent OS instance.
        """
        agent_os = super().agent_os
        if not isinstance(agent_os, PlaywrightAgentOs):
            raise AgentOsTypeError(
                expected_type=PlaywrightAgentOs,
                actual_type=type(agent_os),
            )
        return agent_os

    @agent_os.setter
    def agent_os(
        self, agent_os: ComputerAgentOS | AndroidAgentOs | PlaywrightAgentOs
    ) -> None:
        """Set the agent OS.

        Args:
            agent_os (ComputerAgentOS | AndroidAgentOs | PlaywrightAgentOs): The
                agent OS instance to set.

        Raises:
            TypeError: If the agent OS is not an `PlaywrightAgentOs` instance.
        """
        if not isinstance(agent_os, PlaywrightAgentOs):
            raise AgentOsTypeError(
                expected_type=PlaywrightAgentOs,
                actual_type=type(agent_os),
            )
        self._agent_os = agent_os
