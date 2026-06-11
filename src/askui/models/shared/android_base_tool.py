from typing import Any

from askui.models.shared.tool_tags import ToolTags
from askui.models.shared.tools import ToolWithAgentOS
from askui.tools import ComputerAgentOS
from askui.tools.agent_os_type_error import AgentOsTypeError
from askui.tools.android.agent_os import AndroidAgentOs


class AndroidBaseTool(ToolWithAgentOS):
    """Tool base class that has an AndroidAgentOs available."""

    def __init__(
        self,
        agent_os: AndroidAgentOs | None = None,
        required_tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            required_tags=[ToolTags.ANDROID.value] + (required_tags or []),
            agent_os=agent_os,
            **kwargs,
        )

    @property
    def agent_os(self) -> AndroidAgentOs:
        """Get the agent OS.

        Returns:
            AndroidAgentOs: The agent OS instance.

        Raises:
            TypeError: If the agent OS is not an AndroidAgentOs instance.
        """
        agent_os = super().agent_os
        if not isinstance(agent_os, AndroidAgentOs):
            raise AgentOsTypeError(
                expected_type=AndroidAgentOs,
                actual_type=type(agent_os),
            )
        return agent_os

    @agent_os.setter
    def agent_os(self, agent_os: ComputerAgentOS | AndroidAgentOs) -> None:
        """Set the agent OS.

        Args:
            agent_os (ComputerAgentOS | AndroidAgentOs): The agent OS instance to set.

        Raises:
            TypeError: If the agent OS is not an AndroidAgentOs instance.
        """
        if not isinstance(agent_os, AndroidAgentOs):
            raise AgentOsTypeError(
                expected_type=AndroidAgentOs,
                actual_type=type(agent_os),
            )
        self._agent_os = agent_os
