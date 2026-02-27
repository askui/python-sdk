from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerDisconnectTool(ComputerBaseTool):
    """Computer Disconnect Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="disconnect",
            description=(
                "Disconnect from the agent OS controller. "
                "Needs to be used once you are done with the task and want to stop"
                " the agent OS controller from running. and clean up resources."
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        self.agent_os.disconnect()
        return "Agent OS controller was disconnected."
