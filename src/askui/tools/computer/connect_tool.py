from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerConnectTool(ComputerBaseTool):
    """Computer Connect Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="connect",
            description=(
                "Connect to the agent OS controller to enable computer control. "
                "Useful for establishing an initial connection or reconnecting "
                "after controller errors. If already connected, this will "
                "disconnect and reconnect, which may cause previous configuration "
                "to be lost (e.g., selected display must be reconfigured)."
                "Must be used once before any other computer tools are used."
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        try:
            self.agent_os.disconnect()
        finally:
            self.agent_os.connect()
            return "Agent OS controller was connected."  # noqa: B012
