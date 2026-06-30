from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerGetCurrentComputerTargetIdTool(ComputerBaseTool):
    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="get_current_computer_target_id",
            description="""
                Return the `computer_id` of the currently active Agent OS target
                computer that agent-os actions are routed to.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        return self.agent_os.get_current_computer_target_id()
