from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerListAgentOsTargetComputersTool(ComputerBaseTool):
    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="list_agent_os_target_computers",
            description="""
                List all the registered Agent OS target computers that the agent
                can route actions to. Each target computer has a unique
                `computer_id` that can be used to switch between them.
            """,
            agent_os=agent_os,
        )

    def __call__(self) -> str:
        target_computers = self.agent_os.list_agent_os_target_computers()
        return ",".join(repr(t) for t in target_computers)
