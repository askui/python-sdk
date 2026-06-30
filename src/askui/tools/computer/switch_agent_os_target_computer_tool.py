from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerSwitchAgentOsTargetComputerTool(ComputerBaseTool):
    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="switch_agent_os_target_computer",
            description="""
                Switch the active Agent OS target computer by its `computer_id`.
                Future agent-os actions are routed to the newly selected target
                computer. Use `list_agent_os_target_computers` to discover the
                available computer ids.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "computer_id": {
                        "type": "string",
                    },
                },
                "required": ["computer_id"],
            },
            agent_os=agent_os,
        )

    def __call__(self, computer_id: str) -> str:
        return repr(self.agent_os.switch_agent_os_target_computer(computer_id))
