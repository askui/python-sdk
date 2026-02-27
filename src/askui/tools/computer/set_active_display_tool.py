from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerSetActiveDisplayTool(ComputerBaseTool):
    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="set_active_display",
            description="""
                Set the display screen from which screenshots are taken and on which
                actions are performed.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "display_id": {
                        "type": "integer",
                    },
                },
                "required": ["display_id"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self, display_id: int) -> None:
        self.agent_os.set_display(display_id)
