from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerRetrieveActiveDisplayTool(ComputerBaseTool):
    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="retrieve_active_display",
            description="""
                Retrieve the currently active display on the computer.
                The display is used to take screenshots and perform actions.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        return str(
            self.agent_os.retrieve_active_display().model_dump_json(exclude={"size"})
        )
