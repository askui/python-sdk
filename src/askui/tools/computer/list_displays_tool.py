from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerListDisplaysTool(ComputerBaseTool):
    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="list_displays",
            description="""
                List all the available displays on the computer.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        return self.agent_os.list_displays().model_dump_json(
            exclude={"data": {"__all__": {"size"}}},
        )
