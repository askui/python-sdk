from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerListDisplaysTool(ComputerBaseTool):
    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="list_displays",
            description="""
                List all the available displays on the currently active Agent OS
                target computer. The result is prefixed with the active target
                computer's id so it is clear which computer the displays belong to.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        target_id = self.agent_os.get_current_computer_target_id(report=False)
        displays_json = self.agent_os.list_displays().model_dump_json(
            exclude={"data": {"__all__": {"size"}}},
        )
        return f"[Computer '{target_id}']: {displays_json}"
