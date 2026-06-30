from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerRetrieveActiveDisplayTool(ComputerBaseTool):
    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="retrieve_active_display",
            description="""
                Retrieve the currently active display on the currently active Agent
                OS target computer. The display is used to take screenshots and
                perform actions. The result is prefixed with the active target
                computer's id so it is clear which computer the display belongs to.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        target_id = self.agent_os.get_current_computer_target_id(report=False)
        display_json = self.agent_os.retrieve_active_display().model_dump_json(
            exclude={"size"}
        )
        return f"[Computer '{target_id}']: {display_json}"
