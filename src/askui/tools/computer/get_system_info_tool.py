from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerGetSystemInfoTool(ComputerBaseTool):
    """
    Get the system information of the currently active Agent OS target computer.
    This tool returns the system information as a JSON object prefixed with the
    active target computer's id.
    The JSON object contains the following fields:
    - platform: The operating system platform.
    - label: The operating system label.
    - version: The operating system version.
    - architecture: The operating system architecture.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="get_system_info_tool",
            description="""
                Get the system information of the currently active Agent OS target
                computer. This tool returns the system information as a JSON object
                prefixed with the active target computer's id so it is clear which
                computer the info belongs to.
                The JSON object contains the following fields:
                - platform: The operating system platform.
                - label: The operating system label.
                - version: The operating system version.
                - architecture: The operating system architecture.
            """,
            agent_os=agent_os,
        )

    def __call__(self) -> str:
        target_id = self.agent_os.get_current_computer_target_id(report=False)
        system_info_json = self.agent_os.get_system_info().model_dump_json()
        return f"[Computer '{target_id}']: {system_info_json}"
