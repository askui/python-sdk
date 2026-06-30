from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerListProcessTool(ComputerBaseTool):
    """
    Lists all running processes on the computer that have at least one window.
    This is the first step in the window management workflow to discover available
    applications and their process IDs.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="list_process_tool",
            description="""
            Lists all running processes on the computer that have at least one window.
            This tool is used as the first step in window management workflows to
            discover available applications and obtain their process IDs. Use this
            tool when you need to find a specific application or see what
            applications are currently running with windows. The tool returns a list
            of process names and their corresponding process IDs. After obtaining a
            process ID, you can use list_process_windows_tool to see all windows
            belonging to that process, or use set_window_in_focus_tool or
            add_window_as_virtual_display_tool to interact with specific windows.

            Returns: A formatted string listing all processes with windows,
            including process name and process ID for each entry.
            """,
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        result = self.agent_os.get_process_list(get_extended_info=True)
        processes_with_window = [
            process for process in result.processes if process.extendedInfo.hasWindow
        ]

        if not processes_with_window:
            return "No processes with a window found"

        process_information = [
            (
                f"Process with name: {process.name}  ID: {process.ID}"
                f" and has window: {process.extendedInfo.hasWindow}"
            )
            for process in processes_with_window
        ]

        return (
            f"The processes on the computer with a window are:"
            f" {', '.join(process_information)}"
        )
