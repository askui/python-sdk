from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerListProcessWindowsTool(ComputerBaseTool):
    """
    Lists all windows belonging to a specific process. Use this tool to discover
    available windows within an application after obtaining the process ID from
    list_process_tool.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="list_process_windows_tool",
            description="""
            Lists all windows belonging to a specific process identified by its
            process ID. This tool is used after list_process_tool to discover all
            windows within a particular application. Use this tool when you need to
            see what windows are available for a specific process, such as when an
            application has multiple windows (main window, dialogs, popups, etc.).
            The tool returns the window names and window IDs for each window
            belonging to the specified process. After obtaining a window ID, you
            can use set_window_in_focus_tool to bring that window to the front, or
            add_window_as_virtual_display_tool to add it as a virtual display for
            automation.

            Requires: A valid process ID obtained from list_process_tool.
            Returns: A formatted string listing all windows for the specified
            process, including window name and window ID for each window.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "integer",
                        "description": (
                            "The process ID of the application whose windows should be"
                            " listed. This must be a valid process ID obtained from"
                            " list_process_tool."
                        ),
                    },
                },
                "required": ["process_id"],
            },
            agent_os=agent_os,
        )

    def __call__(self, process_id: int) -> str:
        # the agent occassionally calls this tool with a string i/o an int
        # this would lead to a tool error. to prevent this, we will convert to int
        # manually here
        process_id = int(process_id)
        get_window_list_result = self.agent_os.get_window_list(process_id)
        return (
            f"The Process ID: {process_id} has the"
            f" following windows: {str(get_window_list_result)}"
        )
