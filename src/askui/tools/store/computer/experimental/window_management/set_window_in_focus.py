from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerSetWindowInFocusTool(ComputerBaseTool):
    """
    Brings a specific window to the foreground and sets it as the active
    focused window. Use this tool to switch focus to a particular window
    before performing automation tasks.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="set_window_in_focus_tool",
            description="""
            Brings a specific window to the foreground and sets it as the active
            focused window. This tool is used to switch focus to a particular
            window before performing automation tasks. The window will be brought
            to the front of all other windows and receive keyboard and mouse
            focus. Use this tool when you need to interact with a specific window
            that may be behind other windows or not currently active. This is
            particularly useful before using add_window_as_virtual_display_tool or
            when you want to ensure a window is visible and ready for user
            interaction or automation.

            Requires: A valid process ID (obtained from list_process_tool) and a
            valid window ID (obtained from list_process_windows_tool) for that
            process. The window ID must belong to the specified process ID.
            Returns: A confirmation message indicating that the window has been
            set in focus.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "window_id": {
                        "type": "integer",
                        "description": """
                        The window ID of the window to bring into focus. This must be
                        a valid window ID that belongs to the specified process ID.
                        Obtain window IDs by using list_process_windows_tool with the
                        process ID.
                        """,
                    },
                    "process_id": {
                        "type": "integer",
                        "description": """
                            The process ID of the application that owns the window.
                            This must be a valid process ID obtained from
                            list_process_tool. The window_id must belong to this
                            process.
                        """,
                    },
                },
                "required": ["window_id", "process_id"],
            },
            agent_os=agent_os,
        )

    def __call__(self, window_id: int, process_id: int) -> str:
        self.agent_os.set_window_in_focus(process_id, window_id)
        return f"Window with id {window_id} in process {process_id} was set in focus."
