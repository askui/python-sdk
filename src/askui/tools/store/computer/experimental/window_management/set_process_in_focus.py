from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerSetProcessInFocusTool(ComputerBaseTool):
    """
    Brings a process into focus. The process itself decides which window to bring
    to focus. Use this tool when you want to activate a process and let the
    operating system or the process determine which window should be focused.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="set_process_in_focus_tool",
            description="""
            Brings a process into focus. When you use this tool, it activates the
            specified process and brings it to the foreground. The process itself
            (or the operating system) decides which window belonging to that process
            should be brought to focus. This is useful when you want to activate a
            process but don't need to control which specific window becomes active
            - the system will automatically choose the appropriate window (typically
            the main window or the most recently active window of that process).

            This is different from set_window_in_focus_tool, which allows you to
            specify exactly which window should be brought to focus by providing
            both a process ID and a window ID. Use set_process_in_focus_tool when
            you only care about activating the process and are fine with the system
            choosing the window, or when you don't know which window ID to target.

            Requires: A valid process ID obtained from list_process_tool. The
            process must be running and have at least one window.
            Returns: A confirmation message indicating that the process has been
            set in focus.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "integer",
                        "description": (
                            "The process ID of the application to bring into focus."
                            " This must be a valid process ID obtained from"
                            " list_process_tool. The process must be running and have"
                            " at least one window. The process will decide which"
                            " window to bring to focus."
                        ),
                    },
                },
                "required": ["process_id"],
            },
            agent_os=agent_os,
        )

    def __call__(self, process_id: int) -> str:
        self.agent_os.set_active_process(process_id)
        return f"Process with id {process_id} was set in focus."
