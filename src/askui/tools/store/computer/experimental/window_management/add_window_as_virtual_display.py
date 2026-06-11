from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerAddWindowAsVirtualDisplayTool(ComputerBaseTool):
    """
    Converts a specific window into a virtual display that can be used for automation.
    This tool assigns a display ID to the window, enabling it to be used as a target
    for UI automation tasks.
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="add_window_as_virtual_display_tool",
            description="""
            Converts a specific window into a virtual display and assigns it a
            unique display ID. This tool is essential for automating windows that
            are not the primary display or when you need to automate a specific
            window in a multi-window application. After adding a window as a
            virtual display, you can use the returned display ID to target that
            specific window for automation tasks using computer_tool or other
            automation tools.

            Typical workflow:
            1. Use list_process_tool to find the process ID of the target
               application
            2. Use list_process_windows_tool to find the window ID of the
               specific window
            3. Optionally use set_window_in_focus_tool to bring the window to
               the front
            4. Use this tool to add the window as a virtual display and obtain
               its display ID
            5. Use the returned display ID with automation tools to interact
               with that window

            Requires: A valid process ID (from list_process_tool), a valid
            window ID (from list_process_windows_tool), and the window name (for
            identification purposes). The window ID must belong to the specified
            process ID.
            Returns: A message containing the assigned display ID, which should
            be used to target this window in subsequent automation operations.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "window_id": {
                        "type": "integer",
                        "description": (
                            "The window ID of the window to convert into a virtual"
                            " display. This must be a valid window ID that belongs"
                            " to the specified process ID. Obtain window IDs by"
                            " using list_process_windows_tool with the process ID."
                        ),
                    },
                    "process_id": {
                        "type": "integer",
                        "description": (
                            "The process ID of the application that owns the"
                            " window to convert. This must be a valid process ID"
                            " obtained from list_process_tool. The window_id must"
                            " belong to this process."
                        ),
                    },
                    "window_name": {
                        "type": "string",
                        "description": (
                            "The name of the window to add as a virtual display."
                            " This should match the window name obtained from"
                            " list_process_windows_tool. It is used for"
                            " identification and confirmation purposes in the"
                            " response message."
                        ),
                    },
                },
                "required": ["window_id", "process_id", "window_name"],
            },
            agent_os=agent_os,
        )

    def __call__(self, window_id: int, process_id: int, window_name: str) -> str:
        display_id = self.agent_os.set_active_window(process_id, window_id)
        return f"""
            Window {window_name} with id {window_id} in process {process_id}
            was added as a virtual display with the display id {display_id}.
            To automate it,  set it in focus using the set_window_in_focus_tool
            and select it using the display id {display_id}.
        """
