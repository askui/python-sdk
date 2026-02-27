from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade


class ComputerGetMousePositionTool(ComputerBaseTool):
    """Computer Get Mouse Position Tool"""

    def __init__(self, agent_os: ComputerAgentOsFacade | None = None) -> None:
        super().__init__(
            name="get_mouse_position",
            description="Get the current mouse position.",
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        cursor_position = self.agent_os.get_mouse_position()
        return f"Mouse is at position ({cursor_position.x}, {cursor_position.y})."
