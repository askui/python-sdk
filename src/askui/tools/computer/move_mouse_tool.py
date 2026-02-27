from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade


class ComputerMoveMouseTool(ComputerBaseTool):
    """Computer Mouse Move Tool"""

    def __init__(self, agent_os: ComputerAgentOsFacade | None = None) -> None:
        super().__init__(
            name="move_mouse",
            description="Move the mouse to a specific position.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "The x coordinate of the mouse position as int.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "The y coordinate of the mouse position as int.",
                    },
                },
                "required": ["x", "y"],
            },
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self, x: int, y: int) -> str:
        # for some reason, the agent occasionally calls the tool with the coords
        # encoded as strings, which will lead the tool to failing. To prevent this we
        # will explicitly convert to int here
        x, y = int(x), int(y)
        self.agent_os.mouse_move(x, y)
        return f"Mouse was moved to position ({x}, {y})."
