from typing import get_args

from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs, MouseButton


class ComputerMouseClickTool(ComputerBaseTool):
    """Computer Mouse Click Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="mouse_click",
            description="Click and release the mouse button at the current position.",
            input_schema={
                "type": "object",
                "properties": {
                    "mouse_button": {
                        "type": "string",
                        "description": "The mouse button to click.",
                        "enum": get_args(MouseButton),
                    },
                    "number_of_clicks": {
                        "type": "integer",
                        "description": (
                            "The number of times to click the mouse button."
                            " Defaults to 1"
                        ),
                        "default": 1,
                    },
                },
                "required": ["mouse_button"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self, mouse_button: MouseButton, number_of_clicks: int = 1) -> str:
        self.agent_os.click(mouse_button, number_of_clicks)
        return f"Mouse button {mouse_button} was clicked {number_of_clicks} times."
