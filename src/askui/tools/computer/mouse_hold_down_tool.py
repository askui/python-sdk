from typing import get_args

from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs, MouseButton


class ComputerMouseHoldDownTool(ComputerBaseTool):
    """Computer Mouse Hold Down Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="mouse_hold_down",
            description="Hold down the mouse button at the current position.",
            input_schema={
                "type": "object",
                "properties": {
                    "mouse_button": {
                        "type": "string",
                        "description": "The mouse button to hold down.",
                        "enum": get_args(MouseButton),
                    },
                },
                "required": ["mouse_button"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self, mouse_button: MouseButton) -> str:
        self.agent_os.mouse_down(mouse_button)
        return f"Mouse button {mouse_button} was held down."
