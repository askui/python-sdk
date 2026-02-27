from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs


class ComputerTypeTool(ComputerBaseTool):
    """Computer Type Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="type",
            description="Type text on the computer.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type.",
                    },
                    "typing_speed": {
                        "type": "integer",
                        "description": (
                            "The speed of typing in characters per minute."
                            " Defaults to 50"
                        ),
                        "default": 50,
                    },
                },
                "required": ["text"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self, text: str, typing_speed: int = 50) -> str:
        self.agent_os.type(text, typing_speed)
        return (
            f"Text was typed: {text} with typing speed: "
            f" {typing_speed} characters per minute."
        )
