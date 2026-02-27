from typing import get_args

from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs, ModifierKey, PcKey


class ComputerKeyboardTapTool(ComputerBaseTool):
    """Computer Keyboard Tap Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="keyboard_tap",
            description="Tap (press and release) a keyboard key.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to tap.",
                        "enum": list(get_args(PcKey)) + list(get_args(ModifierKey)),
                    },
                    "modifier_keys": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(get_args(ModifierKey)),
                        },
                        "description": (
                            "List of modifier keys to press along with the main key."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            "The number of times to tap the key. Defaults to 1"
                        ),
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "required": ["key"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> str:
        self.agent_os.keyboard_tap(key, modifier_keys, count)
        modifier_str = (
            f" with modifiers {', '.join(modifier_keys)}" if modifier_keys else ""
        )
        count_str = f" {count} time{'s' if count != 1 else ''}"
        return f"Key {key} was tapped{modifier_str}{count_str}."
