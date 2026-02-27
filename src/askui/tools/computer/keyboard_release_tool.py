from typing import get_args

from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import AgentOs, ModifierKey, PcKey


class ComputerKeyboardReleaseTool(ComputerBaseTool):
    """Computer Keyboard Release Tool"""

    def __init__(self, agent_os: AgentOs | None = None) -> None:
        super().__init__(
            name="keyboard_release",
            description="Release a keyboard key.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to release.",
                        "enum": list(get_args(PcKey)) + list(get_args(ModifierKey)),
                    },
                    "modifier_keys": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(get_args(ModifierKey)),
                        },
                        "description": (
                            "List of modifier keys to release along with the main key."
                        ),
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
    ) -> str:
        self.agent_os.keyboard_release(key, modifier_keys)
        modifier_str = (
            f" with modifiers {', '.join(modifier_keys)}" if modifier_keys else ""
        )
        return f"Key {key} was released{modifier_str}."
