from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerRemoveVirtualDisplaysTool(ComputerBaseTool):
    """
    Removes virtual displays so only physical displays remain active.

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.computer.experimental import (
            ComputerRemoveVirtualDisplaysTool,
        )

        with ComputerAgent(act_tools=[ComputerRemoveVirtualDisplaysTool()]) as agent:
            agent.act("Remove virtual displays so only physical screens are active")

        with ComputerAgent() as agent:
            agent.act(
                "Remove virtual displays so only physical screens are active",
                tools=[ComputerRemoveVirtualDisplaysTool()],
            )
        ```
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="remove_virtual_displays_tool",
            description=(
                "Removes all virtual displays from the current display "
                "configuration, keeping only physical screens. Use after workflows "
                "that attach windows as virtual displays (for example "
                "add_window_as_virtual_display_tool) to restore a normal setup before "
                "continuing automation."
            ),
            agent_os=agent_os,
        )

        self.is_cacheable = True

    def __call__(self) -> str:
        self.agent_os.remove_virtual_displays()
        return "Removed virtual displays; only physical displays remain active."
