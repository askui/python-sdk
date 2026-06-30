"""Example: a custom tool restricted to one device type (computer only).

A tool that needs to drive a device should subclass one of the device-specific
base classes instead of `Tool`:

- `ComputerBaseTool` - typed `self.agent_os` (a `ComputerAgentOS`); the tool is
  tagged `"computer"` and can only be bound to a computer/desktop target.
- `AndroidBaseTool` - typed `self.agent_os` (an `AndroidAgentOs`); the tool is
  tagged `"android"` and can only be bound to an Android target.

When `act()` runs, the SDK binds each tool to the first registered agent OS
whose tags contain all of the tool's `required_tags`. `ComputerBaseTool` sets
`required_tags=["computer"]`, so the tool below is never handed to an Android
device. This matters most with `MultiDeviceAgent`, which registers both a
computer and an Android agent OS in the same `act()` call.

Required environment variables (see .env):
- ASKUI_WORKSPACE_ID, ASKUI_TOKEN - for the default AskUI model stack
"""

import logging

from askui import ComputerAgent
from askui.models.shared import ComputerBaseTool
from askui.tools.askui import LocalComputerTarget, RemoteComputerTarget

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


class ComputerScreenSizeTool(ComputerBaseTool):
    """Reports the pixel size of the active computer screen.

    Subclassing `ComputerBaseTool` tags this tool as `"computer"`, so it is only
    ever bound to a computer (desktop) agent OS - never to an Android device.
    `self.agent_os` is therefore a `ComputerAgentOS`.
    """

    def __init__(self) -> None:
        super().__init__(
            name="get_screen_size",
            description=(
                "Return the width and height in pixels of the active computer screen."
            ),
            input_schema={"type": "object", "properties": {}},
        )

    def __call__(self) -> str:
        screenshot = self.agent_os.screenshot()
        return f"{screenshot.width}x{screenshot.height}"


class ScreenSizeOfMachineTool(ComputerBaseTool):
    """Reports the screen size of one specific computer target (auto-switch).

    The tag system only restricts a tool to a device *type* (computer vs
    Android), not to an individual machine. To bind a tool to one specific
    target, have it auto-switch to that target inside `__call__`:
    `self.agent_os.temporary_select(computer_id)` activates the given target for
    the duration of the block and restores the previously active one on exit
    (even if the body raises). So the tool always acts on its machine without
    disturbing the rest of the run.
    """

    def __init__(self, computer_id: str) -> None:
        super().__init__(
            name="get_screen_size_of_machine",
            description=(
                "Return the screen size of the machine this tool is bound to."
            ),
            input_schema={"type": "object", "properties": {}},
        )
        self._computer_id = computer_id

    def __call__(self) -> str:
        with self.agent_os.temporary_select(self._computer_id):
            screenshot = self.agent_os.screenshot()
            return f"{screenshot.width}x{screenshot.height}"


def computer_only_tool_with_computer_agent() -> None:
    """Use the computer-scoped tool with a plain `ComputerAgent`."""
    with ComputerAgent() as agent:
        agent.act(
            "Report the current screen size using the get_screen_size tool",
            tools=[ComputerScreenSizeTool()],
        )


def computer_only_tool_with_multi_device_agent() -> None:
    """Show that the tool is routed only to the computer in a multi-device run.

    Requires the `android` dependency (`pip install askui[android]`) and a
    connected Android device/emulator.
    """
    from askui import MultiDeviceAgent

    with MultiDeviceAgent(android_device_sn="emulator-5554") as agent:
        agent.act(
            "Read the screen size on the computer, then take a screenshot on "
            "the phone",
            # ComputerScreenSizeTool is given only to the computer agent OS.
            # An AndroidBaseTool subclass would be given only to the device.
            tools=[ComputerScreenSizeTool()],
        )


def tool_pinned_to_a_specific_machine() -> None:
    """Bind a tool to one specific target machine via auto-switch.

    `ScreenSizeOfMachineTool` always measures "remote-box", even though
    "local-box" is the active target. The remote example expects an Agent OS
    controller reachable at the configured address; adjust it to your setup.
    """
    with ComputerAgent(
        agent_os_target_computers=[
            LocalComputerTarget(computer_id="local-box"),
            RemoteComputerTarget(
                address="192.168.1.42:26000",
                description="Remote box",
                computer_id="remote-box",
            ),
        ],
    ) as agent:
        agent.act(
            "Report the screen size of the remote machine",
            tools=[ScreenSizeOfMachineTool(computer_id="remote-box")],
        )


if __name__ == "__main__":
    computer_only_tool_with_computer_agent()
    # computer_only_tool_with_multi_device_agent()
    # tool_pinned_to_a_specific_machine()

    logger.info("Done!")
