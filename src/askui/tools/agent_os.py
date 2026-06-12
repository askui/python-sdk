from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from askui.models.shared.tool_tags import ToolTags

if TYPE_CHECKING:
    from askui.tools.askui.agent_os_target_computer import (
        ComputerTarget,
    )
    from askui.tools.askui.askui_ui_controller_grpc.generated import (
        Controller_V1_pb2 as controller_v1_pbs,
    )
    from askui.tools.askui.askui_ui_controller_grpc.generated.AgentOS_Send_Request_2501 import (  # noqa: E501
        RenderObjectStyle,
    )
    from askui.tools.askui.askui_ui_controller_grpc.generated.AgentOS_Send_Response_2501 import (  # noqa: E501
        GetActiveProcessResponseModel,
        GetActiveWindowResponseModel,
        GetSystemInfoResponseModel,
    )

MouseButton = Literal["left", "middle", "right"]

ModifierKey = Literal[
    "command",
    "alt",
    "control",
    "shift",
    "right_shift",
]
"""Modifier keys for keyboard actions."""

PcKey = Literal[
    "backspace",
    "delete",
    "enter",
    "tab",
    "escape",
    "up",
    "down",
    "right",
    "left",
    "home",
    "end",
    "pageup",
    "pagedown",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "numpad_lock",
    "numpad_0",
    "numpad_1",
    "numpad_2",
    "numpad_3",
    "numpad_4",
    "numpad_5",
    "numpad_6",
    "numpad_7",
    "numpad_8",
    "numpad_9",
    "numpad_+",
    "numpad_-",
    "numpad_*",
    "numpad_/",
    "numpad_.",
    "space",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
]
"""PC keys for keyboard actions."""


class ClickEvent(BaseModel):
    type: Literal["click"] = "click"
    x: int
    y: int
    button: Literal["left", "middle", "right", "unknown"]
    pressed: bool
    injected: bool = False
    timestamp: float


class Coordinate(BaseModel):
    x: int
    y: int


class DisplaySize(BaseModel):
    """Represents the size of a display in pixels."""

    width: int
    height: int


class Display(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
    )

    id: int = Field(validation_alias="displayID")
    name: str | None = Field(default=None, validation_alias="name")
    size: DisplaySize = Field(validation_alias="sizeInPixels")


class DisplaysListResponse(BaseModel):
    data: list[Display] = Field(validation_alias="displays")

    def __str__(self) -> str:
        return ",".join([str(display) for display in self.data])


InputEvent = ClickEvent


class ComputerAgentOS(ABC):
    """
    Abstract base class for Agent OS. Cannot be instantiated directly.

    This class defines the interface for operating system interactions including
    mouse control, keyboard input, and screen capture functionality.
    Implementations should provide concrete functionality for these abstract
    methods.
    """

    @property
    def tags(self) -> list[str]:
        """Get the tags for this agent OS.

        Returns:
            list[str]: A list of tags that identify this agent OS type.
        """
        if not hasattr(self, "_tags"):
            self._tags = [ToolTags.COMPUTER.value]
        return self._tags

    @tags.setter
    def tags(self, tags: list[str]) -> None:
        """Set the tags for this agent OS.

        Args:
            tags (list[str]): A list of tags that identify this agent OS type.
        """
        self._tags = tags

    @abstractmethod
    def connect(self) -> None:
        """
        Establishes a connection to the Agent OS.

        This method is called before performing any OS-level operations.
        It handles any necessary setup or initialization required for the OS
        interaction.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """
        Terminates the connection to the Agent OS.

        This method is called after all OS-level operations are complete.
        It handles any necessary cleanup or resource release.
        """

    @abstractmethod
    def screenshot(self, report: bool = True) -> Image.Image:
        """
        Captures a screenshot of the current display.

        Args:
            report (bool, optional): Whether to include the screenshot in
                reporting. Defaults to `True`.

        Returns:
            Image.Image: A PIL Image object containing the screenshot.
        """
        raise NotImplementedError

    @abstractmethod
    def mouse_move(self, x: int, y: int, duration: int = 500) -> None:
        """
        Moves the mouse cursor to specified screen coordinates.

        Args:
            x (int): The horizontal coordinate (in pixels) to move to.
            y (int): The vertical coordinate (in pixels) to move to.
            duration (int): The duration (in ms) the movement should take.
        """
        raise NotImplementedError

    @abstractmethod
    def type(self, text: str, typing_speed: int = 50) -> None:
        """
        Simulates typing text as if entered on a keyboard.

        Args:
            text (str): The text to be typed.
            typing_speed (int, optional): The speed of typing in characters per
                second. Defaults to `50`.
        """
        raise NotImplementedError

    @abstractmethod
    def click(self, button: MouseButton = "left", count: int = 1) -> None:
        """
        Simulates clicking a mouse button.

        Args:
            button (MouseButton, optional): The mouse
                button to click. Defaults to `"left"`.
            count (int, optional): Number of times to click. Defaults to `1`.
        """
        raise NotImplementedError

    @abstractmethod
    def mouse_down(self, button: MouseButton = "left") -> None:
        """
        Simulates pressing and holding a mouse button.

        Args:
            button (MouseButton, optional): The mouse
                button to press. Defaults to `"left"`.
        """
        raise NotImplementedError

    @abstractmethod
    def mouse_up(self, button: MouseButton = "left") -> None:
        """
        Simulates releasing a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse
                button to release. Defaults to `"left"`.
        """
        raise NotImplementedError

    @abstractmethod
    def mouse_scroll(self, dx: int, dy: int) -> None:
        """
        Simulates scrolling the mouse wheel.

        Args:
            dx (int): The horizontal scroll amount. Positive values scroll right,
                negative values scroll left.
            dy (int): The vertical scroll amount. Positive values scroll down,
                negative values scroll up.
        """
        raise NotImplementedError

    @abstractmethod
    def keyboard_pressed(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Simulates pressing and holding a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to press.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def keyboard_release(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Simulates releasing a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to release.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                release along with the main key. Defaults to `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def keyboard_tap(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> None:
        """
        Simulates pressing and immediately releasing a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to tap.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
            count (int, optional): The number of times to tap the key. Defaults to `1`.
        """
        raise NotImplementedError

    def list_displays(self) -> DisplaysListResponse:
        """
        List all the available displays.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_active_display(self) -> Display:
        """
        Retrieve the currently active display/screen.

        Returns:
            Display: The currently active display/screen.
        """
        raise NotImplementedError

    def set_display(self, display: int = 1) -> None:
        """
        Sets the active display for screen interactions.

        Args:
            display (int, optional): The display ID to set as active.
                Defaults to `1`.
        """
        raise NotImplementedError

    def run_command(self, command: str, timeout_ms: int = 30000) -> None:
        """
        Executes a shell command.

        Args:
            command (str): The command to execute.
            timeout_ms (int, optional): The timeout for command
                execution in milliseconds. Defaults to `30000` (30 seconds).

        """
        raise NotImplementedError

    def start_listening(self) -> None:
        """
        Start listening for mouse and keyboard events.

        IMPORTANT: This method is still experimental and may not work at all and may
        change in the future.
        """
        raise NotImplementedError

    def poll_event(self) -> InputEvent | None:
        """
        Poll for a single input event.

        IMPORTANT: This method is still experimental and may not work at all and may
        change in the future.
        """
        raise NotImplementedError

    def stop_listening(self) -> None:
        """Stop listening for mouse and keyboard events.

        IMPORTANT: This method is still experimental and may not work at all and may
        change in the future.
        """
        raise NotImplementedError

    def get_mouse_position(self) -> Coordinate:
        """
        Get the current mouse cursor position.

        Returns:
            The current mouse position data.
        """
        raise NotImplementedError

    def set_mouse_position(self, x: int, y: int) -> None:
        """
        Set the mouse cursor position to specific coordinates.

        Args:
            x (int): The horizontal coordinate (in pixels) to set the cursor to.
            y (int): The vertical coordinate (in pixels) to set the cursor to.

        """
        raise NotImplementedError

    def render_quad(self, style: "RenderObjectStyle") -> int:
        """
        Render a quad object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the quad.

        Returns:
            Response containing the render object ID.
        """
        raise NotImplementedError

    def render_line(self, style: "RenderObjectStyle", points: list[Coordinate]) -> int:
        """
        Render a line object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the line.
            points (list[Coordinate]): The points defining the line.

        Returns:
            Response containing the render object ID.
        """
        raise NotImplementedError

    def render_image(self, style: "RenderObjectStyle", image_data: str) -> int:
        """
        Render an image object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the image.
            image_data (str): The image data to display.

        Returns:
            Response containing the render object ID.
        """
        raise NotImplementedError

    def render_text(self, style: "RenderObjectStyle", content: str) -> int:
        """
        Render a text object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the text.
            text_content (str): The text content to display.

        Returns:
            Response containing the render object ID.
        """
        raise NotImplementedError

    def update_render_object(self, object_id: int, style: "RenderObjectStyle") -> None:
        """
        Update styling properties of an existing render object.

        Args:
            object_id (int): The ID of the render object to update.
            style (RenderObjectStyle): The new style properties.
        """
        raise NotImplementedError

    def delete_render_object(self, object_id: int) -> None:
        """
        Delete an existing render object from the display.

        Args:
            object_id (int): The ID of the render object to delete.

        Returns:
            Response confirming the deletion.
        """
        raise NotImplementedError

    def clear_render_objects(self) -> None:
        """
        Clear all render objects from the display.

        Returns:
            Response confirming the clearing.
        """
        raise NotImplementedError

    def get_process_list(
        self, get_extended_info: bool = False
    ) -> "controller_v1_pbs.Response_GetProcessList":
        """
        Get a list of running processes.

        Args:
            get_extended_info (bool, optional): Whether to include
                extended process information.
                Defaults to `False`.

        Returns:
            controller_v1_pbs.Response_GetProcessList: Process list response containing:
                - processes: List of ProcessInfo objects
        """
        raise NotImplementedError

    def get_window_list(
        self, process_id: int
    ) -> "controller_v1_pbs.Response_GetWindowList":
        """
        Get a list of windows for a specific process.

        Args:
            process_id (int): The ID of the process to get windows for.

        Returns:
            controller_v1_pbs.Response_GetWindowList: Window list response containing:
                - windows: List of WindowInfo objects with ID and name
        """
        raise NotImplementedError

    def set_mouse_delay(self, delay_ms: int) -> None:
        """
        Configure mouse action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for mouse actions.
        """
        raise NotImplementedError

    def set_keyboard_delay(self, delay_ms: int) -> None:
        """
        Configure keyboard action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for keyboard actions.
        """
        raise NotImplementedError

    def set_active_window(self, process_id: int, window_id: int) -> int:
        """
        Set the active window for automation.
        Adds the window as a virtual display and returns the display ID.
        It raises an error if display length is not increased after adding the window.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.

        Returns:
            int: The new Display ID.

        Raises:
            AskUiControllerError:
            If display length is not increased after adding the window.
        """
        raise NotImplementedError

    def get_system_info(self) -> "GetSystemInfoResponseModel":
        """
        Get the system information.

        Returns:
            GetSystemInfoResponseModel: The system information.
        """
        raise NotImplementedError

    def get_active_process(self) -> "GetActiveProcessResponseModel":
        """
        Get the active process.

        Returns:
            GetActiveProcessResponseModel: The active process.
        """
        raise NotImplementedError

    def set_active_process(self, process_id: int) -> None:
        """
        Set the active process.

        Args:
            process_id (int): The ID of the process to set as active.
        """
        raise NotImplementedError

    def get_active_window(self) -> "GetActiveWindowResponseModel":
        """
        Gets the window id and name in addition to the process id
             and name of the currently active window (in focus).

        Returns:
            GetActiveWindowResponseModel: The active window.
        """
        raise NotImplementedError

    def set_window_in_focus(self, process_id: int, window_id: int) -> None:
        """
        Sets the window with the specified windowId of the process
            with the specified processId active,
            which brings it to the front and gives it focus.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.
        """
        raise NotImplementedError

    def add_agent_os_target_computer(
        self, agent_os_target_computer: "ComputerTarget"
    ) -> "ComputerTarget":
        """Register an additional target computer. Auto-connects if connected."""
        raise NotImplementedError

    def reset_agent_os_target_computers(
        self,
        agent_os_target_computers: "list[ComputerTarget] | None" = None,
    ) -> None:
        """Disconnect (if connected) and replace the target computer list."""
        raise NotImplementedError

    def list_agent_os_target_computers(self) -> "list[ComputerTarget]":
        """Return all registered target computers."""
        raise NotImplementedError

    def get_current_computer_target_id(self, report: bool = True) -> str:
        """Return the `computer_id` of the currently active target computer."""
        raise NotImplementedError

    def switch_agent_os_target_computer(self, computer_id: str) -> "ComputerTarget":
        """Switch the active target computer by its `computer_id`."""
        raise NotImplementedError

    def temporary_select(self, computer_id: str) -> AbstractContextManager[Self]:
        """
        Temporarily switch the active target computer for the duration of a `with`
        block, then restore the previously-active target on exit (even if the
        block raises).

        Args:
            computer_id (str): Computer id of the target to activate inside the
                block.

        Returns:
            AbstractContextManager[Self]: Context manager that yields this
                `ComputerAgentOS` with the selected target active.

        Example:
            ```python
            with agent_os.temporary_select('Remote-Machine') as remote_machine:
                img = remote_machine.screenshot()
                img.save("remote_machine.png")
            # previous active target restored here
            ```
        """
        raise NotImplementedError

    def get_file_names(self, absolute_directory_path: str) -> list[str]:
        """
        List file names in an absolute directory on the automation target
        (desktop Agent OS).

        Args:
            absolute_directory_path (str): Absolute directory path on the target system.

        Returns:
            list[str]: Names of files in that directory.

        Raises:
            NotImplementedError: If the implementation does not support this operation.
        """
        raise NotImplementedError

    def get_file(self, path: str) -> Image.Image | str:
        """
        Read a file from the automation target (desktop Agent OS).

        Binary image payloads are returned as `PIL.Image.Image` when recognized;
        otherwise UTF-8 text when decodable.

        Args:
            path (str): File path on the target system.

        Returns:
            Image.Image | str: Decoded file contents.

        Raises:
            NotImplementedError: If the implementation does not support this operation.
        """
        raise NotImplementedError

    def remove_virtual_displays(self) -> None:
        """
        Remove virtual displays from the controller, leaving real displays only.

        Raises:
            NotImplementedError: If the implementation does not support this operation.
        """
        raise NotImplementedError


AgentOs = ComputerAgentOS
"""Deprecated alias for `ComputerAgentOS`, kept for backward compatibility.

`AgentOs` was renamed to `ComputerAgentOS` to reflect that it is the
computer-specific Agent OS interface (mouse, keyboard, displays, ...) rather
than a universal abstraction across all device types. Prefer `ComputerAgentOS`
in new code.
"""
