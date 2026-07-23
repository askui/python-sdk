from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import List, Literal

from PIL import Image
from typing_extensions import Self

from askui.tools.android.uiautomator_hierarchy import UIElementCollection

ANDROID_KEY = Literal[  # pylint: disable=C0103
    "HOME",
    "BACK",
    "CALL",
    "ENDCALL",
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
    "STAR",
    "POUND",
    "DPAD_UP",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_CENTER",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "POWER",
    "CAMERA",
    "CLEAR",
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
    "COMMA",
    "PERIOD",
    "ALT_LEFT",
    "ALT_RIGHT",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "TAB",
    "SPACE",
    "SYM",
    "EXPLORER",
    "ENVELOPE",
    "ENTER",
    "DEL",
    "GRAVE",
    "MINUS",
    "EQUALS",
    "LEFT_BRACKET",
    "RIGHT_BRACKET",
    "BACKSLASH",
    "SEMICOLON",
    "APOSTROPHE",
    "SLASH",
    "AT",
    "NUM",
    "HEADSETHOOK",
    "FOCUS",
    "PLUS",
    "MENU",
    "NOTIFICATION",
    "SEARCH",
    "MEDIA_PLAY_PAUSE",
    "MEDIA_STOP",
    "MEDIA_NEXT",
    "MEDIA_PREVIOUS",
    "MEDIA_REWIND",
    "MEDIA_FAST_FORWARD",
    "MUTE",
    "PAGE_UP",
    "PAGE_DOWN",
    "SWITCH_CHARSET",
    "ESCAPE",
    "FORWARD_DEL",
    "CTRL_LEFT",
    "CTRL_RIGHT",
    "CAPS_LOCK",
    "SCROLL_LOCK",
    "FUNCTION",
    "BREAK",
    "MOVE_HOME",
    "MOVE_END",
    "INSERT",
    "FORWARD",
    "MEDIA_PLAY",
    "MEDIA_PAUSE",
    "MEDIA_CLOSE",
    "MEDIA_EJECT",
    "MEDIA_RECORD",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "NUM_LOCK",
    "NUMPAD_0",
    "NUMPAD_1",
    "NUMPAD_2",
    "NUMPAD_3",
    "NUMPAD_4",
    "NUMPAD_5",
    "NUMPAD_6",
    "NUMPAD_7",
    "NUMPAD_8",
    "NUMPAD_9",
    "NUMPAD_DIVIDE",
    "NUMPAD_MULTIPLY",
    "NUMPAD_SUBTRACT",
    "NUMPAD_ADD",
    "NUMPAD_DOT",
    "NUMPAD_COMMA",
    "NUMPAD_ENTER",
    "NUMPAD_EQUALS",
    "NUMPAD_LEFT_PAREN",
    "NUMPAD_RIGHT_PAREN",
    "VOLUME_MUTE",
    "INFO",
    "CHANNEL_UP",
    "CHANNEL_DOWN",
    "ZOOM_IN",
    "ZOOM_OUT",
    "WINDOW",
    "GUIDE",
    "BOOKMARK",
    "CAPTIONS",
    "SETTINGS",
    "APP_SWITCH",
    "LANGUAGE_SWITCH",
    "CONTACTS",
    "CALENDAR",
    "MUSIC",
    "CALCULATOR",
    "ASSIST",
    "BRIGHTNESS_DOWN",
    "BRIGHTNESS_UP",
    "MEDIA_AUDIO_TRACK",
    "SLEEP",
    "WAKEUP",
    "PAIRING",
    "MEDIA_TOP_MENU",
    "LAST_CHANNEL",
    "TV_DATA_SERVICE",
    "VOICE_ASSIST",
    "HELP",
    "NAVIGATE_PREVIOUS",
    "NAVIGATE_NEXT",
    "NAVIGATE_IN",
    "NAVIGATE_OUT",
    "DPAD_UP_LEFT",
    "DPAD_DOWN_LEFT",
    "DPAD_UP_RIGHT",
    "DPAD_DOWN_RIGHT",
    "MEDIA_SKIP_FORWARD",
    "MEDIA_SKIP_BACKWARD",
    "MEDIA_STEP_FORWARD",
    "MEDIA_STEP_BACKWARD",
    "SOFT_SLEEP",
    "CUT",
    "COPY",
    "PASTE",
    "ALL_APPS",
    "REFRESH",
]


class AndroidDisplay:
    """A selectable Android display.

    ``display_id`` is the logical DisplayManager id consumed by ``input``
    (tap/swipe/type/keyevent). ``unique_display_id`` is the physical display id
    consumed by ``screencap``. Either may be ``None``, in which case the
    corresponding shell command runs with no ``-d`` flag and therefore targets
    adb's default display (display 0).

    Passing ``None`` for ``display_id`` is only correct when the target screen
    IS the default display; on a multi-display setup where the agent switches
    displays, provide the real logical id instead (a ``None`` input flag ignores
    which display is currently active). Likewise, a display with
    ``unique_display_id=None`` cannot be selected by the model at runtime — the
    ``select_display_by_unique_id`` tool needs a concrete unique id — so use it
    only for a pinned, non-switchable display.
    """

    def __init__(
        self,
        unique_display_id: int | None,
        display_name: str,
        display_id: int | None,
    ) -> None:
        self.unique_display_id: int | None = unique_display_id
        self.display_name: str = display_name
        self.display_id: int | None = display_id

    def __repr__(self) -> str:
        return (
            f"AndroidDisplay(unique_display_id={self.unique_display_id}, "
            f"display_name={self.display_name}, display_id={self.display_id})"
        )

    def __str__(self) -> str:
        # Model-facing rendering (used by the display-info tools). Only surface a
        # unique id the model can hand back to select_display_by_unique_id; a
        # None id means "default display" and is not selectable.
        if self.unique_display_id is None:
            return f"display '{self.display_name}' (default display)"
        return (
            f"display '{self.display_name}' "
            f"(unique_display_id={self.unique_display_id})"
        )

    def get_display_id_flag(self) -> str:
        """
        Returns the display ID flag for input/tap/swipe/type shell commands.

        Returns:
            str: ``-d {display_id}``, or an empty string when ``display_id`` is
            ``None`` (input then targets adb's default display).
        """
        return "" if self.display_id is None else f"-d {self.display_id}"

    def get_display_unique_id_flag(self) -> str:
        """
        Returns the display unique ID flag for the screencap shell command.

        Returns:
            str: ``-d {unique_display_id}``, or an empty string when
            ``unique_display_id`` is ``None`` (screencap then targets the default
            display).
        """
        return (
            ""
            if self.unique_display_id is None
            else f"-d {self.unique_display_id}"
        )


class SingleAndroidDisplay(AndroidDisplay):
    """
    Single display when there is only one display connected.

    Both ids are ``None`` so input and screencap run without a ``-d`` flag
    (adb's default display).
    """

    def __init__(self, display_name: str) -> None:
        super().__init__(None, display_name, None)


class UnknownAndroidDisplay(SingleAndroidDisplay):
    """
    Fallback display for when the Agent OS is not able to determine the displays.
    """

    def __init__(self) -> None:
        super().__init__("Unknown")


class AndroidAgentOs(ABC):
    """
    Abstract base class for Android Agent OS. Cannot be instantiated directly.

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
            self._tags = ["android"]
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
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """
        Terminates the connection to the Agent OS.

        This method is called after all OS-level operations are complete.
        It handles any necessary cleanup or resource release.
        """
        raise NotImplementedError

    @abstractmethod
    def screenshot(self) -> Image.Image:
        """
        Captures a screenshot of the current display.

        Returns:
            Image.Image: A PIL Image object containing the screenshot.
        """
        raise NotImplementedError

    @abstractmethod
    def type(self, text: str) -> None:
        """
        Simulates typing text as if entered on a keyboard.

        Args:
            text (str): The text to be typed.
        """
        raise NotImplementedError

    @abstractmethod
    def tap(self, x: int, y: int) -> None:
        """
        Simulates tapping a screen at specified coordinates.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse
                button to click. Defaults to `"left"`.
            count (int, optional): Number of times to click. Defaults to `1`.
        """
        raise NotImplementedError

    @abstractmethod
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_in_ms: int = 1000,
    ) -> None:
        """
        Simulates swiping a screen from one point to another.

        Args:
            x1 (int): The horizontal coordinate of the start point.
            y1 (int): The vertical coordinate of the start point.
            x2 (int): The horizontal coordinate of the end point.
            y2 (int): The vertical coordinate of the end point.
            duration_in_ms (int, optional): The duration of the swipe in
                milliseconds. Defaults to `1000`.
        """
        raise NotImplementedError

    @abstractmethod
    def drag_and_drop(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_in_ms: int = 1000,
    ) -> None:
        """
        Simulates dragging and dropping an object from one point to another.

        Args:
            x1 (int): The horizontal coordinate of the start point.
            y1 (int): The vertical coordinate of the start point.
            x2 (int): The horizontal coordinate of the end point.
            y2 (int): The vertical coordinate of the end point.
            duration_in_ms (int, optional): The duration of the drag and drop in
                milliseconds. Defaults to `1000`.
        """
        raise NotImplementedError

    @abstractmethod
    def shell(self, command: str) -> str:
        """
        Executes a shell command on the Android device.
        """
        raise NotImplementedError

    @abstractmethod
    def key_tap(self, key: ANDROID_KEY) -> None:
        """
        Simulates a key event on the Android device.
        """
        raise NotImplementedError

    @abstractmethod
    def key_combination(
        self, keys: List[ANDROID_KEY], duration_in_ms: int = 100
    ) -> None:
        """
        Simulates a key combination on the Android device.

        Args:
            keys (List[ANDROID_KEY]): The keys to be pressed.
            duration_in_ms (int, optional): The duration of the key combination in
                milliseconds. Defaults to `100`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_display_by_index(self, display_index: int = 0) -> None:
        """
        Sets the active display for screen interactions by index.
        """
        raise NotImplementedError

    @abstractmethod
    def set_display_by_id(self, display_id: int) -> None:
        """
        Sets the active display for screen interactions by id.
        """
        raise NotImplementedError

    @abstractmethod
    def set_display_by_unique_id(self, display_unique_id: int) -> None:
        """
        Sets the active display for screen interactions by unique id.
        """
        raise NotImplementedError

    @abstractmethod
    def set_display_by_name(self, display_name: str) -> None:
        """
        Sets the active display for screen interactions by name.
        """
        raise NotImplementedError

    @abstractmethod
    def set_device_by_index(self, device_index: int = 0) -> None:
        """
        Sets the active device for screen interactions by index.
        """
        raise NotImplementedError

    @abstractmethod
    def set_device_by_serial_number(self, device_sn: str) -> None:
        """
        Sets the active device for screen interactions by serial number.
        """
        raise NotImplementedError

    @abstractmethod
    def get_connected_displays(self) -> list[AndroidDisplay]:
        """
        Gets the connected displays for screen interactions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_connected_devices_serial_numbers(self) -> list[str]:
        """
        Gets the connected devices serial numbers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_selected_device_infos(self) -> tuple[str, AndroidDisplay]:
        """
        Gets the selected device infos.
        """
        raise NotImplementedError

    @abstractmethod
    def connect_adb_client(self) -> None:
        """
        Connects the adb client to the server.
        """
        raise NotImplementedError

    @abstractmethod
    def push(self, local_path: str, remote_path: str) -> None:
        """
        Pushes a file to the device.
        """
        raise NotImplementedError

    @abstractmethod
    def pull(self, remote_path: str, local_path: str) -> None:
        """
        Pulls a file from the device.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ui_elements(self) -> UIElementCollection:
        """
        Gets the UI elements.
        """
        raise NotImplementedError

    def temporary_select(self, device_sn: str) -> AbstractContextManager[Self]:
        """
        Temporarily switch the active device for the duration of a `with` block,
        then restore the previously-active device on exit (even if the block
        raises).

        Args:
            device_sn (str): Serial number of the device to activate inside the
                block.

        Returns:
            AbstractContextManager[Self]: Context manager that yields this
                `AndroidAgentOs` with `device_sn` active.

        Example:
            ```python
            with android_agent_os.temporary_select('table_phone') as table_phone:
                table_phone.tap(100, 200)
            # previous active device restored here
            ```
        """
        raise NotImplementedError
