import logging
import warnings
from typing import Annotated, overload

from pydantic import ConfigDict, Field, validate_call

from askui.agent_base import Agent
from askui.agent_settings import AgentSettings
from askui.callbacks import ConversationCallback
from askui.container import telemetry
from askui.locators.locators import Locator
from askui.models.models import Point
from askui.models.shared.secrets import Secret
from askui.models.shared.settings import ActSettings, MessageSettings
from askui.models.shared.tools import Tool
from askui.models.shared.truncation_strategies import TruncationStrategy
from askui.prompts.act_prompts import create_android_agent_prompt
from askui.tools.android.agent_os import ANDROID_KEY, AndroidDisplay
from askui.tools.android.agent_os_facade import AndroidAgentOsFacade
from askui.tools.android.ppadb_agent_os import PpadbAgentOs
from askui.tools.android.tools import (
    AndroidDragAndDropTool,
    AndroidGetConnectedDevicesSerialNumbersTool,
    AndroidGetConnectedDisplaysInfosTool,
    AndroidGetCurrentConnectedDeviceInfosTool,
    AndroidKeyCombinationTool,
    AndroidKeyTapEventTool,
    AndroidScreenshotTool,
    AndroidSelectDeviceBySerialNumberTool,
    AndroidSelectDisplayByUniqueIDTool,
    AndroidShellTool,
    AndroidSwipeTool,
    AndroidTapTool,
    AndroidTypeTool,
)
from askui.tools.exception_tool import ExceptionTool

from .reporting import CompositeReporter, Reporter
from .retry import Retry

logger = logging.getLogger(__name__)


class AndroidAgent(Agent):
    """
    A vision-based agent that can interact with Android devices through computer vision and AI.

    This agent can perform various UI interactions on Android devices like tapping, typing, swiping, and more.
    It uses computer vision models to locate UI elements and execute actions on them.

    Args:
        device (str | int, optional): The Android device to connect to. Can be either a serial number (as a `str`) or an index (as an `int`) representing the position in the `adb devices` list. Index `0` refers to the first device. Defaults to `0`.
        display (AndroidDisplay | list[AndroidDisplay] | int | str | None, optional): Which display to drive. An `AndroidDisplay` pins that exact display (bypassing auto-detection, so you control the input/screencap `-d` ids). A `list[AndroidDisplay]` becomes the authoritative set of selectable displays — the first is active and the model may switch among them at runtime with correct ids. An `int` selects by index and a `str` by name, both via auto-detection. `None` (default) auto-detects and selects the first display.
        display_allow_switching (bool, optional): When `False`, the runtime display/device selection tools are removed so a pinned `display` cannot be changed mid-run. Defaults to `True`.
        reporters (list[Reporter] | None, optional): List of reporter instances for logging and reporting. If `None`, an empty list is used.
        settings (AgentSettings | None, optional): Provider-based model settings. If `None`, uses the default AskUI model stack.
        retry (Retry, optional): The retry instance to use for retrying failed actions. Defaults to `ConfigurableRetry` with exponential backoff. Currently only supported for `locate()` method.
        act_tools (list[Tool] | None, optional): Additional tools to make available for the `act()` method.
        secrets (list[Secret] | None, optional): Sensitive values (e.g. passwords) the agent may use but the LLM must never see. The model only sees the placeholder `<|secret|>NAME<|secret|>`; the real value is substituted at execution time and kept out of the LLM prompt, reporter, logs and cache. Also usable in deterministic `type()` and overridable per call via `act(..., secrets=[...])`. Note: a secret typed into a visible field may still appear in screenshots sent to the model; on-screen secrets cannot currently be hidden.

    Example:
        ```python
        from askui import AndroidAgent

        with AndroidAgent() as agent:
            agent.tap("Submit button")
            agent.type("Hello World")
            agent.act("Open settings menu")
        ```
    """

    @telemetry.record_call(
        exclude={
            "reporters",
            "settings",
            "act_tools",
            "callbacks",
            "truncation_strategy",
            "secrets",
        }
    )
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        device: str | int = 0,
        display: "AndroidDisplay | list[AndroidDisplay] | int | str | None" = None,
        display_allow_switching: bool = True,
        reporters: list[Reporter] | None = None,
        settings: AgentSettings | None = None,
        retry: Retry | None = None,
        act_tools: list[Tool] | None = None,
        callbacks: list[ConversationCallback] | None = None,
        truncation_strategy: TruncationStrategy | None = None,
        secrets: list[Secret] | None = None,
    ) -> None:
        reporter = CompositeReporter(reporters=reporters)
        self.os = PpadbAgentOs(
            device_identifier=device, display=display, reporter=reporter
        )
        default_tools = self._apply_display_tool_policy(
            self.get_default_tools(), display_allow_switching
        )
        super().__init__(
            reporter=reporter,
            retry=retry,
            tools=default_tools + (act_tools or []),
            agent_os=self.os,
            settings=settings,
            callbacks=callbacks,
            truncation_strategy=truncation_strategy,
            secrets=secrets,
        )
        self.act_agent_os_facade = AndroidAgentOsFacade(
            self.os,
            coordinate_space=self._vlm_provider.coordinate_space,
            image_scaler=self._vlm_provider.image_scaler,
        )
        self.act_tool_collection.add_agent_os(self.act_agent_os_facade)
        # Override default act settings with Android-specific settings
        self.act_settings = ActSettings(
            messages=MessageSettings(
                system=create_android_agent_prompt(),
                thinking={"type": "disabled"},
                temperature=0.0,
            ),
        )

    @overload
    def tap(
        self,
        target: str | Locator,
    ) -> None: ...

    @overload
    def tap(
        self,
        target: Point,
    ) -> None: ...

    @telemetry.record_call(exclude={"locator"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def tap(
        self,
        target: str | Locator | Point,
    ) -> None:
        """
        Taps on the specified target.

        Args:
            target (str | Locator | Point): The target to tap on. Can be a locator, a point, or a string.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.tap("Submit button")
                agent.tap((100, 100))
        """
        msg = "tap"
        if isinstance(target, tuple):
            msg += f" at ({target[0]}, {target[1]})"
            self._reporter.add_message("User", msg)
            self.os.tap(target[0], target[1])
        else:
            msg += f" on {target}"
            self._reporter.add_message("User", msg)
            logger.debug(
                "AndroidAgent received instruction to click",
                extra={"target": target},
            )
            point = self._locate(locator=target)[0]
            self.os.tap(point[0], point[1])

    @telemetry.record_call(exclude={"text"})
    @validate_call
    def type(
        self,
        text: Annotated[str, Field(min_length=1)],
    ) -> None:
        """
        Types the specified text as if it were entered on a keyboard.

        Args:
            text (str): The text to be typed. Must be at least `1` character long.
            Only ASCII printable characters are supported. other characters will raise an error.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.type("Hello, world!")  # Types "Hello, world!"
                agent.type("user@example.com")  # Types an email address
                agent.type("password123")  # Types a password
            ```
        """
        # Reporter sees the placeholder; the device receives the resolved value.
        self._reporter.add_message("User", f'type: "{self._redact_secrets(text)}"')
        logger.debug("AndroidAgent received instruction to type")
        self.os.type(self._resolve_secrets(text))

    @telemetry.record_call()
    @validate_call
    def key_tap(
        self,
        key: ANDROID_KEY,
    ) -> None:
        """
        Taps the specified key on the Android device.

        Args:
            key (ANDROID_KEY): The key to tap.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.key_tap("HOME")  # Taps the home key
                agent.key_tap("BACK")  # Taps the back key
            ```
        """
        self.os.key_tap(key)

    @telemetry.record_call()
    @validate_call
    def key_combination(
        self,
        keys: Annotated[list[ANDROID_KEY], Field(min_length=2)],
        duration_in_ms: int = 100,
    ) -> None:
        """
        Taps the specified keys on the Android device.

        Args:
            keys (list[ANDROID_KEY]): The keys to tap.
            duration_in_ms (int, optional): The duration in milliseconds to hold the key combination. Default is 100ms.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.key_combination(["HOME", "BACK"])  # Taps the home key and then the back key
                agent.key_combination(["HOME", "BACK"], duration_in_ms=200)  # Taps the home key and then the back key for 200ms.
            ```
        """
        self._reporter.add_message(
            "User",
            f"key_combination(keys=[{', '.join(keys)}], duration_in_ms={duration_in_ms})",
        )
        self.os.key_combination(keys, duration_in_ms)

    @telemetry.record_call()
    @validate_call
    def shell(
        self,
        command: str,
    ) -> str:
        """
        Executes a shell command on the Android device.

        Args:
            command (str): The shell command to execute.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.shell("pm list packages")  # Lists all installed packages
                agent.shell("dumpsys battery")  # Displays battery information
            ```
        """
        self._reporter.add_message("User", f"shell(command='{command}')")
        return self.os.shell(command)

    @telemetry.record_call()
    @validate_call
    def drag_and_drop(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_in_ms: int = 1000,
    ) -> None:
        """
        Drags and drops the specified target.

        Args:
            x1 (int): The x-coordinate of the starting point.
            y1 (int): The y-coordinate of the starting point.
            x2 (int): The x-coordinate of the ending point.
            y2 (int): The y-coordinate of the ending point.
            duration_in_ms (int, optional): The duration in milliseconds to hold the drag and drop. Default is 1000ms.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.drag_and_drop(100, 100, 200, 200)  # Drags and drops from (100, 100) to (200, 200)
                agent.drag_and_drop(100, 100, 200, 200, duration_in_ms=2000)  # Drags and drops from (100, 100) to (200, 200) with a 2000ms duration
        """
        self._reporter.add_message(
            "User",
            f"drag_and_drop(x1={x1}, y1={y1}, x2={x2}, y2={y2}, duration_in_ms={duration_in_ms})",
        )
        self.os.drag_and_drop(x1, y1, x2, y2, duration_in_ms)

    @telemetry.record_call()
    @validate_call
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_in_ms: int = 1000,
    ) -> None:
        """
        Swipes the specified target.

        Args:
            x1 (int): The x-coordinate of the starting point.
            y1 (int): The y-coordinate of the starting point.
            x2 (int): The x-coordinate of the ending point.
            y2 (int): The y-coordinate of the ending point.
            duration_in_ms (int, optional): The duration in milliseconds to hold the swipe. Default is 1000ms.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.swipe(100, 100, 200, 200)  # Swipes from (100, 100) to (200, 200)
                agent.swipe(100, 100, 200, 200, duration_in_ms=2000)  # Swipes from (100, 100) to (200, 200) with a 2000ms duration
        """
        self._reporter.add_message(
            "User",
            f"swipe(x1={x1}, y1={y1}, x2={x2}, y2={y2}, duration_in_ms={duration_in_ms})",
        )
        self.os.swipe(x1, y1, x2, y2, duration_in_ms)

    @telemetry.record_call(
        exclude={"device_sn"},
    )
    @validate_call
    def set_device_by_serial_number(
        self,
        device_sn: str,
    ) -> None:
        """
        Sets the active device for screen interactions by name.

        Args:
            device_sn (str): The serial number of the device to set as active.

        Example:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.set_device_by_serial_number("Pixel 6")  # Sets the active device to the Pixel 6
        """
        self._reporter.add_message(
            "User",
            f"set_device_by_serial_number(device_sn='{device_sn}')",
        )
        self.os.set_device_by_serial_number(device_sn)

    @staticmethod
    def _apply_display_tool_policy(
        tools: list[Tool], display_allow_switching: bool
    ) -> list[Tool]:
        """Drop the display/device *mutating* tools when switching is disabled.

        The read-only display-info tools are kept. Device selection is included
        because selecting a device resets the active display to index 0, which
        would undo a pinned `display`.
        """
        if display_allow_switching:
            return tools
        return [
            t
            for t in tools
            if not isinstance(
                t,
                (
                    AndroidSelectDisplayByUniqueIDTool,
                    AndroidSelectDeviceBySerialNumberTool,
                ),
            )
        ]

    @staticmethod
    def get_default_tools() -> list[Tool]:
        return [
            AndroidScreenshotTool(),
            AndroidTapTool(),
            AndroidTypeTool(),
            AndroidDragAndDropTool(),
            AndroidKeyTapEventTool(),
            AndroidSwipeTool(),
            AndroidKeyCombinationTool(),
            AndroidShellTool(),
            AndroidSelectDeviceBySerialNumberTool(),
            AndroidSelectDisplayByUniqueIDTool(),
            AndroidGetConnectedDevicesSerialNumbersTool(),
            AndroidGetConnectedDisplaysInfosTool(),
            AndroidGetCurrentConnectedDeviceInfosTool(),
            ExceptionTool(),
        ]


class AndroidVisionAgent(AndroidAgent):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        warnings.warn(
            "AndroidVisionAgent is deprecated, use AndroidAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
