import logging
import warnings
from typing import Annotated, Literal, Optional

from pydantic import ConfigDict, Field, validate_call

from askui.agent_base import Agent
from askui.agent_settings import AgentSettings
from askui.container import telemetry
from askui.locators.locators import Locator
from askui.models.models import Point
from askui.models.shared.conversation_callback import ConversationCallback
from askui.models.shared.settings import ActSettings, LocateSettings, MessageSettings
from askui.models.shared.tools import Tool
from askui.prompts.act_prompts import (
    create_computer_agent_prompt,
)
from askui.tools.computer import (
    ComputerGetMousePositionTool,
    ComputerGetSystemInfoTool,
    ComputerKeyboardPressedTool,
    ComputerKeyboardReleaseTool,
    ComputerKeyboardTapTool,
    ComputerListDisplaysTool,
    ComputerMouseClickTool,
    ComputerMouseHoldDownTool,
    ComputerMouseReleaseTool,
    ComputerMouseScrollTool,
    ComputerMoveMouseTool,
    ComputerRetrieveActiveDisplayTool,
    ComputerScreenshotTool,
    ComputerSetActiveDisplayTool,
    ComputerTypeTool,
)
from askui.tools.exception_tool import ExceptionTool

from .reporting import CompositeReporter, Reporter
from .retry import Retry
from .tools import AgentToolbox, ComputerAgentOsFacade, ModifierKey, PcKey
from .tools.askui import AskUiControllerClient

logger = logging.getLogger(__name__)


class ComputerAgent(Agent):
    """
    A vision-based agent that can interact with user interfaces through computer vision and AI.

    This agent can perform various UI interactions like clicking, typing, scrolling, and more.
    It uses computer vision models to locate UI elements and execute actions on them.

    Args:
        display (int, optional): The display number to use for screen interactions. Defaults to `1`.
        reporters (list[Reporter] | None, optional): List of reporter instances for logging and reporting. If `None`, an empty list is used.
        tools (AgentToolbox | None, optional): Custom toolbox instance. If `None`, a default one will be created with `AskUiControllerClient`.
        settings (AgentSettings | None, optional): Provider-based model settings. If `None`, uses the default AskUI model stack.
        retry (Retry, optional): The retry instance to use for retrying failed actions. Defaults to `ConfigurableRetry` with exponential backoff. Currently only supported for `locate()` method.
        act_tools (list[Tool] | None, optional): Additional tools to make available for the `act()` method.

    Example:
        ```python
        from askui import ComputerAgent

        with ComputerAgent() as agent:
            agent.click("Submit button")
            agent.type("Hello World")
            agent.act("Open settings menu")
        ```
    """

    @telemetry.record_call(exclude={"reporters", "tools", "act_tools", "callbacks"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        display: Annotated[int, Field(ge=1)] = 1,
        reporters: list[Reporter] | None = None,
        tools: AgentToolbox | None = None,
        settings: AgentSettings | None = None,
        retry: Retry | None = None,
        act_tools: list[Tool] | None = None,
        callbacks: list[ConversationCallback] | None = None,
    ) -> None:
        reporter = CompositeReporter(reporters=reporters)
        self.tools = tools or AgentToolbox(
            agent_os=AskUiControllerClient(
                display=display,
                reporter=reporter,
            )
        )
        super().__init__(
            reporter=reporter,
            retry=retry,
            tools=[
                ExceptionTool(),
                ComputerGetSystemInfoTool(),
                ComputerGetMousePositionTool(),
                ComputerKeyboardPressedTool(),
                ComputerKeyboardReleaseTool(),
                ComputerKeyboardTapTool(),
                ComputerMouseClickTool(),
                ComputerMouseHoldDownTool(),
                ComputerMouseReleaseTool(),
                ComputerMouseScrollTool(),
                ComputerMoveMouseTool(),
                ComputerScreenshotTool(),
                ComputerTypeTool(),
                ComputerListDisplaysTool(),
                ComputerRetrieveActiveDisplayTool(),
                ComputerSetActiveDisplayTool(),
            ]
            + (act_tools or []),
            agent_os=self.tools.os,
            settings=settings,
            callbacks=callbacks,
        )
        self.act_agent_os_facade: ComputerAgentOsFacade = ComputerAgentOsFacade(
            self.tools.os
        )
        self.act_tool_collection.add_agent_os(self.act_agent_os_facade)
        # Override default act settings with computer-specific settings
        self.act_settings = ActSettings(
            messages=MessageSettings(
                system=create_computer_agent_prompt(),
                thinking={"type": "enabled", "budget_tokens": 2048},
            )
        )

    @telemetry.record_call(exclude={"locator"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def click(
        self,
        locator: Optional[str | Locator | Point] = None,
        button: Literal["left", "middle", "right"] = "left",
        repeat: Annotated[int, Field(gt=0)] = 1,
        offset: Optional[Point] = None,
        locate_settings: LocateSettings | None = None,
    ) -> None:
        """
        Simulates a mouse click on the user interface element identified by the provided locator.

        Args:
            locator (str | Locator | Point | None, optional): UI element description, structured locator, or absolute coordinates (x, y). If `None`, clicks at current position.
            button ('left' | 'middle' | 'right', optional): Specifies which mouse button to click. Defaults to `'left'`.
            repeat (int, optional): The number of times to click. Must be greater than `0`. Defaults to `1`.
            offset (Point | None, optional): Pixel offset (x, y) from the target location. Positive x=right, negative x=left, positive y=down, negative y=up.
            locate_settings (LocateSettings | None, optional): Settings for the locate operation. If `None`, uses agent's default settings.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.click()              # Left click on current position
                agent.click("Edit")        # Left click on text "Edit"
                agent.click((100, 200))    # Left click at absolute coordinates (100, 200)
                agent.click("Edit", button="right")  # Right click on text "Edit"
                agent.click(repeat=2)      # Double left click on current position
                agent.click("Edit", button="middle", repeat=4)   # 4x middle click on text "Edit"
                agent.click("Submit", offset=(10, -5))  # Click 10 pixels right and 5 pixels up from "Submit"
            ```
        """
        msg = "click"
        if button != "left":
            msg = f"{button} " + msg
        if repeat > 1:
            msg += f" {repeat}x times"
        if locator is not None:
            msg += f" on {locator}"
        if offset is not None:
            msg += f" with offset {offset}"
        logger.debug("Agent received instruction to %s", msg)
        self._reporter.add_message("User", msg)
        self._click(locator, button, repeat, offset, locate_settings)

    def _click(
        self,
        locator: Optional[str | Locator | Point],
        button: Literal["left", "middle", "right"],
        repeat: int,
        offset: Optional[Point],
        locate_settings: LocateSettings | None,
    ) -> None:
        if locator is not None:
            self._mouse_move(locator, offset, locate_settings)
        self.tools.os.click(button, repeat)

    def _mouse_move(
        self,
        locator: str | Locator | Point,
        offset: Optional[Point],
        locate_settings: LocateSettings | None,
    ) -> None:
        point: Point = (
            locator
            if isinstance(locator, tuple)
            else self._locate(
                locator=locator,
                locate_settings=locate_settings,
            )[0]
        )
        if offset is not None:
            point = (point[0] + offset[0], point[1] + offset[1])
        self.tools.os.mouse_move(point[0], point[1])

    @telemetry.record_call(exclude={"locator"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def mouse_move(
        self,
        locator: str | Locator | Point,
        offset: Optional[Point] = None,
        locate_settings: LocateSettings | None = None,
    ) -> None:
        """
        Moves the mouse cursor to the UI element identified by the provided locator.

        Args:
            locator (str | Locator | Point): UI element description, structured locator, or absolute coordinates (x, y).
            offset (Point | None, optional): Pixel offset (x, y) from the target location. Positive x=right, negative x=left, positive y=down, negative y=up.
            locate_settings (LocateSettings | None, optional): Settings for the locate operation. If `None`, uses agent's default settings.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.mouse_move("Submit button")  # Moves cursor to submit button
                agent.mouse_move((300, 150))       # Moves cursor to absolute coordinates (300, 150)
                agent.mouse_move("Close")          # Moves cursor to close element
                agent.mouse_move("Menu", offset=(5, 10))  # Move 5 pixels right and 10 pixels down from "Menu"
            ```
        """
        self._reporter.add_message("User", f"mouse_move: {locator}")
        logger.debug("Agent received instruction to mouse_move to %s", locator)
        self._mouse_move(locator, offset, locate_settings)

    @telemetry.record_call()
    @validate_call
    def mouse_scroll(
        self,
        x: int,
        y: int,
    ) -> None:
        """
        Simulates scrolling the mouse wheel by the specified horizontal and vertical amounts.

        Args:
            x (int): The horizontal scroll amount. Positive values typically scroll right, negative values scroll left.
            y (int): The vertical scroll amount. Positive values typically scroll down, negative values scroll up.

        Note:
            The actual scroll direction depends on the operating system's configuration.
            Some systems may have "natural scrolling" enabled, which reverses the traditional direction.

            The meaning of scroll units varies across operating systems and applications.
            A scroll value of `10` might result in different distances depending on the application and system settings.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.mouse_scroll(0, 10)  # Usually scrolls down 10 units
                agent.mouse_scroll(0, -5)  # Usually scrolls up 5 units
                agent.mouse_scroll(3, 0)   # Usually scrolls right 3 units
            ```
        """
        self._reporter.add_message("User", f'mouse_scroll: "{x}", "{y}"')
        self.tools.os.mouse_scroll(x, y)

    @telemetry.record_call(exclude={"text", "locator"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def type(
        self,
        text: Annotated[str, Field(min_length=1)],
        locator: str | Locator | Point | None = None,
        offset: Optional[Point] = None,
        clear: bool = True,
        locate_settings: LocateSettings | None = None,
    ) -> None:
        """
        Types the specified text as if it were entered on a keyboard.

        If `locator` is provided, it will first click on the element to give it focus before typing.
        If `clear` is `True` (default), it will triple click on the element to select the current text (in multi-line inputs like textareas the current line or paragraph) before typing.

        **IMPORTANT:** `clear` only works if a `locator` is provided.

        Args:
            text (str): The text to be typed. Must be at least `1` character long.
            locator (str | Locator | Point | None, optional): UI element description, structured locator, or absolute coordinates (x, y). If `None`, types at current focus.
            offset (Point | None, optional): Pixel offset (x, y) from the target location. Positive x=right, negative x=left, positive y=down, negative y=up.
            clear (bool, optional): Whether to triple click on the element to give it focus and select the current text before typing. Defaults to `True`.
            locate_settings (LocateSettings | None, optional): Settings for the locate operation. If `None`, uses agent's default settings.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.type("Hello, world!")  # Types "Hello, world!" at current focus
                agent.type("user@example.com", locator="Email")  # Clicks on "Email" input, then types
                agent.type("username", locator=(200, 100))  # Clicks at coordinates (200, 100), then types
                agent.type("Hello, world!", locator="Textarea", clear=False)  # Types "Hello, world!" into textarea without clearing
                agent.type("text", locator="Input field", offset=(5, 0))  # Click 5 pixels right of "Input field", then type
            ```
        """
        msg = f'type "{text}"'
        if locator is not None:
            msg += f" into {locator}"
            if clear:
                repeat = 3
                msg += " clearing the current content (line/paragraph) of input field"
            else:
                repeat = 1
            self._click(
                locator=locator,
                button="left",
                repeat=repeat,
                offset=offset,
                locate_settings=locate_settings,
            )
        logger.debug("Agent received instruction to %s", msg)
        self._reporter.add_message("User", msg)
        self.tools.os.type(text)

    @telemetry.record_call()
    @validate_call
    def key_up(
        self,
        key: PcKey | ModifierKey,
    ) -> None:
        """
        Simulates the release of a key.

        Args:
            key (PcKey | ModifierKey): The key to be released.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.key_up('a')  # Release the 'a' key
                agent.key_up('shift')  # Release the 'Shift' key
            ```
        """
        self._reporter.add_message("User", f'key_up "{key}"')
        logger.debug("ComputerAgent received in key_up '%s'", key)
        self.tools.os.keyboard_release(key)

    @telemetry.record_call()
    @validate_call
    def key_down(
        self,
        key: PcKey | ModifierKey,
    ) -> None:
        """
        Simulates the pressing of a key.

        Args:
            key (PcKey | ModifierKey): The key to be pressed.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.key_down('a')  # Press the 'a' key
                agent.key_down('shift')  # Press the 'Shift' key
            ```
        """
        self._reporter.add_message("User", f'key_down "{key}"')
        logger.debug("ComputerAgent received in key_down '%s'", key)
        self.tools.os.keyboard_pressed(key)

    @telemetry.record_call()
    @validate_call
    def mouse_up(
        self,
        button: Literal["left", "middle", "right"] = "left",
    ) -> None:
        """
        Simulates the release of a mouse button.

        Args:
            button ('left' | 'middle' | 'right', optional): The mouse button to be released. Defaults to `'left'`.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.mouse_up()  # Release the left mouse button
                agent.mouse_up('right')  # Release the right mouse button
                agent.mouse_up('middle')  # Release the middle mouse button
            ```
        """
        self._reporter.add_message("User", f'mouse_up "{button}"')
        logger.debug("ComputerAgent received instruction to mouse_up '%s'", button)
        self.tools.os.mouse_up(button)

    @telemetry.record_call()
    @validate_call
    def mouse_down(
        self,
        button: Literal["left", "middle", "right"] = "left",
    ) -> None:
        """
        Simulates the pressing of a mouse button.

        Args:
            button ('left' | 'middle' | 'right', optional): The mouse button to be pressed. Defaults to `'left'`.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.mouse_down()  # Press the left mouse button
                agent.mouse_down('right')  # Press the right mouse button
                agent.mouse_down('middle')  # Press the middle mouse button
            ```
        """
        self._reporter.add_message("User", f'mouse_down "{button}"')
        logger.debug("ComputerAgent received instruction to mouse_down '%s'", button)
        self.tools.os.mouse_down(button)

    @telemetry.record_call()
    @validate_call
    def keyboard(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: Optional[list[ModifierKey]] = None,
        repeat: Annotated[int, Field(gt=0)] = 1,
    ) -> None:
        """
        Simulates pressing (and releasing) a key or key combination on the keyboard.

        Args:
            key (PcKey | ModifierKey): The main key to press. This can be a letter, number, special character, or function key.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to press along with the main key. Common modifier keys include `'ctrl'`, `'alt'`, `'shift'`.
            repeat (int, optional): The number of times to press (and release) the key. Must be greater than `0`. Defaults to `1`.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.keyboard('a')  # Press 'a' key
                agent.keyboard('enter')  # Press 'Enter' key
                agent.keyboard('v', ['control'])  # Press Ctrl+V (paste)
                agent.keyboard('s', ['control', 'shift'])  # Press Ctrl+Shift+S
                agent.keyboard('a', repeat=2)  # Press 'a' key twice
            ```
        """
        msg = f"press and release key '{key}'"
        if modifier_keys is not None:
            modifier_keys_str = " + ".join(f"'{key}'" for key in modifier_keys)
            msg += f" with modifiers key{'s' if len(modifier_keys) > 1 else ''} {modifier_keys_str}"
        if repeat > 1:
            msg += f" {repeat}x times"
        self._reporter.add_message("User", msg)
        logger.debug("ComputerAgent received instruction to press '%s'", key)
        self.tools.os.keyboard_tap(key, modifier_keys, count=repeat)

    @telemetry.record_call(exclude={"command"})
    @validate_call
    def cli(
        self,
        command: Annotated[str, Field(min_length=1)],
    ) -> None:
        """
        Executes a command on the command line interface.

        This method allows running shell commands directly from the agent. The command
        is split on spaces and executed as a subprocess.

        Args:
            command (str): The command to execute on the command line.

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                # Use for Windows
                agent.cli(r'start "" "C:\Program Files\VideoLAN\VLC\vlc.exe"') # Start in VLC non-blocking
                agent.cli(r'"C:\Program Files\VideoLAN\VLC\vlc.exe"') # Start in VLC blocking

                # Mac
                agent.cli("open -a chrome")  # Open Chrome non-blocking for mac
                agent.cli("chrome")  # Open Chrome blocking for linux
                agent.cli("echo Hello World")  # Prints "Hello World"
                agent.cli("python --version")  # Displays Python version

                # Linux
                agent.cli("nohub chrome")  # Open Chrome non-blocking for linux
                agent.cli("chrome")  # Open Chrome blocking for linux
                agent.cli("echo Hello World")  # Prints "Hello World"
                agent.cli("python --version")  # Displays Python version

            ```
        """
        logger.debug(
            "ComputerAgent received instruction to execute '%s' on cli", command
        )
        self.tools.os.run_command(command)


class VisionAgent(ComputerAgent):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        warnings.warn(
            "VisionAgent is deprecated, use ComputerAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
