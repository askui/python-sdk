import re
from typing import get_args

from PIL import Image
from typing_extensions import override

from askui.models.shared import PlaywrightBaseTool, ToolTags
from askui.tools.agent_os import ModifierKey, MouseButton, PcKey
from askui.tools.playwright.agent_os import PlaywrightAgentOs


class PlaywrightScreenshotTool(PlaywrightBaseTool):
    """Takes a screenshot of the current browser page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="screenshot",
            description="Take a screenshot of the current browser page.",
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    @override
    def __call__(self) -> tuple[str, Image.Image]:
        screenshot = self.agent_os.screenshot()
        return "Screenshot was taken.", screenshot


class PlaywrightMouseMoveTool(PlaywrightBaseTool):
    """Moves the mouse to a specific position on the page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="move_mouse",
            description=(
                "Move the mouse to a specific position on the page. "
                "Pass x and y as separate integer values, not as a combined string."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": (
                            "The x (horizontal) pixel coordinate. "
                            "Must be a single integer, e.g. 330."
                        ),
                    },
                    "y": {
                        "type": "integer",
                        "description": (
                            "The y (vertical) pixel coordinate. "
                            "Must be a single integer, e.g. 182."
                        ),
                    },
                },
                "required": ["x", "y"],
            },
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    @override
    def __call__(self, x: int, y: int) -> str:
        # The agent occasionally passes coordinates as strings instead of ints.
        # We parse them to handle both cases.
        if not (isinstance(x, int) and isinstance(y, int)):
            x, y = self._parse_coordinates(x, y)  # type: ignore[unreachable]
        self.agent_os.mouse_move(x, y)
        return f"Mouse was moved to ({x}, {y})."

    @staticmethod
    def _parse_coordinates(x: float | str, y: float | str) -> tuple[int, int]:
        number_pattern = re.compile(r"-?\d+")
        combined = f"{x},{y}"
        numbers = number_pattern.findall(combined)
        if not len(numbers) == 2:
            error_msg = (
                "Could not parse mouse_move coordinates from provided "
                f"parameters x={x}, y={y}. The parameters x and y must "
                "be passed as separate integer values!"
            )
            raise ValueError(error_msg)
        return int(numbers[0]), int(numbers[1])


class PlaywrightMouseClickTool(PlaywrightBaseTool):
    """Clicks the mouse button at the current position on the page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="mouse_click",
            description=(
                "Click and release the mouse button at the current"
                " position on the page."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "mouse_button": {
                        "type": "string",
                        "description": "The mouse button to click.",
                        "enum": get_args(MouseButton),
                    },
                    "number_of_clicks": {
                        "type": "integer",
                        "description": (
                            "The number of times to click the mouse button."
                            " Defaults to 1"
                        ),
                        "default": 1,
                    },
                },
                "required": ["mouse_button"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self, mouse_button: MouseButton, number_of_clicks: int = 1) -> str:
        self.agent_os.click(mouse_button, number_of_clicks)
        return f"Mouse button {mouse_button} was clicked {number_of_clicks} times."


class PlaywrightMouseScrollTool(PlaywrightBaseTool):
    """Scrolls the mouse wheel at the current position on the page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="mouse_scroll",
            description=(
                "Scroll the mouse wheel at the current position on the page. "
                "Positive dy scrolls down, negative dy scrolls up. "
                "Start with dy=150 or dy=-150 for a normal scroll "
                "and adjust based on the result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dx": {
                        "type": "integer",
                        "description": (
                            "The horizontal scroll amount. "
                            "Positive values scroll right, "
                            "negative values scroll left. "
                            "Use 0 if no horizontal scrolling is needed."
                        ),
                    },
                    "dy": {
                        "type": "integer",
                        "description": (
                            "The vertical scroll amount. "
                            "Positive values scroll down, negative values scroll up. "
                            "Use 0 if no vertical scrolling is needed."
                        ),
                    },
                },
                "required": ["dx", "dy"],
            },
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    @override
    def __call__(self, dx: int, dy: int) -> str:
        self.agent_os.mouse_scroll(dx, dy)
        return f"Mouse was scrolled by ({dx}, {dy})."


class PlaywrightMouseHoldDownTool(PlaywrightBaseTool):
    """Holds down the mouse button at the current position on the page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="mouse_hold_down",
            description=(
                "Hold down the mouse button at the current position on the page."
            ),
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

    @override
    def __call__(self, mouse_button: MouseButton) -> str:
        self.agent_os.mouse_down(mouse_button)
        return f"Mouse button {mouse_button} is now held down."


class PlaywrightMouseReleaseTool(PlaywrightBaseTool):
    """Releases the mouse button at the current position on the page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="mouse_release",
            description="Release the mouse button at the current position on the page.",
            input_schema={
                "type": "object",
                "properties": {
                    "mouse_button": {
                        "type": "string",
                        "description": "The mouse button to release.",
                        "enum": get_args(MouseButton),
                    },
                },
                "required": ["mouse_button"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self, mouse_button: MouseButton) -> str:
        self.agent_os.mouse_up(mouse_button)
        return f"Mouse button {mouse_button} was released."


class PlaywrightTypeTool(PlaywrightBaseTool):
    """Types text in the browser page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="type",
            description=(
                "Type text in the browser page. To enter a secret/sensitive value, "
                "pass its placeholder `<|secret|>NAME<|secret|>` as the text; the real "
                "value is substituted securely at runtime and is hidden from you."
            ),
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

    @override
    def __call__(self, text: str, typing_speed: int = 50) -> str:
        self.agent_os.type(text, typing_speed)
        return f"Text '{text}' was typed."


class PlaywrightKeyboardTapTool(PlaywrightBaseTool):
    """Taps (press and release) a keyboard key in the browser."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="keyboard_tap",
            description="Tap (press and release) a keyboard key in the browser.",
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

    @override
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


class PlaywrightKeyboardPressedTool(PlaywrightBaseTool):
    """Presses and holds a keyboard key in the browser."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="keyboard_pressed",
            description="Press and hold a keyboard key in the browser.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to press.",
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
                },
                "required": ["key"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
    ) -> str:
        self.agent_os.keyboard_pressed(key, modifier_keys)
        modifier_str = (
            f" with modifiers {', '.join(modifier_keys)}" if modifier_keys else ""
        )
        return f"Key {key} is now pressed{modifier_str}."


class PlaywrightKeyboardReleaseTool(PlaywrightBaseTool):
    """Releases a keyboard key in the browser."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="keyboard_release",
            description="Release a keyboard key in the browser.",
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

    @override
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


class PlaywrightGotoTool(PlaywrightBaseTool):
    """
    Navigates to a specific URL in the browser.
    """

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="playwright_goto_tool",
            description=(
                """
                Navigates the browser to a specific URL.
                This will load the webpage at the given URL and make it the current
                page. The browser will wait for the page to load completely before
                proceeding.
                """
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The URL to navigate to. Must be a valid URL including "
                            "the protocol (e.g., 'https://example.com')."
                        ),
                    },
                },
                "required": ["url"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self, url: str) -> str:
        self.agent_os.goto(url)
        return f"Navigated to: {url}"


class PlaywrightBackTool(PlaywrightBaseTool):
    """Navigates back to the previous page in the browser history."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="playwright_back_tool",
            description=(
                """
                Navigates back to the previous page in the browser history.
                This is equivalent to clicking the back button in a browser.
                If there is no previous page in the history, this action will have no
                effect.
                """
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        self.agent_os.back()
        return "Navigated back to the previous page"


class PlaywrightForwardTool(PlaywrightBaseTool):
    """Navigates forward to the next page in the browser history."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="playwright_forward_tool",
            description=(
                """
                Navigates forward to the next page in the browser history.
                This is equivalent to clicking the forward button in a browser.
                If there is no next page in the history, this action will have no
                effect.
                """
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        self.agent_os.forward()
        return "Navigated forward to the next page"


class PlaywrightGetPageTitleTool(PlaywrightBaseTool):
    """Gets the title of the current page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="playwright_get_page_title_tool",
            description=(
                """
                Retrieves the title of the currently loaded webpage.
                The title is typically displayed in the browser tab and represents
                the main heading or name of the page content.
                """
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        title = self.agent_os.get_page_title()
        return f"Page title: {title}"


class PlaywrightGetPageUrlTool(PlaywrightBaseTool):
    """Gets the URL of the current page."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="playwright_get_page_url_tool",
            description=(
                """
                Retrieves the URL of the currently loaded webpage.
                This returns the full URL including protocol, domain, path, and query
                parameters.
                """
            ),
            agent_os=agent_os,
        )
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        url = self.agent_os.get_page_url()
        return f"Current page URL: {url}"


class PlaywrightListTabsTool(PlaywrightBaseTool):
    """Lists all open browser tabs with their index, title, and URL."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="list_tabs",
            description=(
                "List all currently open browser tabs. "
                "Returns the index, title, and URL of each tab. "
                "Use the index with switch_tab or close_tab."
            ),
            agent_os=agent_os,
        )

    @override
    def __call__(self) -> str:
        tabs = self.agent_os.list_tabs()
        if not tabs:
            return "No open tabs."
        lines = [f"[{t['index']}] {t['title']} — {t['url']}" for t in tabs]
        return "Open tabs:\n" + "\n".join(lines)


class PlaywrightSwitchTabTool(PlaywrightBaseTool):
    """Switches the active browser tab to a tab at a given index."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="switch_tab",
            description=(
                "Switch to a different browser tab by its index. "
                "Use list_tabs to see all open tabs and their indices."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": (
                            "Zero-based index of the tab to switch to (from list_tabs)."
                        ),
                    },
                },
                "required": ["index"],
            },
            agent_os=agent_os,
        )

    @override
    def __call__(self, index: int) -> str:
        self.agent_os.switch_tab(index)
        return f"Switched to tab {index}."


class PlaywrightCloseTabTool(PlaywrightBaseTool):
    """Closes a browser tab by its index."""

    def __init__(self, agent_os: PlaywrightAgentOs | None = None) -> None:
        super().__init__(
            name="close_tab",
            description=(
                "Close a browser tab by its index. "
                "Use list_tabs to see available tabs. "
                "The last remaining tab cannot be closed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": (
                            "Zero-based index of the tab to close (from list_tabs)."
                        ),
                    },
                },
                "required": ["index"],
            },
            agent_os=agent_os,
        )

    @override
    def __call__(self, index: int) -> str:
        self.agent_os.close_tab(index)
        return f"Tab {index} was closed."
