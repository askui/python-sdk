"""Web-specific tools.

These tools require a `PlaywrightAgentOs` and are designed for use with
`WebVisionAgent`.

"""

from askui.tools.store.web.save_screenshot_tool import WebSaveScreenshotTool

__all__ = [
    "WebSaveScreenshotTool",
]
