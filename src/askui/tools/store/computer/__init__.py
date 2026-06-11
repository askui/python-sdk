"""Computer-specific tools.

These tools require ComputerAgentOS (or ComputerAgentOsFacade) and are designed
for use with VisionAgent.

"""

from askui.tools.store.computer.save_screenshot_tool import ComputerSaveScreenshotTool

__all__ = [
    "ComputerSaveScreenshotTool",
]
