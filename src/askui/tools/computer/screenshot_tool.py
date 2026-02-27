from PIL import Image

from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade


class ComputerScreenshotTool(ComputerBaseTool):
    """Computer Screenshot Tool"""

    def __init__(self, agent_os: ComputerAgentOsFacade | None = None) -> None:
        super().__init__(
            name="screenshot",
            description="Take a screenshot of the current screen.",
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self) -> tuple[str, Image.Image]:
        screenshot = self.agent_os.screenshot()
        return "Screenshot was taken.", screenshot
