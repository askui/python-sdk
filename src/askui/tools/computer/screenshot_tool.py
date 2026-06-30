from PIL import Image

from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade


class ComputerScreenshotTool(ComputerBaseTool):
    """Computer Screenshot Tool"""

    def __init__(self, agent_os: ComputerAgentOsFacade | None = None) -> None:
        super().__init__(
            name="screenshot",
            description=(
                "Take a screenshot of the current screen on the currently active "
                "Agent OS target computer. The accompanying message is prefixed "
                "with the active target computer's id so it is clear which "
                "computer the screenshot was taken on."
            ),
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self) -> tuple[str, Image.Image]:
        target_id = self.agent_os.get_current_computer_target_id(report=False)
        screenshot = self.agent_os.screenshot()
        return (
            f"[Computer '{target_id}']: Screenshot was taken.",
            screenshot,
        )
