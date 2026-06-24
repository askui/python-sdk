from typing import cast

from PIL import Image

from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade


class ComputerZoomTool(ComputerBaseTool):
    """
    Views a region of the screen at full resolution to inspect small details.

    Screenshots are downscaled before they reach the model, so small UI elements
    (icons, tab titles, status-bar text, line numbers, tiny buttons) can become
    illegible. This tool crops the requested region from the full-resolution
    screenshot and returns it magnified. The returned image is only a magnified
    view; coordinates for subsequent actions still use the original screen
    coordinate space.

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.computer.experimental import ComputerZoomTool

        with ComputerAgent(act_tools=[ComputerZoomTool()]) as agent:
            agent.act("Enable the tiny checkbox next to 'Advanced options'")

        with ComputerAgent() as agent:
            agent.act(
                "Enable the tiny checkbox next to 'Advanced options'",
                tools=[ComputerZoomTool()],
            )
        ```
    """

    def __init__(self, agent_os: ComputerAgentOsFacade | None = None) -> None:
        super().__init__(
            name="zoom",
            description=(
                "View a specific region of the screen at full resolution. Use "
                "this to read small text or to locate small UI elements (icons, "
                "tab titles, status-bar text, line numbers, tiny buttons) that "
                "are not legible in a normal screenshot. Provide the region as "
                "[x1, y1, x2, y2], the top-left and bottom-right corners in the "
                "same coordinates you use for clicking. The returned image is "
                "only a magnified view; coordinates for subsequent actions still "
                "use the original screen coordinate space."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "array",
                        "description": (
                            "The region to zoom into as [x1, y1, x2, y2]: the "
                            "top-left and bottom-right corners in screen "
                            "coordinates."
                        ),
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["region"],
            },
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self, region: list[int]) -> tuple[str, Image.Image]:
        if len(region) != 4:  # noqa: PLR2004
            error_msg = (
                f"region must contain exactly 4 values [x1, y1, x2, y2], "
                f"got {len(region)}"
            )
            raise ValueError(error_msg)

        agent_os = cast("ComputerAgentOsFacade", self.agent_os)
        screenshot = agent_os.screenshot(unscaled=True)

        x1, y1, x2, y2 = region
        left, top = agent_os.scale_point_to_real_screen(x1, y1)
        right, bottom = agent_os.scale_point_to_real_screen(x2, y2)

        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        left = max(0, min(left, screenshot.width))
        right = max(left + 1, min(right, screenshot.width))
        top = max(0, min(top, screenshot.height))
        bottom = max(top + 1, min(bottom, screenshot.height))

        crop = screenshot.crop((left, top, right, bottom))
        message = (
            f"Zoomed into region [{x1}, {y1}, {x2}, {y2}] shown at full "
            "resolution. Coordinates for further actions remain in the original "
            "screen coordinate space."
        )
        return message, crop
