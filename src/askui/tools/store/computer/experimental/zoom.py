from typing import cast

from PIL import Image

from askui.models.shared import ComputerBaseTool, ToolTags
from askui.reporting import NULL_REPORTER, Reporter
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

    Args:
        agent_os (`ComputerAgentOsFacade`, optional): The agent OS facade. Injected
            automatically when the tool is registered with an agent.
        reporter (`Reporter`, optional): Reporter used to show the cropped image
            (the exact image handed to the model) in the report. Defaults to a
            null reporter that discards messages.

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

    def __init__(
        self,
        agent_os: ComputerAgentOsFacade | None = None,
        reporter: Reporter = NULL_REPORTER,
    ) -> None:
        super().__init__(
            name="zoom",
            description=(
                "View a specific region of the screen at full resolution. This "
                "is a last resort for reading content that is genuinely too small "
                "to make out in the normal screenshot (e.g. tiny text, icons, "
                "status-bar text, line numbers) when that detail is required to "
                "decide your next action.\n"
                "Use it sparingly. Before zooming, rely on the normal screenshot "
                "you already have. Do NOT use this tool when:\n"
                "- the relevant text or element is already legible in the normal "
                "screenshot;\n"
                "- you only need to locate or click an element (the normal "
                "screenshot coordinates are sufficient for that);\n"
                "- you have already zoomed into this region — do not re-zoom the "
                "same area.\n"
                "Provide the region as [x1, y1, x2, y2], the top-left and "
                "bottom-right corners in the same coordinates you use for "
                "clicking. The returned image is only a magnified view; "
                "coordinates for subsequent actions still use the original screen "
                "coordinate space."
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
                        "items": {"type": "number"},
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
        self._reporter = reporter

    def __call__(self, region: list[float]) -> tuple[str, Image.Image]:
        if len(region) != 4:  # noqa: PLR2004
            error_msg = (
                f"region must contain exactly 4 values [x1, y1, x2, y2], "
                f"got {len(region)}"
            )
            raise ValueError(error_msg)

        agent_os = cast("ComputerAgentOsFacade", self.agent_os)
        # Suppress reporting of the uncropped screenshot; we report the crop below.
        screenshot = agent_os.screenshot(unscaled=True, report=False)

        # Map the model-space corners to real screen pixels. Skip the mapper's
        # bounds check; we clamp to the screenshot below so a slightly oversized
        # region from the model crops to the edge instead of erroring.
        x1, y1, x2, y2 = region
        left, top = agent_os.scale_point_to_real_screen(
            x1, y1, check_coordinates_in_bounds=False
        )
        right, bottom = agent_os.scale_point_to_real_screen(
            x2, y2, check_coordinates_in_bounds=False
        )

        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        left = max(0, min(left, screenshot.width - 1))
        right = max(left + 1, min(right, screenshot.width))
        top = max(0, min(top, screenshot.height - 1))
        bottom = max(top + 1, min(bottom, screenshot.height))

        crop = screenshot.crop((left, top, right, bottom))
        crop = agent_os.scale_image_for_model(crop)
        # Report the region in real screen pixels (where the crop was actually
        # taken), not the raw coordinates the model passed.
        self._reporter.add_message(
            "AgentOS", f"zoom([{left}, {top}, {right}, {bottom}])", crop
        )
        message = (
            f"Zoomed into region [{x1}, {y1}, {x2}, {y2}] shown at full "
            "resolution. Coordinates for further actions remain in the original "
            "screen coordinate space. Now proceed with the next action (e.g. "
            "move/click) using those coordinates; do not zoom again unless a "
            "different region is still too small to read."
        )
        return message, crop
