from typing import Literal

from PIL import Image

from askui.models.shared.coordinate_space import VlmCoordinateSpace
from askui.models.shared.image_scaler import ImageScaler
from askui.models.shared.tool_tags import ToolTags
from askui.tools.agent_os import Display, ModifierKey, PcKey
from askui.tools.playwright.agent_os import PlaywrightAgentOs
from askui.utils.image_utils import scale_coordinates


class PlaywrightAgentOsFacade(PlaywrightAgentOs):
    """Facade for `PlaywrightAgentOs` that adds coordinate scaling.

    Screenshots are scaled using the provider's image scaler so that the
    AI model sees an optimally sized image.  Coordinate-based inputs
    (``mouse_move``) are scaled back up to the real page resolution before
    being forwarded to the underlying agent OS.

    Args:
        agent_os (PlaywrightAgentOs): The real Playwright agent OS to wrap.
        coordinate_space (VlmCoordinateSpace): Coordinate grid the model uses.
        image_scaler (ImageScaler): Callable to preprocess screenshots.
    """

    def __init__(
        self,
        agent_os: PlaywrightAgentOs,
        coordinate_space: VlmCoordinateSpace,
        image_scaler: ImageScaler,
    ) -> None:
        self._agent_os = agent_os
        self._image_scaler = image_scaler
        self._target_resolution: tuple[int, int] | None = None
        self._coordinate_space: VlmCoordinateSpace = coordinate_space
        self._real_screen_resolution: tuple[int, int] | None = None
        self.tags = self._agent_os.tags + [ToolTags.SCALED_AGENT_OS.value]

    def connect(self) -> None:
        self._agent_os.connect()
        self._real_screen_resolution = self._agent_os.screenshot(
            report=False,
        ).size

    def disconnect(self) -> None:
        self._agent_os.disconnect()
        self._real_screen_resolution = None

    def screenshot(self, report: bool = True) -> Image.Image:
        screenshot = self._agent_os.screenshot(report=report)
        self._real_screen_resolution = screenshot.size
        scaled = self._image_scaler(screenshot)
        self._target_resolution = scaled.size
        return scaled

    def _ensure_target_resolution(self) -> tuple[int, int]:
        if self._target_resolution is None:
            self.screenshot(report=False)
        assert self._target_resolution is not None  # noqa: S101
        return self._target_resolution

    def _scale_coordinates(
        self,
        x: float,
        y: float,
        from_agent: bool = True,
    ) -> tuple[int, int]:
        if self._real_screen_resolution is None:
            self._real_screen_resolution = self._agent_os.screenshot(
                report=False,
            ).size

        target_resolution = self._ensure_target_resolution()

        if from_agent:
            if self._coordinate_space.maps_to_screenshot_pixels:
                mapped_x, mapped_y = self._coordinate_space.map_to_target(
                    x, y, target_resolution
                )
                return scale_coordinates(
                    (mapped_x, mapped_y),
                    self._real_screen_resolution,
                    target_resolution,
                    inverse=True,
                )
            return self._coordinate_space.map_to_target(
                x, y, self._real_screen_resolution
            )

        return scale_coordinates(
            (int(x), int(y)),
            self._real_screen_resolution,
            target_resolution,
            inverse=False,
        )

    def mouse_move(self, x: float, y: float, duration: int = 500) -> None:
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        # scaled_x, scaled_y = x, y
        self._agent_os.mouse_move(scaled_x, scaled_y, duration)

    def type(self, text: str, typing_speed: int = 50) -> None:
        self._agent_os.type(text, typing_speed)

    def click(
        self,
        button: Literal["left", "middle", "right"] = "left",
        count: int = 1,
    ) -> None:
        self._agent_os.click(button, count)

    def mouse_down(self, button: Literal["left", "middle", "right"] = "left") -> None:
        self._agent_os.mouse_down(button)

    def mouse_up(self, button: Literal["left", "middle", "right"] = "left") -> None:
        self._agent_os.mouse_up(button)

    def mouse_scroll(self, dx: int, dy: int) -> None:
        self._agent_os.mouse_scroll(dx, dy)

    def keyboard_pressed(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
    ) -> None:
        self._agent_os.keyboard_pressed(key, modifier_keys)

    def keyboard_release(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
    ) -> None:
        self._agent_os.keyboard_release(key, modifier_keys)

    def keyboard_tap(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> None:
        self._agent_os.keyboard_tap(key, modifier_keys, count)

    def retrieve_active_display(self) -> Display:
        return self._agent_os.retrieve_active_display()

    def goto(self, url: str) -> None:
        self._agent_os.goto(url)

    def back(self) -> None:
        self._agent_os.back()

    def forward(self) -> None:
        self._agent_os.forward()

    def get_page_title(self) -> str:
        return self._agent_os.get_page_title()

    def get_page_url(self) -> str:
        return self._agent_os.get_page_url()
