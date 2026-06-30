"""Coordinate scaling helper used by all agent OS facades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from askui.utils.image_utils import scale_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

    from askui.models.shared.coordinate_space import VlmCoordinateSpace
    from askui.models.shared.image_scaler import ImageScaler


class CoordinateScaler:
    """Maps coordinates between model space and device space.

    Each agent OS facade owns an instance and delegates scaling to it.

    Args:
        coordinate_space (`VlmCoordinateSpace`): Coordinate grid the model uses.
        image_scaler (`ImageScaler`): Callable to preprocess screenshots.
        fetch_real_resolution (`Callable`): Callback that returns the real
            ``(width, height)`` of the screen/device when it is not yet known.
        take_screenshot (`Callable`): Callback that triggers a screenshot
            so that ``target_resolution`` can be populated.
    """

    def __init__(
        self,
        coordinate_space: VlmCoordinateSpace,
        image_scaler: ImageScaler,
        fetch_real_resolution: Callable[[], tuple[int, int]],
        take_screenshot: Callable[[], Image.Image],
    ) -> None:
        self._coordinate_space = coordinate_space
        self._image_scaler = image_scaler
        self._fetch_real_resolution = fetch_real_resolution
        self._take_screenshot = take_screenshot
        self.target_resolution: tuple[int, int] | None = None
        self.real_screen_resolution: tuple[int, int] | None = None

    def scale_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """Record real resolution, apply scaler, record target resolution."""
        self.real_screen_resolution = screenshot.size
        scaled = self._image_scaler(screenshot)
        self.target_resolution = scaled.size
        return scaled

    def scale_coordinates(
        self,
        x: float,
        y: float,
        from_agent: bool = True,
        check_coordinates_in_bounds: bool = True,
    ) -> tuple[int, int]:
        """Map coordinates between model space and device space.

        When ``from_agent=True``, maps model-emitted coordinates to real
        device pixels.  When ``from_agent=False``, maps device coordinates
        to model space (e.g. for reporting element positions back to the model).
        """
        if self.real_screen_resolution is None:
            self.real_screen_resolution = self._fetch_real_resolution()

        target_resolution = self._ensure_target_resolution()

        if from_agent:
            if self._coordinate_space.maps_to_screenshot_pixels:
                mapped_x, mapped_y = self._coordinate_space.map_to_target(
                    x, y, target_resolution
                )
                return scale_coordinates(
                    (mapped_x, mapped_y),
                    self.real_screen_resolution,
                    target_resolution,
                    inverse=True,
                    check_coordinates_in_bounds=check_coordinates_in_bounds,
                )
            return self._coordinate_space.map_to_target(
                x, y, self.real_screen_resolution
            )

        return scale_coordinates(
            (int(x), int(y)),
            self.real_screen_resolution,
            target_resolution,
            inverse=False,
            check_coordinates_in_bounds=check_coordinates_in_bounds,
        )

    def _ensure_target_resolution(self) -> tuple[int, int]:
        if self.target_resolution is None:
            self._take_screenshot()
        assert self.target_resolution is not None  # noqa: S101
        return self.target_resolution
