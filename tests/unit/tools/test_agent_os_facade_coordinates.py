"""Tests for coordinate mapping in agent OS facades.

Verifies that non-pixel coordinate spaces (Qwen 0-1000, Kimi 0.0-1.0)
map directly to device resolution, bypassing the padded screenshot space.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from askui.models.shared.coordinate_space import (
    NormalizedCoordinateSpace,
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
)
from askui.tools.android.agent_os_facade import AndroidAgentOsFacade
from askui.utils.llm_image_utils import compute_contained_size, resize_image


def _default_scaler(image: Image.Image) -> Image.Image:
    """Scaler that mimics the default contained-size logic."""
    target = compute_contained_size(image.width, image.height, 1024, 768)
    return resize_image(image, target)


def _make_android_facade(
    device_size: tuple[int, int],
    coordinate_space: PixelCoordinateSpace
    | ScaledCoordinateSpace
    | NormalizedCoordinateSpace,
) -> AndroidAgentOsFacade:
    """Create an AndroidAgentOsFacade with a mocked agent OS."""
    mock_os = MagicMock()
    mock_os.tags = []
    mock_os.screenshot.return_value = Image.new("RGB", device_size)
    facade = AndroidAgentOsFacade(
        mock_os,
        coordinate_space=coordinate_space,
        image_scaler=_default_scaler,
    )
    facade._real_screen_resolution = device_size
    # Set target resolution as the scaler would produce it
    scaled = _default_scaler(Image.new("RGB", device_size))
    facade._target_resolution = scaled.size
    return facade


class TestScaledCoordinateSpaceTallDevice:
    """Qwen 0-1000 grid on a tall Android device (1080x2400).

    Non-pixel coordinate spaces map directly to device resolution,
    so no padding offset is involved.
    """

    device = (1080, 2400)
    cs = ScaledCoordinateSpace(width=1000, height=1000)

    def test_center_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(500, 500)
        assert (x, y) == (540, 1200)

    def test_left_side_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(200, 500)
        assert (x, y) == (216, 1200)

    def test_swipe_across(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x1, y1 = facade._scale_coordinates(500, 500)
        x2, y2 = facade._scale_coordinates(200, 500)
        assert (x1, y1) == (540, 1200)
        assert (x2, y2) == (216, 1200)

    def test_origin(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(0, 0)
        assert (x, y) == (0, 0)

    def test_max_corner(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(1000, 1000)
        assert (x, y) == (1080, 2400)


class TestNormalizedCoordinateSpaceTallDevice:
    """Kimi 0.0-1.0 grid on a tall Android device (1080x2400)."""

    device = (1080, 2400)
    cs = NormalizedCoordinateSpace()

    def test_center_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(0.5, 0.5)
        assert (x, y) == (540, 1200)

    def test_left_side_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(0.2, 0.5)
        assert (x, y) == (216, 1200)


class TestPixelCoordinateSpaceTallDevice:
    """Claude pixel coordinates on a tall Android device (1080x2400).

    With the no-padding scaler, a 1080x2400 device is scaled to
    compute_contained_size(1080, 2400, 1024, 768) = (345, 768).
    Pixel coordinates are in the (345, 768) screenshot space and go
    through the padding-aware inverse scaling pipeline.  Because the
    image nearly fills the target (only ~2 px rounding slack), offsets
    are close to zero but not exactly zero.
    """

    device = (1080, 2400)
    cs = PixelCoordinateSpace()

    def test_center_of_content(self) -> None:
        """The center of the content area in the scaled screenshot."""
        facade = _make_android_facade(self.device, self.cs)
        # Target resolution is (345, 768) — nearly no padding
        x, y = facade._scale_coordinates(172, 384)
        assert x == pytest.approx(538, abs=5)
        assert y == pytest.approx(1200, abs=5)

    def test_near_top_left_of_content(self) -> None:
        """Coordinate near top-left corner maps back close to origin."""
        facade = _make_android_facade(self.device, self.cs)
        # Use (1, 2) instead of exact origin to avoid rounding-offset
        # edge case that can produce small negative values.
        x, y = facade._scale_coordinates(1, 2)
        assert x == pytest.approx(3, abs=5)
        assert y == pytest.approx(3, abs=5)


class TestSquareDevice:
    """Verify no regression on a device with matching aspect ratio."""

    device = (1024, 768)
    cs = ScaledCoordinateSpace(width=1000, height=1000)

    def test_center(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scale_coordinates(500, 500)
        assert (x, y) == (512, 384)


class TestFromAgentFalse:
    """from_agent=False always maps device → screenshot pixel space."""

    def test_device_to_screenshot_scaled_space(self) -> None:
        facade = _make_android_facade(
            (1080, 2400), ScaledCoordinateSpace(width=1000, height=1000)
        )
        x, y = facade._scale_coordinates(540, 1200, from_agent=False)
        # Target resolution is (345, 768), no padding
        # Forward scaling: factor = 768/2400 = 0.32
        # x = 540 * 0.32 = 172.8 → 172, y = 1200 * 0.32 = 384
        assert x == pytest.approx(172, abs=2)
        assert y == pytest.approx(384, abs=2)
