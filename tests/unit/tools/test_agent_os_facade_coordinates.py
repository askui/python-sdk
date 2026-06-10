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
    facade = AndroidAgentOsFacade(mock_os, coordinate_space=coordinate_space)
    facade._real_screen_resolution = device_size
    return facade


class TestScaledCoordinateSpaceTallDevice:
    """Qwen 0-1000 grid on a tall Android device (1080x2400).

    The screenshot is scaled to 345x768 with 339px horizontal padding,
    so the old code would produce negative x when x_model < ~331.
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

    Pixel coordinates are in the padded 1024x768 screenshot space
    and must go through the padding-aware inverse scaling pipeline.
    """

    device = (1080, 2400)
    cs = PixelCoordinateSpace()

    def test_center_of_content(self) -> None:
        """The center of the content area in the padded screenshot."""
        facade = _make_android_facade(self.device, self.cs)
        # Content area: x=[339..684], y=[0..768] in 1024x768 screenshot
        # Center of content: x=511, y=384
        x, y = facade._scale_coordinates(511, 384)
        # (511 - 339) / 0.32 = 537.5 → 537, (384 - 0) / 0.32 = 1200
        assert x == pytest.approx(537, abs=2)
        assert y == 1200

    def test_top_left_of_content(self) -> None:
        """Top-left corner of the content area."""
        facade = _make_android_facade(self.device, self.cs)
        # Content starts at x=339 in the padded screenshot
        x, y = facade._scale_coordinates(339, 0)
        assert x == pytest.approx(0, abs=2)
        assert y == 0


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
        # Forward scaling: (540 * 0.32 + 339, 1200 * 0.32 + 0) ≈ (512, 384)
        assert x == pytest.approx(512, abs=2)
        assert y == pytest.approx(384, abs=2)
