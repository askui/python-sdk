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
from askui.models.shared.image_scaler import ContainedImageScaler
from askui.tools.android.agent_os_facade import AndroidAgentOsFacade

_default_scaler = ContainedImageScaler()


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
    facade._scaler.real_screen_resolution = device_size
    # Set target resolution as the scaler would produce it
    scaled = _default_scaler(Image.new("RGB", device_size))
    facade._scaler.target_resolution = scaled.size
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
        x, y = facade._scaler.scale_coordinates(500, 500)
        assert (x, y) == (540, 1200)

    def test_left_side_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(200, 500)
        assert (x, y) == (216, 1200)

    def test_swipe_across(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x1, y1 = facade._scaler.scale_coordinates(500, 500)
        x2, y2 = facade._scaler.scale_coordinates(200, 500)
        assert (x1, y1) == (540, 1200)
        assert (x2, y2) == (216, 1200)

    def test_origin(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(0, 0)
        assert (x, y) == (0, 0)

    def test_max_corner(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(1000, 1000)
        assert (x, y) == (1080, 2400)


class TestNormalizedCoordinateSpaceTallDevice:
    """Kimi 0.0-1.0 grid on a tall Android device (1080x2400)."""

    device = (1080, 2400)
    cs = NormalizedCoordinateSpace()

    def test_center_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(0.5, 0.5)
        assert (x, y) == (540, 1200)

    def test_left_side_tap(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(0.2, 0.5)
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
        x, y = facade._scaler.scale_coordinates(172, 384)
        assert x == pytest.approx(538, abs=5)
        assert y == pytest.approx(1200, abs=5)

    def test_near_top_left_of_content(self) -> None:
        """Coordinate near top-left corner maps back close to origin."""
        facade = _make_android_facade(self.device, self.cs)
        # Use (1, 2) instead of exact origin to avoid rounding-offset
        # edge case that can produce small negative values.
        x, y = facade._scaler.scale_coordinates(1, 2)
        assert x == pytest.approx(3, abs=5)
        assert y == pytest.approx(3, abs=5)


class TestSquareDevice:
    """Verify no regression on a device with matching aspect ratio."""

    device = (1024, 768)
    cs = ScaledCoordinateSpace(width=1000, height=1000)

    def test_center(self) -> None:
        facade = _make_android_facade(self.device, self.cs)
        x, y = facade._scaler.scale_coordinates(500, 500)
        assert (x, y) == (512, 384)


class TestFromAgentFalse:
    """from_agent=False always maps device → screenshot pixel space."""

    def test_device_to_screenshot_scaled_space(self) -> None:
        facade = _make_android_facade(
            (1080, 2400), ScaledCoordinateSpace(width=1000, height=1000)
        )
        x, y = facade._scaler.scale_coordinates(540, 1200, from_agent=False)
        # Target resolution is (345, 768), no padding
        # Forward scaling: factor = 768/2400 = 0.32
        # x = 540 * 0.32 = 172.8 → 172, y = 1200 * 0.32 = 384
        assert x == pytest.approx(172, abs=2)
        assert y == pytest.approx(384, abs=2)


# ---------------------------------------------------------------------------
# Parametrized tests across multiple resolutions
# ---------------------------------------------------------------------------

_DEVICE_SIZES = [
    pytest.param((1080, 1920), id="FHD portrait"),
    pytest.param((1920, 1080), id="FHD landscape"),
    pytest.param((1440, 2560), id="QHD portrait"),
    pytest.param((2560, 1440), id="QHD landscape"),
    pytest.param((1080, 2400), id="tall Android"),
    pytest.param((768, 1024), id="iPad portrait"),
    pytest.param((320, 480), id="small phone"),
    pytest.param((3840, 2160), id="4K landscape"),
]


class TestScaledCenterAcrossResolutions:
    """Center tap (500, 500) in 0-1000 grid should always map to device center."""

    cs = ScaledCoordinateSpace(width=1000, height=1000)

    @pytest.mark.parametrize("device_size", _DEVICE_SIZES)
    def test_center_maps_to_device_center(self, device_size: tuple[int, int]) -> None:
        facade = _make_android_facade(device_size, self.cs)
        x, y = facade._scaler.scale_coordinates(500, 500)
        assert x == device_size[0] // 2
        assert y == device_size[1] // 2


class TestNormalizedCenterAcrossResolutions:
    """Center tap (0.5, 0.5) in normalized grid should always map to device center."""

    cs = NormalizedCoordinateSpace()

    @pytest.mark.parametrize("device_size", _DEVICE_SIZES)
    def test_center_maps_to_device_center(self, device_size: tuple[int, int]) -> None:
        facade = _make_android_facade(device_size, self.cs)
        x, y = facade._scaler.scale_coordinates(0.5, 0.5)
        assert x == device_size[0] // 2
        assert y == device_size[1] // 2


class TestPixelRoundTripAcrossResolutions:
    """Pixel-space center of scaled image should round-trip close to device center."""

    cs = PixelCoordinateSpace()

    @pytest.mark.parametrize("device_size", _DEVICE_SIZES)
    def test_pixel_center_round_trip(self, device_size: tuple[int, int]) -> None:
        facade = _make_android_facade(device_size, self.cs)
        target = facade._scaler.target_resolution
        assert target is not None
        cx, cy = target[0] // 2, target[1] // 2
        x, y = facade._scaler.scale_coordinates(cx, cy)
        assert x == pytest.approx(device_size[0] // 2, abs=5)
        assert y == pytest.approx(device_size[1] // 2, abs=5)


# ---------------------------------------------------------------------------
# Negative / edge-case tests
# ---------------------------------------------------------------------------


class TestOutOfBoundsCoordinates:
    """Coordinates outside the valid range should raise ValueError."""

    def test_negative_coordinates_pixel_space(self) -> None:
        facade = _make_android_facade((1080, 1920), PixelCoordinateSpace())
        with pytest.raises(ValueError, match="out of bounds"):
            facade._scaler.scale_coordinates(-10, -10)

    def test_exceeding_target_pixel_space(self) -> None:
        facade = _make_android_facade((1080, 1920), PixelCoordinateSpace())
        target = facade._scaler.target_resolution
        assert target is not None
        with pytest.raises(ValueError, match="out of bounds"):
            facade._scaler.scale_coordinates(target[0] + 100, target[1] + 100)

    def test_bounds_check_can_be_disabled(self) -> None:
        facade = _make_android_facade((1080, 1920), PixelCoordinateSpace())
        target = facade._scaler.target_resolution
        assert target is not None
        # Should not raise when bounds checking is off
        facade._scaler.scale_coordinates(
            target[0] + 100, target[1] + 100, check_coordinates_in_bounds=False
        )


class TestResolutionLazyInit:
    """Verify that real_screen_resolution is fetched lazily when not set."""

    def test_fetches_resolution_on_first_scale(self) -> None:
        mock_os = MagicMock()
        mock_os.tags = []
        device_size = (1080, 1920)
        mock_os.screenshot.return_value = Image.new("RGB", device_size)
        cs = ScaledCoordinateSpace(width=1000, height=1000)
        facade = AndroidAgentOsFacade(
            mock_os, coordinate_space=cs, image_scaler=_default_scaler
        )
        # real_screen_resolution starts unset
        assert facade._scaler.real_screen_resolution is None  # noqa: S101
        # Trigger a screenshot to populate target_resolution
        facade.screenshot()
        # Now scale — should have both resolutions set
        scaler = facade._scaler
        x, y = scaler.scale_coordinates(500, 500)
        assert scaler.real_screen_resolution == device_size
        assert x == 540
        assert y == 960
