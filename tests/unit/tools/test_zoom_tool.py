"""Tests for `ComputerZoomTool`.

The zoom tool returns a magnified, full-resolution crop of a region the model
specifies in its (downscaled) coordinate space. The region is mapped back to
real screen pixels before cropping, so a small box in model space becomes a
larger, more legible crop.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from askui.models.shared.coordinate_space import (
    NormalizedCoordinateSpace,
    PixelCoordinateSpace,
    VlmCoordinateSpace,
)
from askui.models.shared.image_scaler import ContainedImageScaler
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade
from askui.tools.store.computer.experimental import ComputerZoomTool


def _make_facade(
    real_size: tuple[int, int],
    coordinate_space: VlmCoordinateSpace | None = None,
) -> ComputerAgentOsFacade:
    """Create a facade wrapping a mocked agent OS with a known screen size."""
    mock_os = MagicMock()
    mock_os.tags = []
    mock_os.screenshot.return_value = Image.new("RGB", real_size)
    return ComputerAgentOsFacade(
        mock_os,
        coordinate_space=coordinate_space or PixelCoordinateSpace(),
        image_scaler=ContainedImageScaler(),
    )


class TestComputerZoomTool:
    """A 2048x1536 screen maps 1:2 onto the 1024x768 model space (no padding)."""

    def test_crops_region_mapped_to_real_resolution(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        message, crop = tool(region=[100, 100, 200, 200])

        # 100px box in model space -> 200px box at full resolution.
        # The model-image scaler only downscales, so a crop within bounds
        # passes through unchanged.
        assert crop.size == (200, 200)
        assert "[100, 100, 200, 200]" in message

    def test_oversized_crop_is_scaled_to_model_bounds(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        # Full model space -> full 2048x1536 real region, larger than the
        # 1024x768 scaler bounds, so it is downscaled like a screenshot.
        _, crop = tool(region=[0, 0, 1024, 768])

        assert crop.size == (1024, 768)

    def test_requests_unscaled_screenshot(self, mocker: MockerFixture) -> None:
        facade = _make_facade((2048, 1536))
        spy = mocker.spy(facade, "screenshot")
        tool = ComputerZoomTool(agent_os=facade)

        tool(region=[0, 0, 100, 100])

        assert any(call.kwargs.get("unscaled") is True for call in spy.call_args_list)

    def test_normalizes_unordered_corners(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        _, crop = tool(region=[200, 200, 100, 100])

        assert crop.size == (200, 200)

    def test_rejects_region_with_wrong_length(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        with pytest.raises(ValueError, match="exactly 4 values"):
            tool(region=[100, 100, 200])

    def test_reports_the_cropped_image_not_the_full_screenshot(self) -> None:
        reporter = MagicMock()
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade, reporter=reporter)

        _, crop = tool(region=[100, 100, 200, 200])

        # The underlying screenshot is never fetched with reporting enabled, so
        # the uncropped image is not shown in the report.
        screenshot_mock = cast("MagicMock", facade._agent_os).screenshot
        assert screenshot_mock.call_count >= 1
        assert all(
            call.kwargs.get("report") is False
            for call in screenshot_mock.call_args_list
        )
        # Exactly the cropped image handed to the model is reported.
        reporter.add_message.assert_called_once()
        reported_image = reporter.add_message.call_args.args[2]
        assert reported_image is crop

    def test_reports_scaled_back_coordinates_not_model_coordinates(self) -> None:
        reporter = MagicMock()
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade, reporter=reporter)

        tool(region=[100, 100, 200, 200])

        # Model coords [100, 100, 200, 200] map 1:2 onto the real screen.
        reported_message = reporter.add_message.call_args.args[1]
        assert "200, 200, 400, 400" in reported_message
        assert "100, 100, 200, 200" not in reported_message

    def test_out_of_bounds_region_is_clamped_not_rejected(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        # Region extends well past the model bounds; it must clamp to the screen
        # edge and crop the whole screen instead of raising.
        _, crop = tool(region=[0, 0, 5000, 5000])

        assert crop.size == (1024, 768)

    def test_accepts_normalized_float_region(self) -> None:
        facade = _make_facade((1920, 1080), NormalizedCoordinateSpace())
        tool = ComputerZoomTool(agent_os=facade)

        # Kimi-style 0.0-1.0 coordinates: 0.4..0.6 spans 20% of each axis.
        _, crop = tool(region=[0.4, 0.4, 0.6, 0.6])

        # 0.2 * 1920 = 384 wide, 0.2 * 1080 = 216 tall.
        assert crop.size == (384, 216)

    def test_returns_text_and_image(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        result = tool(region=[10, 20, 110, 120])

        message, crop = result
        assert isinstance(message, str)
        assert isinstance(crop, Image.Image)
