"""Tests for `ComputerZoomTool`.

The zoom tool returns a magnified, full-resolution crop of a region the model
specifies in its (downscaled) coordinate space. The region is mapped back to
real screen pixels before cropping, so a small box in model space becomes a
larger, more legible crop.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from askui.models.shared.coordinate_space import PixelCoordinateSpace
from askui.models.shared.image_scaler import ContainedImageScaler
from askui.tools.computer_agent_os_facade import ComputerAgentOsFacade
from askui.tools.store.computer.experimental import ComputerZoomTool


def _make_facade(real_size: tuple[int, int]) -> ComputerAgentOsFacade:
    """Create a facade wrapping a mocked agent OS with a known screen size."""
    mock_os = MagicMock()
    mock_os.tags = []
    mock_os.screenshot.return_value = Image.new("RGB", real_size)
    return ComputerAgentOsFacade(
        mock_os,
        coordinate_space=PixelCoordinateSpace(),
        image_scaler=ContainedImageScaler(),
    )


class TestComputerZoomTool:
    """A 2048x1536 screen maps 1:2 onto the 1024x768 model space (no padding)."""

    def test_crops_region_mapped_to_real_resolution(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        message, crop = tool(region=[100, 100, 200, 200])

        # 100px box in model space -> 200px box at full resolution
        assert crop.size == (200, 200)
        assert "[100, 100, 200, 200]" in message

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

    def test_returns_text_and_image(self) -> None:
        facade = _make_facade((2048, 1536))
        tool = ComputerZoomTool(agent_os=facade)

        result = tool(region=[10, 20, 110, 120])

        message, crop = result
        assert isinstance(message, str)
        assert isinstance(crop, Image.Image)
