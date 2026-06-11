from __future__ import annotations

from typing import TYPE_CHECKING

from askui.models.shared.tool_tags import ToolTags
from askui.tools.android.agent_os import ANDROID_KEY, AndroidAgentOs, AndroidDisplay
from askui.tools.coordinate_scaling_mixin import CoordinateScaler

if TYPE_CHECKING:
    from PIL import Image

    from askui.models.shared.coordinate_space import VlmCoordinateSpace
    from askui.models.shared.image_scaler import ImageScaler
    from askui.tools.android.uiautomator_hierarchy import UIElementCollection


class AndroidAgentOsFacade(AndroidAgentOs):
    """Facade for `AndroidAgentOs` that adds coordinate scaling.

    Screenshots are scaled using the provider's image scaler so that the
    AI model sees an optimally sized image.  Coordinate-based inputs
    (``tap``, ``swipe``, ``drag_and_drop``) are scaled back up to the
    real device resolution before being forwarded to the underlying agent OS.

    Args:
        agent_os (`AndroidAgentOs`): The real Android agent OS to wrap.
        coordinate_space (`VlmCoordinateSpace`): Coordinate grid the model uses.
        image_scaler (`ImageScaler`): Callable to preprocess screenshots.
    """

    def __init__(
        self,
        agent_os: AndroidAgentOs,
        coordinate_space: VlmCoordinateSpace,
        image_scaler: ImageScaler,
    ) -> None:
        self._agent_os: AndroidAgentOs = agent_os
        self._scaler = CoordinateScaler(
            coordinate_space=coordinate_space,
            image_scaler=image_scaler,
            fetch_real_resolution=lambda: self._agent_os.screenshot().size,
            take_screenshot=lambda: self.screenshot(),
        )
        self.tags = self._agent_os.tags + [ToolTags.SCALED_AGENT_OS.value]

    def connect(self) -> None:
        self._agent_os.connect()
        self._scaler.real_screen_resolution = self._agent_os.screenshot().size

    def disconnect(self) -> None:
        self._agent_os.disconnect()
        self._scaler.real_screen_resolution = None

    def screenshot(self) -> Image.Image:
        screenshot = self._agent_os.screenshot()
        return self._scaler.scale_screenshot(screenshot)

    def tap(self, x: float, y: float) -> None:
        x, y = self._scaler.scale_coordinates(x, y)
        self._agent_os.tap(x, y)

    def swipe(
        self, x1: float, y1: float, x2: float, y2: float, duration_in_ms: int = 1000
    ) -> None:
        x1, y1 = self._scaler.scale_coordinates(x1, y1)
        x2, y2 = self._scaler.scale_coordinates(x2, y2)
        self._agent_os.swipe(x1, y1, x2, y2, duration_in_ms)

    def drag_and_drop(
        self, x1: float, y1: float, x2: float, y2: float, duration_in_ms: int = 1000
    ) -> None:
        x1, y1 = self._scaler.scale_coordinates(x1, y1)
        x2, y2 = self._scaler.scale_coordinates(x2, y2)
        self._agent_os.drag_and_drop(x1, y1, x2, y2, duration_in_ms)

    def type(self, text: str) -> None:
        self._agent_os.type(text)

    def key_tap(self, key: ANDROID_KEY) -> None:
        self._agent_os.key_tap(key)

    def key_combination(
        self, keys: list[ANDROID_KEY], duration_in_ms: int = 100
    ) -> None:
        self._agent_os.key_combination(keys, duration_in_ms)

    def shell(self, command: str) -> str:
        return self._agent_os.shell(command)

    def get_connected_displays(self) -> list[AndroidDisplay]:
        return self._agent_os.get_connected_displays()

    def set_display_by_index(self, display_index: int = 0) -> None:
        self._agent_os.set_display_by_index(display_index)
        self._scaler.real_screen_resolution = None

    def set_display_by_unique_id(self, display_unique_id: int) -> None:
        self._agent_os.set_display_by_unique_id(display_unique_id)
        self._scaler.real_screen_resolution = None

    def set_display_by_id(self, display_id: int) -> None:
        self._agent_os.set_display_by_id(display_id)
        self._scaler.real_screen_resolution = None

    def set_display_by_name(self, display_name: str) -> None:
        self._agent_os.set_display_by_name(display_name)
        self._scaler.real_screen_resolution = None

    def set_device_by_index(self, device_index: int = 0) -> None:
        self._agent_os.set_device_by_index(device_index)
        self._scaler.real_screen_resolution = None

    def set_device_by_serial_number(self, device_sn: str) -> None:
        self._agent_os.set_device_by_serial_number(device_sn)
        self._scaler.real_screen_resolution = None

    def get_connected_devices_serial_numbers(self) -> list[str]:
        return self._agent_os.get_connected_devices_serial_numbers()

    def get_selected_device_infos(self) -> tuple[str, AndroidDisplay]:
        device_sn, selected_display = self._agent_os.get_selected_device_infos()
        return device_sn, selected_display

    def connect_adb_client(self) -> None:
        self._agent_os.connect_adb_client()

    def push(self, local_path: str, remote_path: str) -> None:
        self._agent_os.push(local_path, remote_path)

    def pull(self, remote_path: str, local_path: str) -> None:
        self._agent_os.pull(remote_path, local_path)

    def get_ui_elements(self) -> UIElementCollection:
        ui_elemet_collection = self._agent_os.get_ui_elements()
        for element in ui_elemet_collection:
            if element.center is None:
                continue
            element.set_center(
                self._scaler.scale_coordinates(
                    x=element.center[0],
                    y=element.center[1],
                    from_agent=False,
                )
            )
        return ui_elemet_collection
