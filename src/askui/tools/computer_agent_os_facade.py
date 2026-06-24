from typing import TYPE_CHECKING

from PIL import Image

from askui.models.shared.coordinate_space import VlmCoordinateSpace
from askui.models.shared.image_scaler import ImageScaler
from askui.models.shared.tool_tags import ToolTags
from askui.tools.agent_os import (
    AgentOs,
    Coordinate,
    Display,
    DisplaysListResponse,
    InputEvent,
    ModifierKey,
    MouseButton,
    PcKey,
)
from askui.tools.askui.askui_controller import RenderObjectStyle  # noqa: TC001
from askui.tools.coordinate_scaler import CoordinateScaler

if TYPE_CHECKING:
    from askui.tools.askui.askui_ui_controller_grpc.generated import (
        Controller_V1_pb2 as controller_v1_pbs,
    )
    from askui.tools.askui.askui_ui_controller_grpc.generated.AgentOS_Send_Response_2501 import (  # noqa: E501
        GetActiveProcessResponseModel,
        GetActiveWindowResponseModel,
        GetSystemInfoResponseModel,
    )


class ComputerAgentOsFacade(AgentOs):
    """Facade for `AgentOs` that adds coordinate scaling.

    Screenshots are scaled using the provider's image scaler so that the
    AI model sees an optimally sized image.  Coordinate-based inputs
    are scaled back up to the real screen resolution before being forwarded
    to the underlying agent OS.

    Args:
        agent_os (`AgentOs`): The real agent OS to wrap.
        coordinate_space (`VlmCoordinateSpace`): Coordinate grid the model uses.
        image_scaler (`ImageScaler`): Callable to preprocess screenshots.
    """

    def __init__(
        self,
        agent_os: AgentOs,
        coordinate_space: VlmCoordinateSpace,
        image_scaler: ImageScaler,
    ) -> None:
        self._agent_os = agent_os
        self._image_scaler = image_scaler
        self._scaler = CoordinateScaler(
            coordinate_space=coordinate_space,
            image_scaler=image_scaler,
            fetch_real_resolution=self._fetch_real_screen_resolution,
            take_screenshot=self._take_silent_screenshot,
        )
        self.tags.append(ToolTags.SCALED_AGENT_OS.value)

    def connect(self) -> None:
        self._agent_os.connect()
        self._scaler.real_screen_resolution = self._fetch_real_screen_resolution()

    def disconnect(self) -> None:
        self._agent_os.disconnect()
        self._scaler.real_screen_resolution = None

    def screenshot(self, report: bool = True, unscaled: bool = False) -> Image.Image:
        screenshot = self._agent_os.screenshot(report=report)
        if unscaled:
            self._scaler.real_screen_resolution = screenshot.size
            return screenshot
        return self._scaler.scale_screenshot(screenshot)

    def scale_image_for_model(self, image: Image.Image) -> Image.Image:
        """Apply the same scaling screenshots receive, without recording state.

        Unlike `screenshot`, this does not update the coordinate scaler's
        recorded resolutions, so it is safe to call on arbitrary images (e.g. a
        cropped region) without corrupting coordinate mapping.

        Args:
            image (Image.Image): The image to scale for model consumption.

        Returns:
            Image.Image: The scaled image.
        """
        return self._image_scaler(image)

    def scale_point_to_real_screen(
        self, x: float, y: float, check_coordinates_in_bounds: bool = True
    ) -> tuple[int, int]:
        """Map a point from the model coordinate space to real screen pixels.

        Args:
            x (float): The horizontal coordinate in the model coordinate space.
            y (float): The vertical coordinate in the model coordinate space.
            check_coordinates_in_bounds (bool, optional): Whether to raise if the
                mapped coordinate falls outside the screen. Set to `False` when the
                caller clamps the result itself. Defaults to `True`.

        Returns:
            tuple[int, int]: The corresponding `(x, y)` in real screen pixels.
        """
        return self._scaler.scale_coordinates(
            x, y, check_coordinates_in_bounds=check_coordinates_in_bounds
        )

    def _take_silent_screenshot(self) -> Image.Image:
        return self.screenshot(report=False)

    def _fetch_real_screen_resolution(self) -> tuple[int, int]:
        display = self._agent_os.retrieve_active_display()
        return display.size.width, display.size.height

    def mouse_move(self, x: float, y: float, duration: int = 500) -> None:
        scaled_x, scaled_y = self._scaler.scale_coordinates(x, y)
        self._agent_os.mouse_move(scaled_x, scaled_y, duration)

    def get_mouse_position(self) -> Coordinate:
        mouse_position = self._agent_os.get_mouse_position()
        scaled_x, scaled_y = self._scaler.scale_coordinates(
            mouse_position.x, mouse_position.y, from_agent=False
        )
        return Coordinate(x=scaled_x, y=scaled_y)

    def set_mouse_position(self, x: float, y: float) -> None:
        scaled_x, scaled_y = self._scaler.scale_coordinates(x, y)
        self._agent_os.set_mouse_position(scaled_x, scaled_y)

    def type(self, text: str, typing_speed: int = 50) -> None:
        self._agent_os.type(text, typing_speed)

    def click(self, button: MouseButton = "left", count: int = 1) -> None:
        self._agent_os.click(button, count)

    def mouse_down(self, button: MouseButton = "left") -> None:
        self._agent_os.mouse_down(button)

    def mouse_up(self, button: MouseButton = "left") -> None:
        self._agent_os.mouse_up(button)

    def mouse_scroll(self, dx: int, dy: int) -> None:
        self._agent_os.mouse_scroll(dx, dy)

    def keyboard_pressed(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        self._agent_os.keyboard_pressed(key, modifier_keys)

    def keyboard_release(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        self._agent_os.keyboard_release(key, modifier_keys)

    def keyboard_tap(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> None:
        self._agent_os.keyboard_tap(key, modifier_keys, count)

    def list_displays(self) -> DisplaysListResponse:
        return self._agent_os.list_displays()

    def retrieve_active_display(self) -> Display:
        return self._agent_os.retrieve_active_display()

    def set_display(self, display: int = 1) -> None:
        self._agent_os.set_display(display)
        self._scaler.real_screen_resolution = None

    def run_command(self, command: str, timeout_ms: int = 30000) -> None:
        self._agent_os.run_command(command, timeout_ms)

    def start_listening(self) -> None:
        self._agent_os.start_listening()

    def poll_event(self) -> InputEvent | None:
        return self._agent_os.poll_event()

    def stop_listening(self) -> None:
        self._agent_os.stop_listening()

    def render_quad(self, style: "RenderObjectStyle") -> int:
        return self._agent_os.render_quad(style)

    def render_line(self, style: "RenderObjectStyle", points: list[Coordinate]) -> int:
        return self._agent_os.render_line(style, points)

    def render_image(self, style: "RenderObjectStyle", image_data: str) -> int:
        return self._agent_os.render_image(style, image_data)

    def render_text(self, style: "RenderObjectStyle", content: str) -> int:
        return self._agent_os.render_text(style, content)

    def update_render_object(self, object_id: int, style: "RenderObjectStyle") -> None:
        self._agent_os.update_render_object(object_id, style)

    def delete_render_object(self, object_id: int) -> None:
        self._agent_os.delete_render_object(object_id)

    def clear_render_objects(self) -> None:
        self._agent_os.clear_render_objects()

    def get_process_list(
        self, get_extended_info: bool = False
    ) -> "controller_v1_pbs.Response_GetProcessList":
        """
        Get a list of running processes.

        Args:
            get_extended_info (bool, optional): Whether to include
                extended process information.
                Defaults to `False`.

        Returns:
            controller_v1_pbs.Response_GetProcessList: Process list response containing:
                - processes: List of ProcessInfo objects
        """
        return self._agent_os.get_process_list(get_extended_info)

    def get_window_list(
        self, process_id: int
    ) -> "controller_v1_pbs.Response_GetWindowList":
        """
        Get a list of windows for a specific process.

        Args:
            process_id (int): The ID of the process to get windows for.

        Returns:
            controller_v1_pbs.Response_GetWindowList: Window list response containing:
                - windows: List of WindowInfo objects with ID and name
        """
        return self._agent_os.get_window_list(process_id)

    def set_mouse_delay(self, delay_ms: int) -> None:
        """
        Configure mouse action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for mouse actions.
        """
        self._agent_os.set_mouse_delay(delay_ms)

    def set_keyboard_delay(self, delay_ms: int) -> None:
        """
        Configure keyboard action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for keyboard actions.
        """
        self._agent_os.set_keyboard_delay(delay_ms)

    def set_active_window(self, process_id: int, window_id: int) -> int:
        """
        Set the active window for automation.
        Adds the window as a virtual display and returns the display ID.
        It raises an error if display length is not increased after adding the window.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.

        Returns:
            int: The new Display ID.

        Raises:
            AskUiControllerError:
            If display length is not increased after adding the window.
        """
        return self._agent_os.set_active_window(process_id, window_id)

    def get_system_info(self) -> "GetSystemInfoResponseModel":
        """
        Get the system information.

        Returns:
            GetSystemInfoResponseModel: The system information.
        """
        return self._agent_os.get_system_info()

    def get_active_process(self) -> "GetActiveProcessResponseModel":
        """
        Get the current active process.

        Returns:
            GetActiveProcessResponseModel: The active process.
        """
        return self._agent_os.get_active_process()

    def set_active_process(self, process_id: int) -> None:
        """
        Set the active process.

        Args:
            process_id (int): The ID of the process to set as active.
        """
        self._agent_os.set_active_process(process_id)

    def get_active_window(self) -> "GetActiveWindowResponseModel":
        """
        Gets the window id and name in addition to the process id
             and name of the currently active window (in focus).

        Returns:
            GetActiveWindowResponseModel: The active window.
        """
        return self._agent_os.get_active_window()

    def set_window_in_focus(self, process_id: int, window_id: int) -> None:
        """
        Sets the window with the specified windowId of the process
            with the specified processId active,
            which brings it to the front and gives it focus.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.
        """
        self._agent_os.set_window_in_focus(process_id, window_id)

    def get_file_names(self, absolute_directory_path: str) -> list[str]:
        """
        List file names in an absolute directory on the automation target.

        Args:
            absolute_directory_path (str): Absolute directory path on the target system.

        Returns:
            list[str]: Names of files in that directory.
        """
        return self._agent_os.get_file_names(absolute_directory_path)

    def get_file(self, path: str) -> Image.Image | str:
        """
        Read a file from the automation target.

        Args:
            path (str): File path on the target system.

        Returns:
            Image.Image | str: Decoded file contents.
        """
        response = self._agent_os.get_file(path)
        if isinstance(response, Image.Image):
            return self._scaler.scale_screenshot(response)
        return response

    def remove_virtual_displays(self) -> None:
        """
        Remove virtual displays from the controller, leaving real displays only.
        """
        self._agent_os.remove_virtual_displays()
        self._scaler.real_screen_resolution = None
