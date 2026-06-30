import base64
import logging
import pathlib
import subprocess
import sys
import time
import types
import uuid
from typing import Literal, Type

import grpc
from google.protobuf.json_format import MessageToDict
from PIL import Image
from typing_extensions import Self, override

from askui.container import telemetry
from askui.reporting import NULL_REPORTER, Reporter
from askui.tools.agent_os import (
    AgentOs,
    Coordinate,
    Display,
    DisplaysListResponse,
    ModifierKey,
    PcKey,
)
from askui.tools.askui.askui_controller_client_settings import (
    AskUiControllerClientSettings,
)
from askui.tools.askui.askui_controller_settings import AskUiControllerSettings
from askui.tools.askui.askui_ui_controller_grpc.desktop_agent_os_error import (
    DesktopAgentOsError,
)
from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2 as controller_v1_pbs,
)
from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2_grpc as controller_v1,
)
from askui.tools.askui.askui_ui_controller_grpc.generated.AgentOS_Send_Request_2501 import (  # noqa: E501
    AddRenderObjectCommand,
    AskUIAgentOSSendRequestSchema,
    ClearRenderObjectsCommand,
    Command,
    DeleteRenderObjectCommand,
    GetActiveProcessCommand,
    GetActiveWindowCommand,
    GetFileCommand,
    GetFileNamesCommand,
    GetMousePositionCommand,
    GetSystemInfoCommand,
    Guid,
    Header,
    Length,
    Location,
    Message,
    Parameter3,
    RemoveVirtualDisplaysCommand,
    RenderImage,
    RenderObjectId,
    RenderObjectStyle,
    RenderText,
    SetActiveProcessCommand,
    SetActiveWindowCommand,
    SetMousePositionCommand,
    UpdateRenderObjectCommand,
)
from askui.tools.askui.askui_ui_controller_grpc.generated.AgentOS_Send_Response_2501 import (  # noqa: E501
    AskUIAgentOSSendResponseSchema,
    GetActiveProcessResponse,
    GetActiveProcessResponseModel,
    GetActiveWindowResponse,
    GetActiveWindowResponseModel,
    GetFileNamesResponse,
    GetFileResponse,
    GetSystemInfoResponse,
    GetSystemInfoResponseModel,
)
from askui.utils.annotated_image import AnnotatedImage
from askui.utils.image_utils import base64_to_image

from ..utils import process_exists, wait_for_port
from .exceptions import (
    AskUiControllerError,
    AskUiControllerInvalidCommandError,
    AskUiControllerOperationTimeoutError,
)

logger = logging.getLogger(__name__)


class AskUiControllerServer:
    """
    Concrete implementation of `ControllerServer` for managing the AskUI Remote Device
    Controller process.
    Handles process discovery, startup, and shutdown for the native controller binary.

    Args:
        settings (AskUiControllerSettings | None, optional): Settings for the AskUI.
    """

    def __init__(self, settings: AskUiControllerSettings | None = None) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._settings = settings or AskUiControllerSettings()

    def _start_process(
        self,
        path: pathlib.Path,
        args: str | None = None,
    ) -> None:
        commands = [str(path)]
        if args:
            commands.extend(args.split())
        if not logger.isEnabledFor(logging.DEBUG):
            self._process = subprocess.Popen(
                commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            self._process = subprocess.Popen(commands)
        wait_for_port(23000)

    def start(self, clean_up: bool = False) -> None:
        """
        Start the controller process.

        Args:
            clean_up (bool, optional): Whether to clean up existing processes
                (only on Windows) before starting. Defaults to `False`.
        """
        if (
            sys.platform == "win32"
            and clean_up
            and process_exists("AskuiRemoteDeviceController.exe")
        ):
            self.clean_up()
        logger.debug(
            "Starting AskUI Remote Device Controller",
            extra={"path": str(self._settings.controller_path)},
        )
        self._start_process(
            self._settings.controller_path, self._settings.controller_args
        )
        time.sleep(0.5)

    def clean_up(self) -> None:
        subprocess.run("taskkill.exe /IM AskUI*")
        time.sleep(0.1)

    def stop(self, force: bool = False) -> None:
        """
        Stop the controller process.

        Args:
            force (bool, optional): Whether to forcefully terminate the process.
                Defaults to `False`.
        """
        if self._process is None:
            return  # Nothing to stop

        try:
            if force:
                self._process.kill()
                if sys.platform == "win32":
                    self.clean_up()
            else:
                self._process.terminate()
        except Exception:  # noqa: BLE001 - We want to catch all other exceptions here
            logger.exception("Controller error")
        finally:
            self._process = None


class AskUiControllerClient(AgentOs):
    """
    Implementation of `AgentOs` that communicates with the AskUI Remote Device
    Controller via gRPC.

    Args:
        reporter (Reporter): Reporter used for reporting with the `"AgentOs"`.
        display (int, optional): Display number to use. Defaults to `1`.
        controller_server (AskUiControllerServer | None, optional): Custom controller
            server. Defaults to `ControllerServer`.
    """

    @telemetry.record_call(exclude={"reporter", "controller_server"})
    def __init__(
        self,
        reporter: Reporter = NULL_REPORTER,
        display: int = 1,
        controller_server: AskUiControllerServer | None = None,
        settings: AskUiControllerClientSettings | None = None,
    ) -> None:
        self._stub: controller_v1.ControllerAPIStub | None = None
        self._channel: grpc.Channel | None = None
        self._session_info: controller_v1_pbs.SessionInfo | None = None
        self._pre_action_wait = 0
        self._post_action_wait = 0.05
        self._max_retries = 10
        self._display = display
        self._reporter = reporter
        self._controller_server = controller_server or AskUiControllerServer()
        self._session_guid = "{" + str(uuid.uuid4()) + "}"
        self._settings = settings or AskUiControllerClientSettings()

    @telemetry.record_call()
    @override
    def connect(self) -> None:
        """
        Establishes a connection to the AskUI Remote Device Controller.

        This method starts the controller server, establishes a gRPC channel,
        creates a session, and sets up the initial display.
        """
        if self._settings.server_autostart:
            self._controller_server.start()
        self._channel = grpc.insecure_channel(
            self._settings.server_address,
            options=[
                ("grpc.max_send_message_length", 2**30),
                ("grpc.max_receive_message_length", 2**30),
                ("grpc.default_deadline", 300000),
            ],
        )
        self._stub = controller_v1.ControllerAPIStub(self._channel)
        self._start_session()
        self._start_execution()
        self.set_display(self._display)
        if self._settings.clean_virtual_displays:
            logger.info(
                "clean_virtual_displays is enabled. Removing all virtual displays ... "
            )
            self.remove_virtual_displays()
            logger.info("Virtual displays removed.")

    def _get_stub(self) -> controller_v1.ControllerAPIStub:
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized. Call `connect()` first."
        )
        return self._stub

    def _run_recorder_action(
        self,
        acion_class_id: controller_v1_pbs.ActionClassID,
        action_parameters: controller_v1_pbs.ActionParameters,
    ) -> controller_v1_pbs.Response_RunRecordedAction:
        time.sleep(self._pre_action_wait)
        response: controller_v1_pbs.Response_RunRecordedAction = (
            self._get_stub().RunRecordedAction(
                controller_v1_pbs.Request_RunRecordedAction(
                    sessionInfo=self._session_info,
                    actionClassID=acion_class_id,
                    actionParameters=action_parameters,
                )
            )
        )

        time.sleep((response.requiredMilliseconds / 1000))
        num_retries = 0
        for _ in range(self._max_retries):
            poll_response: controller_v1_pbs.Response_Poll = self._get_stub().Poll(
                controller_v1_pbs.Request_Poll(
                    sessionInfo=self._session_info,
                    pollEventID=controller_v1_pbs.PollEventID.PollEventID_ActionFinished,
                )
            )
            if (
                poll_response.pollEventParameters.actionFinished.actionID
                == response.actionID
            ):
                break
            time.sleep(self._post_action_wait)
            num_retries += 1
        if num_retries == self._max_retries - 1:
            raise AskUiControllerOperationTimeoutError
        return response

    @telemetry.record_call()
    @override
    def disconnect(self) -> None:
        """
        Terminates the connection to the AskUI Remote Device Controller.

        This method stops the execution, ends the session, closes the gRPC channel,
        and stops the controller server.
        """
        try:
            self._stop_execution()
            self._stop_session()
            if self._channel is not None:
                self._channel.close()
            self._controller_server.stop()
        except Exception as e:  # noqa: BLE001
            # We want to catch all other exceptions here and not re-raise them
            msg = (
                "Error while disconnecting from the AskUI Remote Device Controller"
                f" Error: {e}"
            )
            logger.exception(msg)

    @telemetry.record_call()
    def __enter__(self) -> Self:
        """
        Context manager entry point that establishes the connection.

        Returns:
            Self: The instance of AskUiControllerClient.
        """
        self.connect()
        return self

    @telemetry.record_call(exclude={"exc_value", "traceback"})
    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """
        Context manager exit point that disconnects the client.

        Args:
            exc_type: The exception type if an exception was raised.
            exc_value: The exception value if an exception was raised.
            traceback: The traceback if an exception was raised.
        """
        self.disconnect()

    def _start_session(self) -> None:
        response = self._get_stub().StartSession(
            controller_v1_pbs.Request_StartSession(
                sessionGUID=self._session_guid, immediateExecution=True
            )
        )
        self._session_info = response.sessionInfo

    def _stop_session(self) -> None:
        self._get_stub().EndSession(
            controller_v1_pbs.Request_EndSession(sessionInfo=self._session_info)
        )

    def _start_execution(self) -> None:
        self._get_stub().StartExecution(
            controller_v1_pbs.Request_StartExecution(sessionInfo=self._session_info)
        )

    def _stop_execution(self) -> None:
        self._get_stub().StopExecution(
            controller_v1_pbs.Request_StopExecution(sessionInfo=self._session_info)
        )

    @telemetry.record_call()
    @override
    def screenshot(self, report: bool = True, unscaled: bool = False) -> Image.Image:
        """
        Take a screenshot of the current screen.

        Args:
            report (bool, optional): Whether to include the screenshot in reporting.
                Defaults to `True`.
            unscaled (bool, optional): Accepted for interface compatibility. This
                client always returns the native screen resolution, so it has no
                effect. Defaults to `False`.

        Returns:
            Image.Image: A PIL Image object containing the screenshot.

        """
        screenResponse = self._get_stub().CaptureScreen(
            controller_v1_pbs.Request_CaptureScreen(
                sessionInfo=self._session_info,
                captureParameters=controller_v1_pbs.CaptureParameters(
                    displayID=self._display
                ),
            )
        )
        r, g, b, _ = Image.frombytes(
            "RGBA",
            (screenResponse.bitmap.width, screenResponse.bitmap.height),
            screenResponse.bitmap.data,
        ).split()
        image = Image.merge("RGB", (b, g, r))
        if report:
            self._reporter.add_message("AgentOS", "screenshot()", image)
        return image

    @telemetry.record_call()
    @override
    def mouse_move(self, x: int, y: int, duration: int = 500) -> None:
        """
        Moves the mouse cursor to specified screen coordinates.

        Args:
            x (int): The horizontal coordinate (in pixels) to move to.
            y (int): The vertical coordinate (in pixels) to move to.
            duration (int): The duration (in ms) the movement should take.
        """
        self._reporter.add_message(
            "AgentOS",
            f"mouse_move({x}, {y}, duration={duration})",
            AnnotatedImage(lambda: self.screenshot(report=False), point_list=[(x, y)]),
        )
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_MouseMove,
            action_parameters=controller_v1_pbs.ActionParameters(
                mouseMove=controller_v1_pbs.ActionParameters_MouseMove(
                    position=controller_v1_pbs.Coordinate2(x=x, y=y),
                    milliseconds=duration,
                )
            ),
        )

    @telemetry.record_call(exclude={"text"})
    @override
    def type(self, text: str, typing_speed: int = 50) -> None:
        """
        Type text at current cursor position as if entered on a keyboard.

        Args:
            text (str): The text to type.
            typing_speed (int, optional): The speed of typing in characters per second.
                Defaults to `50`.
        """
        self._reporter.add_message("AgentOS", f'type("{text}", {typing_speed})')
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_KeyboardType_UnicodeText,
            action_parameters=controller_v1_pbs.ActionParameters(
                keyboardTypeUnicodeText=controller_v1_pbs.ActionParameters_KeyboardType_UnicodeText(
                    text=text.encode("utf-16-le"),
                    typingSpeed=typing_speed,
                    typingSpeedValue=controller_v1_pbs.TypingSpeedValue.TypingSpeedValue_CharactersPerSecond,
                )
            ),
        )

    @telemetry.record_call()
    @override
    def click(
        self, button: Literal["left", "middle", "right"] = "left", count: int = 1
    ) -> None:
        """
        Click a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse button to
                click. Defaults to `"left"`.
            count (int, optional): Number of times to click. Defaults to `1`.
        """
        self._reporter.add_message("AgentOS", f'click("{button}", {count})')
        mouse_button = None
        match button:
            case "left":
                mouse_button = controller_v1_pbs.MouseButton_Left
            case "middle":
                mouse_button = controller_v1_pbs.MouseButton_Middle
            case "right":
                mouse_button = controller_v1_pbs.MouseButton_Right
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_MouseButton_PressAndRelease,
            action_parameters=controller_v1_pbs.ActionParameters(
                mouseButtonPressAndRelease=controller_v1_pbs.ActionParameters_MouseButton_PressAndRelease(
                    mouseButton=mouse_button, count=count
                )
            ),
        )

    @telemetry.record_call()
    @override
    def mouse_down(self, button: Literal["left", "middle", "right"] = "left") -> None:
        """
        Press and hold a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse button to
                press. Defaults to `"left"`.
        """
        self._reporter.add_message("AgentOS", f'mouse_down("{button}")')
        mouse_button = None
        match button:
            case "left":
                mouse_button = controller_v1_pbs.MouseButton_Left
            case "middle":
                mouse_button = controller_v1_pbs.MouseButton_Middle
            case "right":
                mouse_button = controller_v1_pbs.MouseButton_Right
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_MouseButton_Press,
            action_parameters=controller_v1_pbs.ActionParameters(
                mouseButtonPress=controller_v1_pbs.ActionParameters_MouseButton_Press(
                    mouseButton=mouse_button
                )
            ),
        )

    @telemetry.record_call()
    @override
    def mouse_up(self, button: Literal["left", "middle", "right"] = "left") -> None:
        """
        Release a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse button to
                release. Defaults to `"left"`.
        """
        self._reporter.add_message("AgentOS", f'mouse_up("{button}")')
        mouse_button = None
        match button:
            case "left":
                mouse_button = controller_v1_pbs.MouseButton_Left
            case "middle":
                mouse_button = controller_v1_pbs.MouseButton_Middle
            case "right":
                mouse_button = controller_v1_pbs.MouseButton_Right
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_MouseButton_Release,
            action_parameters=controller_v1_pbs.ActionParameters(
                mouseButtonRelease=controller_v1_pbs.ActionParameters_MouseButton_Release(
                    mouseButton=mouse_button
                )
            ),
        )

    @telemetry.record_call()
    @override
    def mouse_scroll(self, dx: int, dy: int) -> None:
        """
        Scroll the mouse wheel.

        Args:
            dx (int): The horizontal scroll amount. Positive values scroll right,
                negative values scroll left.
            dy (int): The vertical scroll amount. Positive values scroll down,
                negative values scroll up.
        """
        self._reporter.add_message("AgentOS", f"mouse_scroll({dx}, {dy})")
        if dx != 0:
            self._run_recorder_action(
                acion_class_id=controller_v1_pbs.ActionClassID_MouseWheelScroll,
                action_parameters=controller_v1_pbs.ActionParameters(
                    mouseWheelScroll=controller_v1_pbs.ActionParameters_MouseWheelScroll(
                        direction=controller_v1_pbs.MouseWheelScrollDirection.MouseWheelScrollDirection_Horizontal,
                        deltaType=controller_v1_pbs.MouseWheelDeltaType.MouseWheelDelta_Raw,
                        delta=dx,
                        milliseconds=50,
                    )
                ),
            )
        if dy != 0:
            self._run_recorder_action(
                acion_class_id=controller_v1_pbs.ActionClassID_MouseWheelScroll,
                action_parameters=controller_v1_pbs.ActionParameters(
                    mouseWheelScroll=controller_v1_pbs.ActionParameters_MouseWheelScroll(
                        direction=controller_v1_pbs.MouseWheelScrollDirection.MouseWheelScrollDirection_Vertical,
                        deltaType=controller_v1_pbs.MouseWheelDeltaType.MouseWheelDelta_Raw,
                        delta=dy,
                        milliseconds=50,
                    )
                ),
            )

    @telemetry.record_call()
    @override
    def keyboard_pressed(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Press and hold a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to press.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
        """
        self._reporter.add_message(
            "AgentOS", f'keyboard_pressed("{key}", {modifier_keys})'
        )
        if modifier_keys is None:
            modifier_keys = []
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_KeyboardKey_Press,
            action_parameters=controller_v1_pbs.ActionParameters(
                keyboardKeyPress=controller_v1_pbs.ActionParameters_KeyboardKey_Press(
                    keyName=key, modifierKeyNames=modifier_keys
                )
            ),
        )

    @telemetry.record_call()
    @override
    def keyboard_release(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Release a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to release.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                release along with the main key. Defaults to `None`.
        """
        self._reporter.add_message(
            "AgentOS", f'keyboard_release("{key}", {modifier_keys})'
        )
        if modifier_keys is None:
            modifier_keys = []
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_KeyboardKey_Release,
            action_parameters=controller_v1_pbs.ActionParameters(
                keyboardKeyRelease=controller_v1_pbs.ActionParameters_KeyboardKey_Release(
                    keyName=key, modifierKeyNames=modifier_keys
                )
            ),
        )

    @telemetry.record_call()
    @override
    def keyboard_tap(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> None:
        """
        Press and immediately release a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to tap.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
            count (int, optional): The number of times to tap the key. Defaults to `1`.
        """
        self._reporter.add_message(
            "AgentOS",
            f'keyboard_tap("{key}", {modifier_keys}, {count})',
        )
        if modifier_keys is None:
            modifier_keys = []
        for _ in range(count):
            self._run_recorder_action(
                acion_class_id=controller_v1_pbs.ActionClassID_KeyboardKey_PressAndRelease,
                action_parameters=controller_v1_pbs.ActionParameters(
                    keyboardKeyPressAndRelease=controller_v1_pbs.ActionParameters_KeyboardKey_PressAndRelease(
                        keyName=key, modifierKeyNames=modifier_keys
                    )
                ),
            )

    @telemetry.record_call()
    @override
    def set_display(self, display: int = 1) -> None:
        """
        Set the active display.

        Args:
            display (int, optional): The display ID to set as active.
                This can be either a real display ID or a virtual display ID.
                Defaults to `1`.
        """
        self._get_stub().SetActiveDisplay(
            controller_v1_pbs.Request_SetActiveDisplay(displayID=display)
        )
        self._display = display
        self._reporter.add_message("AgentOS", f"set_display({display})")

    @telemetry.record_call(exclude={"command"})
    @override
    def run_command(self, command: str, timeout_ms: int = 30000) -> None:
        """
        Execute a shell command.

        Args:
            command (str): The command to execute.
            timeout_ms (int, optional): The timeout for command
                execution in milliseconds. Defaults to `30000` (30 seconds).
        """
        self._reporter.add_message("AgentOS", f'run_command("{command}", {timeout_ms})')
        self._run_recorder_action(
            acion_class_id=controller_v1_pbs.ActionClassID_RunCommand,
            action_parameters=controller_v1_pbs.ActionParameters(
                runcommand=controller_v1_pbs.ActionParameters_RunCommand(
                    command=command, timeoutInMilliseconds=timeout_ms
                )
            ),
        )

    @telemetry.record_call()
    @override
    def retrieve_active_display(self) -> Display:
        """
        Retrieve the currently active display/screen.

        Returns:
            Display: The currently active display/screen.
        """
        self._reporter.add_message("AgentOS", "retrieve_active_display()")
        displays_list_response = self.list_displays()
        for display in displays_list_response.data:
            if display.id == self._display:
                self._reporter.add_message(
                    "AgentOS", f"retrieve_active_display() -> {display}"
                )
                return display
        error_msg = f"Display {self._display} not found"
        raise ValueError(error_msg)

    @telemetry.record_call()
    @override
    def list_displays(
        self,
    ) -> DisplaysListResponse:
        """
        List all available Displays from the controller.
        It includes both real and virtual displays
            without describing the type of display (virtual or real).

        Returns:
            DisplaysListResponse
        """

        self._reporter.add_message("AgentOS", "list_displays()")

        response: controller_v1_pbs.Response_GetDisplayInformation = (
            self._get_stub().GetDisplayInformation(controller_v1_pbs.Request_Void())
        )

        response_dict = MessageToDict(
            response,
            preserving_proto_field_name=True,
        )

        displays = DisplaysListResponse.model_validate(response_dict)

        self._reporter.add_message("AgentOS", f"list_displays() ->{str(displays)}")

        return displays

    @telemetry.record_call()
    def get_process_list(
        self, get_extended_info: bool = False
    ) -> controller_v1_pbs.Response_GetProcessList:
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

        self._reporter.add_message("AgentOS", f"get_process_list({get_extended_info})")

        response: controller_v1_pbs.Response_GetProcessList = (
            self._get_stub().GetProcessList(
                controller_v1_pbs.Request_GetProcessList(
                    getExtendedInfo=get_extended_info
                )
            )
        )
        self._reporter.add_message(
            "AgentOS", f"get_process_list({get_extended_info}) -> {response}"
        )

        return response

    @telemetry.record_call()
    def get_window_list(
        self, process_id: int
    ) -> controller_v1_pbs.Response_GetWindowList:
        """
        Get a list of windows for a specific process.

        Args:
            process_id (int): The ID of the process to get windows for.

        Returns:
            controller_v1_pbs.Response_GetWindowList: Window list response containing:
                - windows: List of WindowInfo objects with ID and name
        """

        self._reporter.add_message("AgentOS", f"get_window_list({process_id})")

        response: controller_v1_pbs.Response_GetWindowList = (
            self._get_stub().GetWindowList(
                controller_v1_pbs.Request_GetWindowList(processID=process_id)
            )
        )

        self._reporter.add_message(
            "AgentOS", f"get_window_list({process_id}) -> {response}"
        )

        return response

    @telemetry.record_call()
    def get_automation_target_list(
        self,
    ) -> controller_v1_pbs.Response_GetAutomationTargetList:
        """
        Get a list of available automation targets.

        Returns:
            controller_v1_pbs.Response_GetAutomationTargetList:
                Automation target list response:
                - targets: List of AutomationTarget objects
        """

        self._reporter.add_message("AgentOS", "get_automation_target_list()")

        response: controller_v1_pbs.Response_GetAutomationTargetList = (
            self._get_stub().GetAutomationTargetList(controller_v1_pbs.Request_Void())
        )
        self._reporter.add_message(
            "AgentOS", f"get_automation_target_list() -> {response}"
        )

        return response

    @telemetry.record_call()
    def set_mouse_delay(self, delay_ms: int) -> None:
        """
        Configure mouse action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for mouse actions.
        """

        self._reporter.add_message("AgentOS", f"set_mouse_delay({delay_ms})")

        self._get_stub().SetMouseDelay(
            controller_v1_pbs.Request_SetMouseDelay(
                sessionInfo=self._session_info, delayInMilliseconds=delay_ms
            )
        )

    @telemetry.record_call()
    def set_keyboard_delay(self, delay_ms: int) -> None:
        """
        Configure keyboard action delay.

        Args:
            delay_ms (int): The delay in milliseconds to set for keyboard actions.
        """

        self._reporter.add_message("AgentOS", f"set_keyboard_delay({delay_ms})")

        self._get_stub().SetKeyboardDelay(
            controller_v1_pbs.Request_SetKeyboardDelay(
                sessionInfo=self._session_info, delayInMilliseconds=delay_ms
            )
        )

    @telemetry.record_call()
    def set_active_window(self, process_id: int, window_id: int) -> int:
        """
        Set the active window for automation.
        Adds the window as a virtual display and returns the display ID.
        It raises an error if display length is not increased after adding the window.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.

        returns:
            int: The new Display ID.
        Raises:
            AskUiControllerError:
            If display length is not increased after adding the window.
        """

        self._reporter.add_message(
            "AgentOS", f"set_active_window({process_id}, {window_id})"
        )

        display_length_before_adding_window = len(self.list_displays().data)

        self._get_stub().SetActiveWindow(
            controller_v1_pbs.Request_SetActiveWindow(
                processID=process_id, windowID=window_id
            )
        )
        new_display_length = len(self.list_displays().data)
        if new_display_length <= display_length_before_adding_window:
            msg = f"Failed to set active window {window_id} for process {process_id}"
            raise AskUiControllerError(msg)
        self._reporter.add_message(
            "AgentOS",
            f"set_active_window({process_id}, {window_id}) -> {new_display_length}",
        )
        return new_display_length

    @telemetry.record_call()
    def set_active_automation_target(self, target_id: int) -> None:
        """
        Set the active automation target.

        Args:
            target_id (int): The ID of the automation target to set as active.
        """

        self._reporter.add_message(
            "AgentOS", f"set_active_automation_target({target_id})"
        )

        self._get_stub().SetActiveAutomationTarget(
            controller_v1_pbs.Request_SetActiveAutomationTarget(ID=target_id)
        )

    @telemetry.record_call()
    def schedule_batched_action(
        self,
        action_class_id: controller_v1_pbs.ActionClassID,
        action_parameters: controller_v1_pbs.ActionParameters,
    ) -> controller_v1_pbs.Response_ScheduleBatchedAction:
        """
        Schedule an action for batch execution.

        Args:
            action_class_id (controller_v1_pbs.ActionClassID): The class ID
                of the action to schedule.
            action_parameters (controller_v1_pbs.ActionParameters):
                Parameters for the action.

        Returns:
            controller_v1_pbs.Response_ScheduleBatchedAction: Response containing
                the scheduled action ID.
        """

        self._reporter.add_message(
            "AgentOS",
            f"schedule_batched_action({action_class_id}, {action_parameters})",
        )

        response: controller_v1_pbs.Response_ScheduleBatchedAction = (
            self._get_stub().ScheduleBatchedAction(
                controller_v1_pbs.Request_ScheduleBatchedAction(
                    sessionInfo=self._session_info,
                    actionClassID=action_class_id,
                    actionParameters=action_parameters,
                )
            )
        )

        return response

    @telemetry.record_call()
    def start_batch_run(self) -> None:
        """
        Start executing batched actions.
        """

        self._reporter.add_message("AgentOS", "start_batch_run()")

        self._get_stub().StartBatchRun(
            controller_v1_pbs.Request_StartBatchRun(sessionInfo=self._session_info)
        )

    @telemetry.record_call()
    def stop_batch_run(self) -> None:
        """
        Stop executing batched actions.
        """

        self._reporter.add_message("AgentOS", "stop_batch_run()")

        self._get_stub().StopBatchRun(
            controller_v1_pbs.Request_StopBatchRun(sessionInfo=self._session_info)
        )

    @telemetry.record_call()
    def get_action_count(self) -> controller_v1_pbs.Response_GetActionCount:
        """
        Get the count of recorded or batched actions.

        Returns:
            controller_v1_pbs.Response_GetActionCount: Response
                containing the action count.
        """

        response: controller_v1_pbs.Response_GetActionCount = (
            self._get_stub().GetActionCount(
                controller_v1_pbs.Request_GetActionCount(sessionInfo=self._session_info)
            )
        )
        self._reporter.add_message("AgentOS", f"get_action_count() -> {response}")
        return response

    @telemetry.record_call()
    def get_action(self, action_index: int) -> controller_v1_pbs.Response_GetAction:
        """
        Get a specific action by its index.

        Args:
            action_index (int): The index of the action to retrieve.

        Returns:
            controller_v1_pbs.Response_GetAction: Action information containing:
                - actionID: The action ID
                - actionClassID: The action class ID
                - actionParameters: The action parameters
        """

        self._reporter.add_message("AgentOS", f"get_action({action_index})")

        response: controller_v1_pbs.Response_GetAction = self._get_stub().GetAction(
            controller_v1_pbs.Request_GetAction(
                sessionInfo=self._session_info, actionIndex=action_index
            )
        )

        return response

    @telemetry.record_call()
    def remove_action(self, action_id: int) -> None:
        """
        Remove a specific action by its ID.

        Args:
            action_id (int): The ID of the action to remove.
        """

        self._reporter.add_message("AgentOS", f"remove_action({action_id})")

        self._get_stub().RemoveAction(
            controller_v1_pbs.Request_RemoveAction(
                sessionInfo=self._session_info, actionID=action_id
            )
        )

    @telemetry.record_call()
    def remove_all_actions(self) -> None:
        """
        Clear all recorded or batched actions.
        """

        self._reporter.add_message("AgentOS", "remove_all_actions()")

        self._get_stub().RemoveAllActions(
            controller_v1_pbs.Request_RemoveAllActions(sessionInfo=self._session_info)
        )

    def _send_command(self, command: Command) -> AskUIAgentOSSendResponseSchema:
        """
        Send a general command to the controller.

        Args:
            command (Command): The command to send to the controller.

        Returns:
            AskUIAgentOSSendResponseSchema: Response containing
                the message from the controller.

        Raises:
            AskUiControllerInvalidCommandError: If the command fails schema validation
                on the server side.
        """

        header = Header(authentication=Guid(root=self._session_guid))
        message = Message(header=header, command=command)

        request = AskUIAgentOSSendRequestSchema(message=message)

        request_str = request.model_dump_json(exclude_none=True, by_alias=True)

        try:
            response: controller_v1_pbs.Response_Send = self._get_stub().Send(
                controller_v1_pbs.Request_Send(message=request_str)
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                details = e.details() or None
                raise AskUiControllerInvalidCommandError(details) from e
            raise

        return AskUIAgentOSSendResponseSchema.model_validate_json(response.message)

    @telemetry.record_call()
    def get_mouse_position(self) -> Coordinate:
        """
        Get the mouse cursor position

        Returns:
            Coordinate: Response containing the result of the mouse position change.
        """
        self._reporter.add_message("AgentOS", "get_mouse_position()")
        res = self._send_command(GetMousePositionCommand())
        coordinate = Coordinate(
            x=res.message.command.response.position.x.root,  # type: ignore[union-attr]
            y=res.message.command.response.position.y.root,  # type: ignore[union-attr]
        )
        self._reporter.add_message("AgentOS", f"get_mouse_position() -> {coordinate}")
        return coordinate

    @telemetry.record_call()
    def set_mouse_position(self, x: int, y: int) -> None:
        """
        Set the mouse cursor position to specific coordinates.

        Args:
            x (int): The horizontal coordinate (in pixels) to set the cursor to.
            y (int): The vertical coordinate (in pixels) to set the cursor to.
        """
        location = Location(x=Length(root=x), y=Length(root=y))
        command = SetMousePositionCommand(parameters=[location])
        self._reporter.add_message("AgentOS", f"set_mouse_position({x},{y})")
        self._send_command(command)

    @telemetry.record_call()
    def render_quad(self, style: RenderObjectStyle) -> int:
        """
        Render a quad object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the quad.

        Returns:
            int: Object ID.
        """
        self._reporter.add_message("AgentOS", f"render_quad({style})")
        command = AddRenderObjectCommand(parameters=["Quad", style])
        res = self._send_command(command)
        return int(res.message.command.response.id.root)  # type: ignore[union-attr]

    @telemetry.record_call()
    def render_line(self, style: RenderObjectStyle, points: list[Coordinate]) -> int:
        """
        Render a line object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the line.
            points (list[Coordinates]): The points defining the line.

        Returns:
            int: Object ID.
        """
        self._reporter.add_message("AgentOS", f"render_line({style}, {points})")
        command = AddRenderObjectCommand(parameters=["Line", style, points])
        res = self._send_command(command)
        return int(res.message.command.response.id.root)  # type: ignore[union-attr]

    @telemetry.record_call(exclude={"image_data"})
    def render_image(self, style: RenderObjectStyle, image_data: str) -> int:
        """
        Render an image object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the image.
            image_data (str): The base64-encoded image data.

        Returns:
            int: Object ID.
        """
        self._reporter.add_message("AgentOS", f"render_image({style}, [image_data])")
        image = RenderImage(root=image_data)
        command = AddRenderObjectCommand(parameters=["Image", style, image])
        res = self._send_command(command)

        return int(res.message.command.response.id.root)  # type: ignore[union-attr]

    @telemetry.record_call()
    def render_text(self, style: RenderObjectStyle, content: str) -> int:
        """
        Render a text object to the display.

        Args:
            style (RenderObjectStyle): The style properties for the text.
            content (str): The text content to display.

        Returns:
            int: Object ID.
        """
        self._reporter.add_message("AgentOS", f"render_text({style}, {content})")
        text = RenderText(root=content)
        command = AddRenderObjectCommand(parameters=["Text", style, text])
        res = self._send_command(command)
        return int(res.message.command.response.id.root)  # type: ignore[union-attr]

    @telemetry.record_call()
    def update_render_object(self, object_id: int, style: RenderObjectStyle) -> None:
        """
        Update styling properties of an existing render object.

        Args:
            object_id (float): The ID of the render object to update.
            style (RenderObjectStyle): The new style properties.

        Returns:
            int: Object ID.
        """
        self._reporter.add_message(
            "AgentOS", f"update_render_object({object_id}, {style})"
        )
        render_object_id = RenderObjectId(root=object_id)
        command = UpdateRenderObjectCommand(parameters=[render_object_id, style])
        self._send_command(command)

    @telemetry.record_call()
    def delete_render_object(self, object_id: int) -> None:
        """
        Delete an existing render object from the display.

        Args:
            object_id (RenderObjectId): The ID of the render object to delete.
        """
        self._reporter.add_message("AgentOS", f"delete_render_object({object_id})")
        render_object_id = RenderObjectId(root=object_id)
        command = DeleteRenderObjectCommand(parameters=[render_object_id])
        self._send_command(command)

    @telemetry.record_call()
    def clear_render_objects(self) -> None:
        """
        Clear all render objects from the display.
        """
        self._reporter.add_message("AgentOS", "clear_render_objects()")
        command = ClearRenderObjectsCommand()
        self._send_command(command)

    def get_system_info(self) -> GetSystemInfoResponseModel:
        """
        Get the system information.

        Returns:
            SystemInfo: The system information.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", "get_system_info()")
        command = GetSystemInfoCommand()
        res = self._send_command(command).message.command
        if not isinstance(res, GetSystemInfoResponse):
            message = f"unexpected response type: {res}"
            raise DesktopAgentOsError(message)
        self._reporter.add_message("AgentOS", f"get_system_info() -> {res.response}")
        return res.response

    def get_active_process(self) -> GetActiveProcessResponseModel:
        """
        Get the active process.

        Returns:
            GetActiveProcessResponseModel: The active process.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", "get_active_process()")
        command = GetActiveProcessCommand()
        res = self._send_command(command).message.command
        if not isinstance(res, GetActiveProcessResponse):
            message = f"unexpected response type: {res}"
            raise DesktopAgentOsError(message)
        self._reporter.add_message("AgentOS", f"get_active_process() -> {res.response}")
        return res.response

    def set_active_process(self, process_id: int) -> None:
        """
        Set the active process.

        Args:
            process_id (int): The ID of the process to set as active.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", f"set_active_process({process_id})")
        _process_id = Parameter3(root=process_id)
        command = SetActiveProcessCommand(parameters=[_process_id])
        self._send_command(command)

    def get_active_window(self) -> GetActiveWindowResponseModel:
        """
        Gets the window id and name in addition to the process id
             and name of the currently active window (in focus).


        Returns:
            GetActiveWindowResponseModel: The active window.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", "get_active_window()")
        command = GetActiveWindowCommand()
        res = self._send_command(command).message.command
        if not isinstance(res, GetActiveWindowResponse):
            message = f"unexpected response type: {res}"
            raise DesktopAgentOsError(message)
        self._reporter.add_message("AgentOS", f"get_active_window() -> {res.response}")
        return res.response

    def set_window_in_focus(self, process_id: int, window_id: int) -> None:
        """
        Sets the window with the specified windowId of the process
            with the specified processId active,
            which brings it to the front and gives it focus.

        Args:
            process_id (int): The ID of the process that owns the window.
            window_id (int): The ID of the window to set as active.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message(
            "AgentOS", f"set_window_in_focus({process_id}, {window_id})"
        )
        _process_id = Parameter3(root=process_id)
        _window_id = Parameter3(root=window_id)
        command = SetActiveWindowCommand(parameters=[_process_id, _window_id])
        self._send_command(command)

    def get_file_names(self, absolute_directory_path: str) -> list[str]:
        """
        Get the file names in the given absolute directory on the device under
        automation.

        Args:
            absolute_directory_path (str): The absolute directory path to list
                file names from.

        Returns:
            list[str]: The file names returned by the controller.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message(
            "AgentOS", f"get_file_names({absolute_directory_path})"
        )
        command = GetFileNamesCommand(parameters=[absolute_directory_path])
        res = self._send_command(command).message.command
        if not isinstance(res, GetFileNamesResponse):
            message = f"unexpected response type: {res}"
            raise DesktopAgentOsError(message)
        if res.error is not None:
            raise DesktopAgentOsError(res.error)
        if res.response is None:
            message = f"{type(res).__name__} is missing both error and response"
            raise DesktopAgentOsError(message)
        self._reporter.add_message(
            "AgentOS", f"get_file_names({absolute_directory_path}) -> {res.response}"
        )
        return res.response.fileNames

    def get_file(self, path: str) -> Image.Image | str:
        """
        Get the contents of a file at the given path on the device under
        automation.

        The controller returns the file as a Base64-encoded string, which is
        decoded and returned as `PIL.Image.Image` when the bytes can be opened
        as an image (PNG, JPEG, BMP, GIF, WebP, TIFF, ...), or as `str` when
        they decode cleanly as UTF-8 text.

        Args:
            path (str): The file path to read on the device under automation.

        Returns:
            Image.Image | str: The decoded file contents.

        Raises:
            DesktopAgentOsError: If the file cannot be read or the response is invalid.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", f"get_file({path})")
        command = GetFileCommand(parameters=[path])
        res = self._send_command(command).message.command
        if not isinstance(res, GetFileResponse):
            message = f"unexpected response type: {res}"
            raise DesktopAgentOsError(message)
        if res.error is not None:
            raise DesktopAgentOsError(res.error)
        if res.response is None:
            message = f"{type(res).__name__} is missing both error and response"
            raise DesktopAgentOsError(message)
        decoded = self._decode_file_payload(res.response.file.content)
        if isinstance(decoded, Image.Image):
            detail = f"image ({decoded.format}, {decoded.size[0]}x{decoded.size[1]})"
            self._reporter.add_message(
                "AgentOS", f"get_file({path}) -> {detail}", decoded
            )
            return decoded

        detail = f"text ({len(decoded)} chars)"
        self._reporter.add_message("AgentOS", f"get_file({path}) -> {detail}")
        return decoded

    def remove_virtual_displays(self) -> None:
        """
        Remove all virtual displays from the controller, leaving only real
        displays active.
        """
        assert isinstance(self._stub, controller_v1.ControllerAPIStub), (
            "Stub is not initialized"
        )
        self._reporter.add_message("AgentOS", "remove_virtual_displays()")
        command = RemoveVirtualDisplaysCommand()
        self._send_command(command)
        self._reporter.add_message("AgentOS", "remove_virtual_displays() -> done")

    @staticmethod
    def _decode_file_payload(base64_data: str) -> Image.Image | str:
        try:
            return base64_to_image(base64_data)
        except ValueError:
            pass
        data = base64.b64decode(base64_data, validate=True)
        if b"\x00" not in data:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        message = "File contents are neither a supported image nor UTF-8 text"
        raise DesktopAgentOsError(message)
