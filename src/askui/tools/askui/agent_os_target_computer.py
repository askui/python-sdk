import logging
import pathlib
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

from typing_extensions import override

from askui.tools.askui.askui_controller_settings import AskUiControllerSettings
from askui.tools.askui.computer_target_connection import ComputerTargetConnection
from askui.tools.askui.exceptions import AskUiControllerError
from askui.tools.utils import process_exists, wait_for_port

logger = logging.getLogger(__name__)


class ComputerTarget:
    """
    Base class describing a computer target (a machine running the AskUI Agent
    OS) that a `MultiComputerTargetAgentOS` client can connect to.

    A computer target runs the server-side counterpart of the `ComputerAgentOS`
    client abstraction: it exposes a gRPC API for OS-level operations
    (screenshot, mouse, keyboard, ...) and is identified by a unique session
    GUID. Each computer target also tracks which display it is currently
    operating against.

    Args:
        address (str): gRPC address of the target computer
            (e.g. ``"localhost:23000"``).
        description (str): Human-readable description.
        display (int, optional): Display ID selected for this target computer.
            Defaults to `1`.
        computer_id (str | None, optional): Stable, human-friendly identifier for
            the target computer. Used by `ComputerTargetPool` lookup
            helpers. Must be unique across registered target computers. Defaults
            to the target computer's `session_guid`.
    """

    def __init__(
        self,
        address: str,
        description: str,
        display: int = 1,
        computer_id: str | None = None,
    ) -> None:
        self._session_guid = "{" + str(uuid.uuid4()) + "}"
        self._address = address
        self._description = description
        self._display = display
        self._computer_id = (
            computer_id if computer_id is not None else self._session_guid
        )
        self._connection: ComputerTargetConnection | None = None

    @property
    def session_guid(self) -> str:
        """Unique session GUID assigned to this target computer."""
        return self._session_guid

    @property
    def computer_id(self) -> str:
        """
        Stable identifier for this target computer. Defaults to `session_guid`
        when no custom id was supplied at construction time.
        """
        return self._computer_id

    @property
    def address(self) -> str:
        """gRPC address of the target computer."""
        return self._address

    @property
    def description(self) -> str:
        """Description of this target computer."""
        return self._description

    @property
    def display(self) -> int:
        """Display ID currently selected for this target computer."""
        return self._display

    @display.setter
    def display(self, value: int) -> None:
        self._display = value

    @property
    def is_local(self) -> bool:
        """Whether this target computer represents a locally-managed process."""
        return False

    @property
    def is_connected(self) -> bool:
        """Whether an open gRPC connection to this target computer exists."""
        return self._connection is not None

    @property
    def connection(self) -> ComputerTargetConnection:
        """
        The open gRPC connection to this target computer.

        Raises:
            AskUiControllerError: If this target computer is not connected (i.e.
                `connect()` has not been called).
        """
        if self._connection is None:
            error_msg = (
                f"Agent OS target computer {self._description!r} "
                f"(computer_id={self._computer_id!r}, address={self._address}) "
                "is not connected. Call `MultiComputerTargetAgentOS.connect()` "
                "first."
            )
            raise AskUiControllerError(error_msg)
        return self._connection

    def connect(self) -> None:
        """
        Open the gRPC connection to this target computer. Idempotent: returns
        silently if already connected. Delegates the gRPC specifics to
        `ComputerTargetConnection.open()`.
        """
        if self._connection is None:
            self._connection = ComputerTargetConnection.open(self)

    def disconnect(self) -> None:
        """
        Close the gRPC connection to this target computer. No-op if not
        connected. Delegates the gRPC teardown to
        `ComputerTargetConnection.close()`.
        """
        conn = self._connection
        if conn is None:
            return
        self._connection = None
        conn.close(self)

    def start(self, clean_up: bool = False) -> None:
        """Start the underlying controller process. No-op for non-local targets."""

    def stop(self, force: bool = False) -> None:
        """Stop the underlying controller process. No-op for non-local targets."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"computer_id={self._computer_id!r}, "
            f"description={self._description!r}, "
            f"display={self._display!r})"
        )


class LocalComputerTarget(ComputerTarget):
    """
    Local computer target: manages an AskUI Remote Device Controller
    subprocess on this machine.

    Args:
        settings (AskUiControllerSettings | None, optional): Process-level settings
            (executable path, args). Defaults to a fresh `AskUiControllerSettings`.
        address (str, optional): gRPC address. Defaults to ``"localhost:23000"``.
        is_service (bool, optional): When `True`, `start()` does not launch the
            controller binary because it is managed externally (e.g. AskUI Core
            Service on Windows). Defaults to `False`.
        discover_service (bool, optional): On Windows, probe for a running
            ``askuicoreservice`` and, if found, switch the address to port
            ``26000`` and set `is_service` to `True`. Defaults to `True`.
        description (str, optional)
        display (int, optional): Display ID selected for this target computer.
            Defaults to `1`.
    """

    _ASKUI_CORE_SERVICE_NAME = "AskuiCoreService"
    _ASKUI_CORE_SERVICE_PORT = 26000

    def __init__(
        self,
        description: str = "Local computer target",
        settings: AskUiControllerSettings | None = None,
        address: str = "localhost:23000",
        discover_service: bool = True,
        display: int = 1,
        computer_id: str | None = None,
    ) -> None:
        super().__init__(
            address=address,
            description=description,
            display=display,
            computer_id=computer_id,
        )
        self._is_service = False
        self._settings = settings or AskUiControllerSettings()
        self._process: subprocess.Popen[bytes] | None = None
        if discover_service:
            self._discover_service(address)

    @property
    @override
    def is_local(self) -> bool:
        return True

    @property
    def is_service(self) -> bool:
        """Whether the controller process is managed externally (skip `start()`)."""
        return self._is_service

    @staticmethod
    def _is_askui_core_service_running() -> bool:
        """Return `True` when the `AskuiCoreService` Windows service is RUNNING."""
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    [
                        "sc",
                        "query",
                        LocalComputerTarget._ASKUI_CORE_SERVICE_NAME,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode != 0:
                    return False
            except (OSError, subprocess.SubprocessError) as e:
                error_msg = (
                    "Failed to query "
                    f"{LocalComputerTarget._ASKUI_CORE_SERVICE_NAME} service: {e}"
                )
                logger.debug(error_msg)
                return False
            return "RUNNING" in result.stdout.upper()
        return False

    def _discover_service(self, address: str) -> None:
        if LocalComputerTarget._is_askui_core_service_running():
            service_msg = (
                f"Detected running {self._ASKUI_CORE_SERVICE_NAME}; using port "
                f"{self._ASKUI_CORE_SERVICE_PORT} (controller managed by service)"
            )
            logger.info(service_msg)
            address = LocalComputerTarget.replace_port(
                address, self._ASKUI_CORE_SERVICE_PORT
            )
            self._is_service = True

    @staticmethod
    def replace_port(address: str, port: int) -> str:
        addr = address if "://" in address else "//" + address
        parsed = urlparse(addr)
        host = parsed.hostname or "localhost"
        return f"{host}:{port}"

    def _parse_port(self) -> int:
        addr = self._address if "://" in self._address else "//" + self._address
        parsed = urlparse(addr)
        if parsed.port is None:
            error_msg = (
                f"Could not parse port from address {self._address!r}. "
                "Expected format 'host:port' (e.g. 'localhost:23000')."
            )
            raise ValueError(error_msg)
        return parsed.port

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
        wait_for_port(self._parse_port())

    @override
    def start(self, clean_up: bool = False) -> None:
        """
        Start the controller process unless this target uses a service-managed
        binary.

        Args:
            clean_up (bool, optional): Whether to clean up existing processes
                (only on Windows) before starting. Defaults to `False`.
        """
        if self._is_service:
            logger.debug(
                "Skipping local controller start; process is managed by service"
            )
            return
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

    @override
    def stop(self, force: bool = False) -> None:
        """
        Stop the controller process.

        Args:
            force (bool, optional): Whether to forcefully terminate the process.
                Defaults to `False`.
        """
        if self._process is None:
            return

        try:
            if force:
                self._process.kill()
                if sys.platform == "win32":
                    self.clean_up()
            else:
                self._process.terminate()
        except Exception:  # noqa: BLE001 - We want to catch all other exceptions here
            logger.exception("Error stopping local controller process")
        finally:
            self._process = None


class RemoteComputerTarget(ComputerTarget):
    """
    Remote computer target: the client connects to an already-running
    controller on another machine.

    No process management is performed; `start()` and `stop()` are no-ops.

    Args:
        address (str): gRPC address of the remote target computer (required).
        description (str): Human-readable description.
        display (int, optional): Display ID selected for this target computer.
            Defaults to `1`.
        computer_id (str | None, optional): Stable, human-friendly identifier for
            the target computer. Defaults to the target computer's
            `session_guid`.
    """

    def __init__(
        self,
        address: str,
        description: str,
        display: int = 1,
        computer_id: str | None = None,
    ) -> None:
        super().__init__(
            address=address,
            description=description,
            display=display,
            computer_id=computer_id,
        )


__all__ = [
    "ComputerTarget",
    "LocalComputerTarget",
    "RemoteComputerTarget",
]
