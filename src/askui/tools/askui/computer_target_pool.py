import logging
from dataclasses import dataclass

import grpc

from askui.tools.askui.agent_os_target_computer import (
    ComputerTarget,
    LocalComputerTarget,
    RemoteComputerTarget,
)
from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2 as controller_v1_pbs,
)
from askui.tools.askui.askui_ui_controller_grpc.generated import (
    Controller_V1_pb2_grpc as controller_v1,
)
from askui.tools.askui.exceptions import AskUiControllerError

logger = logging.getLogger(__name__)


@dataclass
class _Connection:
    """gRPC connection state for a single computer target."""

    target: ComputerTarget
    channel: grpc.Channel
    stub: controller_v1.ControllerAPIStub
    session_info: controller_v1_pbs.SessionInfo
    started_process: bool


class ComputerTargetPool:
    """
    Manages a collection of `ComputerTarget` instances and their gRPC
    connections, and tracks the currently active one.

    Responsibilities:
        - Register / unregister `ComputerTarget` instances with uniqueness
          constraints (at most one local, unique computer ids / session GUIDs,
          unique remote addresses).
        - Open and close gRPC channels and sessions to each registered target.
        - Track which registered target is currently active and expose the stub /
          session info needed to route agent-os actions to it.

    The first target added becomes active by default. Use `switch` to change
    which target is active. `connect_all` opens connections to every registered
    target; subsequently `add` / `add_remote` / `switch` auto-connect any
    newly-introduced target whenever the manager already holds at least one
    open connection.

    Targets are addressed exclusively by their `computer_id`.

    Args:
        agent_os_target_computers (list[ComputerTarget] | None, optional):
            Initial targets to register.
    """

    def __init__(
        self,
        agent_os_target_computers: list[ComputerTarget] | None = None,
    ) -> None:
        # Single store. Python dicts preserve insertion order, so this also
        # defines `list()` order and the first-added-is-active semantics.
        self._by_computer_id: dict[str, ComputerTarget] = {}
        # Open gRPC connections, keyed by `computer_id`.
        self._connections: dict[str, _Connection] = {}
        self._active_computer_id: str | None = None
        if agent_os_target_computers:
            for target in agent_os_target_computers:
                self.add(target)

    @property
    def is_connected(self) -> bool:
        """`True` when at least one gRPC connection is open."""
        return bool(self._connections)

    def add(self, target: ComputerTarget) -> ComputerTarget:
        """
        Register an Agent OS target computer. Auto-connects when the manager
        already has at least one open connection.

        Args:
            target (ComputerTarget): The target computer to register.

        Returns:
            ComputerTarget: The registered target.

        Raises:
            ValueError: If another local target is already registered, the same
                session GUID or computer id is already registered, or another
                remote target with the same address is already registered.
        """
        self._validate_addable(target)
        self._by_computer_id[target.computer_id] = target
        if self._active_computer_id is None:
            self._active_computer_id = target.computer_id
        if self.is_connected:
            self.connect(target)
        return target

    def add_remote(
        self,
        address: str,
        description: str,
    ) -> RemoteComputerTarget:
        """
        Convenience method to construct and register a remote Agent OS target
        computer. Auto-connects when the manager already has at least one open
        connection.

        Args:
            address (str): gRPC address of the remote Agent OS target computer.
            description (str): Human-readable description.

        Returns:
            RemoteComputerTarget: The newly registered target.
        """
        target = RemoteComputerTarget(address=address, description=description)
        self.add(target)
        return target

    def reset(self) -> None:
        """Disconnect every open connection and remove all registered targets."""
        self.disconnect_all()
        self._by_computer_id.clear()
        self._active_computer_id = None

    def remove(self, computer_id: str) -> None:
        """
        Remove a registered target by its `computer_id`. If the target was
        connected, its connection is closed first.

        Args:
            computer_id (str): The computer id of the target to remove.

        Raises:
            KeyError: If no target with the given computer id is registered.
        """
        self._require(computer_id)
        self.disconnect(computer_id)
        del self._by_computer_id[computer_id]
        if self._active_computer_id == computer_id:
            self._active_computer_id = next(iter(self._by_computer_id), None)

    def list(self) -> list[ComputerTarget]:
        """Return all registered targets in registration order."""
        return list(self._by_computer_id.values())

    def get(self, computer_id: str) -> ComputerTarget:
        """
        Return the registered target with the given `computer_id`.

        Raises:
            KeyError: If no target with the given computer id is registered.
        """
        return self._require(computer_id)

    def switch(self, computer_id: str) -> ComputerTarget:
        """
        Set the active target by its `computer_id`. Auto-connects the new
        active target when the manager already has at least one open connection
        but this target is not yet connected.

        Args:
            computer_id (str): The computer id of the target to activate.

        Returns:
            ComputerTarget: The newly active target.

        Raises:
            KeyError: If no target with the given computer id is registered.
        """
        target = self._require(computer_id)
        self._active_computer_id = computer_id
        if self.is_connected and computer_id not in self._connections:
            self.connect(target)
        return target

    @property
    def active(self) -> ComputerTarget | None:
        """The currently active target, or `None` if no targets are registered."""
        if self._active_computer_id is None:
            return None
        return self._by_computer_id.get(self._active_computer_id)

    def require_active(self) -> ComputerTarget:
        """
        Return the currently active target.

        Raises:
            AskUiControllerError: If no target is currently active.
        """
        target = self.active
        if target is None:
            error_msg = (
                "No active Agent OS target computer. Register one via "
                "`MultiComputerTargetAgentOS.add_agent_os_target_computer()` / "
                "`add_remote_agent_os_target_computer()`, or pass "
                "`agent_os_target_computers` to the `MultiComputerTargetAgentOS` "
                "constructor."
            )
            raise AskUiControllerError(error_msg)
        return target

    def active_connection(self) -> _Connection:
        """
        Return the gRPC connection for the currently active target.

        Raises:
            AskUiControllerError: If no target is currently active or the active
                target has no open connection (i.e. `connect_all()` has not been
                called).
        """
        target = self.require_active()
        conn = self._connections.get(target.computer_id)
        if conn is None:
            error_msg = (
                f"Active Agent OS target computer {target.description!r} "
                f"(computer_id={target.computer_id!r}, "
                f"address={target.address}) "
                "is not connected. Call `MultiComputerTargetAgentOS.connect()` first."
            )
            raise AskUiControllerError(error_msg)
        return conn

    def connect_all(self) -> None:
        """
        Open a gRPC channel and session to every registered Agent OS target.

        For each target: starts the local process when `is_local` and
        `is_service` is `False`, opens an insecure gRPC channel, starts a
        session, starts execution, and sets the configured display. Targets
        already connected are skipped, so calling `connect_all()` twice is
        safe.

        Raises:
            AskUiControllerError: If no targets are registered.

        On failure mid-loop, all targets connected so far are rolled back via
        `disconnect_all()` before re-raising.
        """
        if not self._by_computer_id:
            error_msg = (
                "Cannot connect: no Agent OS target computers registered. Provide "
                "at least one via the `MultiComputerTargetAgentOS` constructor's "
                "`agent_os_target_computers` argument, or call "
                "`add_agent_os_target_computer()` / "
                "`add_remote_agent_os_target_computer()` before `connect()`."
            )
            raise AskUiControllerError(error_msg)
        try:
            for target in self._by_computer_id.values():
                self.connect(target)
        except Exception:
            self.disconnect_all()
            raise

    def connect(self, target: ComputerTarget) -> None:
        """
        Open a gRPC channel and session to a single registered Agent OS target.
        Idempotent: returns silently if the target is already connected.
        """
        if target.computer_id in self._connections:
            return
        started_process = False
        if isinstance(target, LocalComputerTarget) and not target.is_service:
            target.start()
            started_process = True
        channel = grpc.insecure_channel(
            target.address,
            options=[
                ("grpc.max_send_message_length", 2**30),
                ("grpc.max_receive_message_length", 2**30),
                ("grpc.default_deadline", 300000),
            ],
        )
        stub = controller_v1.ControllerAPIStub(channel)
        try:
            session_response: controller_v1_pbs.Response_StartSession = (
                stub.StartSession(
                    controller_v1_pbs.Request_StartSession(
                        sessionGUID=target.session_guid,
                        immediateExecution=True,
                    )
                )
            )
            session_info = session_response.sessionInfo
            stub.StartExecution(
                controller_v1_pbs.Request_StartExecution(sessionInfo=session_info)
            )
            stub.SetActiveDisplay(
                controller_v1_pbs.Request_SetActiveDisplay(displayID=target.display)
            )
        except Exception as e:
            try:
                channel.close()
            finally:
                if started_process:
                    target.stop()
            if hasattr(e, "add_note"):
                e.add_note(
                    f"While connecting to Agent OS target computer "
                    f"{target.description!r} "
                    f"(computer_id={target.computer_id!r}, "
                    f"session_guid={target.session_guid}, "
                    f"display={target.display}, "
                    f"address={target.address})"
                )
            raise
        self._connections[target.computer_id] = _Connection(
            target=target,
            channel=channel,
            stub=stub,
            session_info=session_info,
            started_process=started_process,
        )

    def disconnect_all(self) -> None:
        """
        Close every open Agent OS target connection.

        For each connection: stops execution, ends the session, closes the gRPC
        channel, and (only when this manager started the local process) stops
        the controller process. Errors are logged but do not abort the loop -
        a partial failure on one target still releases the others.
        """
        for computer_id in list(self._connections.keys()):
            self.disconnect(computer_id)

    def disconnect(self, computer_id: str) -> None:
        """
        Close a single open Agent OS target connection identified by its
        `computer_id`. No-op if no such connection is open.
        """
        conn = self._connections.pop(computer_id, None)
        if conn is None:
            return
        try:
            conn.stub.StopExecution(
                controller_v1_pbs.Request_StopExecution(sessionInfo=conn.session_info)
            )
            conn.stub.EndSession(
                controller_v1_pbs.Request_EndSession(sessionInfo=conn.session_info)
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error stopping execution/session for controller %s", computer_id
            )
        try:
            conn.channel.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error closing channel for controller %s", computer_id)
        if conn.started_process:
            try:
                conn.target.stop()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error stopping client-started controller process for %s",
                    computer_id,
                )

    def __len__(self) -> int:
        return len(self._by_computer_id)

    def __contains__(self, computer_id: object) -> bool:
        return isinstance(computer_id, str) and computer_id in self._by_computer_id

    def _validate_addable(self, target: ComputerTarget) -> None:
        if target.is_local:
            existing_local = next(
                (t for t in self._by_computer_id.values() if t.is_local), None
            )
            if existing_local is not None:
                error_msg = (
                    "Cannot register a second local Agent OS target computer. At "
                    "most one local target is supported. Existing local target: "
                    f"{existing_local.description!r} "
                    f"(computer_id={existing_local.computer_id!r}). "
                    "Remove it first via `remove(computer_id)`."
                )
                raise ValueError(error_msg)
        if target.computer_id in self._by_computer_id:
            error_msg = (
                "An Agent OS target computer with "
                f"computer_id={target.computer_id!r} is already registered. "
                "Each target must have a unique computer_id."
            )
            raise ValueError(error_msg)
        if not target.is_local and any(
            (not t.is_local) and t.address == target.address
            for t in self._by_computer_id.values()
        ):
            error_msg = (
                f"A remote Agent OS target computer with address "
                f"{target.address!r} is already registered. Each remote target "
                "must have a unique address."
            )
            raise ValueError(error_msg)

    def _require(self, computer_id: str) -> ComputerTarget:
        target = self._by_computer_id.get(computer_id)
        if target is not None:
            return target
        registered = ", ".join(repr(cid) for cid in self._by_computer_id) or "none"
        error_msg = (
            f"No Agent OS target computer with computer_id={computer_id!r} is "
            f"registered. Registered computer ids: {registered}. Use "
            "`list_agent_os_target_computers()` to inspect the registered targets."
        )
        raise KeyError(error_msg)


__all__ = ["ComputerTargetPool"]
