from askui.tools.askui.agent_os_target_computer import (
    ComputerTarget,
)
from askui.tools.askui.computer_target_connection import ComputerTargetConnection
from askui.tools.askui.exceptions import AskUiControllerError


class ComputerTargetPool:
    """
    Manages a collection of `ComputerTarget` instances and tracks the currently
    active one. Each target owns its own gRPC connection
    (`ComputerTarget.connection`); the pool only orchestrates connecting /
    disconnecting them and selecting the active one.

    Responsibilities:
        - Register / unregister `ComputerTarget` instances with uniqueness
          constraints (at most one local, unique computer ids / session GUIDs,
          unique remote addresses).
        - Drive `connect()` / `disconnect()` on registered targets (individually
          or all at once).
        - Track which registered target is currently active and expose its
          connection needed to route agent-os actions to it.

    The first target added becomes active by default. Use `switch` to change
    which target is active. `connect` opens connections to every registered
    target; subsequently `add` / `switch` auto-connect any
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
        # defines `list()` order and the first-added-is-active semantics. Each
        # target owns its own connection, so no separate connection store is
        # needed here.
        self._by_computer_id: dict[str, ComputerTarget] = {}
        self._active_computer_id: str | None = None
        if agent_os_target_computers:
            for target in agent_os_target_computers:
                self.add(target)

    @property
    def is_connected(self) -> bool:
        """`True` when at least one registered target has an open connection."""
        return any(t.is_connected for t in self._by_computer_id.values())

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
            self.connect_target(target)
        return target

    def reset(self) -> None:
        """Disconnect every open connection and remove all registered targets."""
        self.disconnect()
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
        self.disconnect_target(computer_id)
        del self._by_computer_id[computer_id]
        if self._active_computer_id == computer_id:
            self._active_computer_id = next(iter(self._by_computer_id), None)

    def describe(self) -> list[str]:
        """
        Return the `repr()` of every registered target, in registration order.
        """
        return [repr(target) for target in self._by_computer_id.values()]

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
        if self.is_connected and not target.is_connected:
            self.connect_target(target)
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
                "`MultiComputerTargetAgentOS.add_agent_os_target_computer()`, or "
                "pass `agent_os_target_computers` to the "
                "`MultiComputerTargetAgentOS` constructor."
            )
            raise AskUiControllerError(error_msg)
        return target

    def active_connection(self) -> ComputerTargetConnection:
        """
        Return the gRPC connection for the currently active target.

        Raises:
            AskUiControllerError: If no target is currently active or the active
                target has no open connection (i.e. `connect()` has not been
                called).
        """
        return self.require_active().connection

    def connect(self) -> None:
        """
        Open the connection to every registered Agent OS target via
        `ComputerTarget.connect()`. Targets already connected are skipped, so
        calling `connect()` twice is safe.

        Raises:
            AskUiControllerError: If no targets are registered.

        On failure mid-loop, all targets connected so far are rolled back via
        `disconnect()` before re-raising.
        """
        if not self._by_computer_id:
            error_msg = (
                "Cannot connect: no Agent OS target computers registered. Provide "
                "at least one via the `MultiComputerTargetAgentOS` constructor's "
                "`agent_os_target_computers` argument, or call "
                "`add_agent_os_target_computer()` before `connect()`."
            )
            raise AskUiControllerError(error_msg)
        try:
            for target in self._by_computer_id.values():
                self.connect_target(target)
        except Exception:
            self.disconnect()
            raise

    def connect_target(self, target: ComputerTarget) -> None:
        """
        Open the connection to a single registered Agent OS target. Idempotent:
        returns silently if the target is already connected. Delegates to
        `ComputerTarget.connect()`.
        """
        target.connect()

    def disconnect(self) -> None:
        """
        Close every open Agent OS target connection. Errors on one connection
        are logged but do not abort the loop - a partial failure still releases
        the others.
        """
        for target in self._by_computer_id.values():
            target.disconnect()

    def disconnect_target(self, computer_id: str) -> None:
        """
        Close a single open Agent OS target connection identified by its
        `computer_id`. No-op if no such connection is open or no such target is
        registered. Delegates to `ComputerTarget.disconnect()`.
        """
        target = self._by_computer_id.get(computer_id)
        if target is not None:
            target.disconnect()

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
            "`describe_agent_os_target_computers()` to inspect the registered "
            "targets."
        )
        raise KeyError(error_msg)


__all__ = ["ComputerTargetPool"]
