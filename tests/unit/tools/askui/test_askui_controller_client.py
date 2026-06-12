"""
Unit tests for `MultiComputerTargetAgentOS`'s multi-target registration / routing
logic. These tests intentionally avoid exercising the gRPC code path (which
needs a real controller binary). They cover the in-memory bookkeeping done by
the client and its `ComputerTargetPool`.
"""

import pytest

from askui.tools.askui.agent_os_target_computer import (
    LocalComputerTarget,
    RemoteComputerTarget,
)
from askui.tools.askui.askui_controller import MultiComputerTargetAgentOS
from askui.tools.askui.computer_target_pool import (
    ComputerTargetPool,
)
from askui.tools.askui.exceptions import AskUiControllerError


def _make_local(
    description: str = "local", computer_id: str | None = None, display: int = 1
) -> LocalComputerTarget:
    return LocalComputerTarget(
        description=description,
        discover_service=False,
        computer_id=computer_id,
        display=display,
    )


def _make_remote(
    address: str = "1.2.3.4:23000",
    description: str = "remote",
    computer_id: str | None = None,
    display: int = 1,
) -> RemoteComputerTarget:
    return RemoteComputerTarget(
        address=address,
        description=description,
        computer_id=computer_id,
        display=display,
    )


class TestConstruction:
    def test_default_registers_single_local_target(self) -> None:
        client = MultiComputerTargetAgentOS()
        manager = client.agent_os_target_computer_manager
        assert len(manager) == 1
        assert isinstance(manager.active, LocalComputerTarget)

    def test_default_propagates_display_to_default_local_target(self) -> None:
        client = MultiComputerTargetAgentOS(display=3)
        active = client.agent_os_target_computer_manager.active
        assert active is not None
        assert active.display == 3

    def test_accepts_explicit_targets(self) -> None:
        a = _make_local(computer_id="local")
        b = _make_remote(computer_id="remote")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        assert client.agent_os_target_computer_manager.describe() == [
            repr(a),
            repr(b),
        ]
        assert client.agent_os_target_computer_manager.active is a

    def test_explicit_targets_keep_their_own_display(self) -> None:
        """Constructor's display arg only seeds the auto-created default target."""
        a = _make_local(computer_id="local", display=2)
        b = _make_remote(computer_id="remote", display=3)
        client = MultiComputerTargetAgentOS(display=5, agent_os_target_computers=[a, b])
        assert client.agent_os_target_computer_manager.get("local").display == 2
        assert client.agent_os_target_computer_manager.get("remote").display == 3

    def test_is_connected_false_before_connect(self) -> None:
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[_make_remote()])
        assert client.is_connected is False


class TestActiveTarget:
    def test_get_current_returns_first_registered_id(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        assert client.get_current_computer_target_id() == "a"

    def test_get_current_with_empty_manager_raises(self) -> None:
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[_make_remote()])
        client.agent_os_target_computer_manager.reset()
        with pytest.raises(
            AskUiControllerError, match="No active Agent OS target computer"
        ):
            client.get_current_computer_target_id(report=False)


class TestSwitchAgentOsTargetComputer:
    def test_switch_changes_active_when_disconnected(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        client.switch_agent_os_target_computer("b")
        assert client.agent_os_target_computer_manager.active is b

    def test_switch_unknown_computer_id_raises_keyerror(self) -> None:
        client = MultiComputerTargetAgentOS(
            agent_os_target_computers=[_make_local(computer_id="a")]
        )
        with pytest.raises(KeyError, match="missing"):
            client.switch_agent_os_target_computer("missing")

    def test_switch_returns_the_new_active_target(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        result = client.switch_agent_os_target_computer("b")
        assert result is b

    def test_per_target_display_preserved_across_switch(self) -> None:
        a = _make_local(computer_id="a", display=1)
        b = _make_remote(computer_id="b", display=4)
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        client.switch_agent_os_target_computer("b")
        active_b = client.agent_os_target_computer_manager.active
        assert active_b is not None
        assert active_b.display == 4
        client.switch_agent_os_target_computer("a")
        active_a = client.agent_os_target_computer_manager.active
        assert active_a is not None
        assert active_a.display == 1


class TestDescribeAndReset:
    def test_describe_returns_registered_target_summaries(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        assert client.describe_agent_os_target_computers() == [repr(a), repr(b)]

    def test_reset_with_no_args_leaves_manager_empty(self) -> None:
        client = MultiComputerTargetAgentOS(
            agent_os_target_computers=[_make_remote(computer_id="r")]
        )
        client.reset_agent_os_target_computers()
        assert client.describe_agent_os_target_computers() == []

    def test_reset_with_new_list_replaces_registrations(self) -> None:
        client = MultiComputerTargetAgentOS(
            agent_os_target_computers=[_make_remote(computer_id="old")]
        )
        new_agent_os_target_computer = _make_remote(
            address="9.9.9.9:23000", computer_id="new"
        )
        client.reset_agent_os_target_computers([new_agent_os_target_computer])
        assert client.describe_agent_os_target_computers() == [
            repr(new_agent_os_target_computer)
        ]
        assert (
            client.agent_os_target_computer_manager.active
            is new_agent_os_target_computer
        )


class TestAddAgentOsTargetComputerWhileDisconnected:
    def test_add_already_constructed_target(self) -> None:
        client = MultiComputerTargetAgentOS(
            agent_os_target_computers=[_make_local(computer_id="l")]
        )
        extra = _make_remote(address="2.2.2.2:23000", computer_id="r")
        result = client.add_agent_os_target_computer(extra)
        assert result is extra
        assert repr(extra) in client.describe_agent_os_target_computers()


class TestTemporarySelect:
    def test_temporary_select_restores_previous_active(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        manager = client.agent_os_target_computer_manager
        before = manager.active
        assert before is a
        with client.temporary_select("b"):
            inside = manager.active
            assert inside is b
        after = manager.active
        assert after is a

    def test_temporary_select_restores_previous_even_on_exception(self) -> None:
        a = _make_local(computer_id="a")
        b = _make_remote(computer_id="b")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a, b])
        error_message = "boom"
        with (
            pytest.raises(RuntimeError, match=error_message),
            client.temporary_select("b"),
        ):
            assert client.agent_os_target_computer_manager.active is b
            raise RuntimeError(error_message)
        assert client.agent_os_target_computer_manager.active is a

    def test_temporary_select_same_id_is_a_noop_around_yield(self) -> None:
        a = _make_local(computer_id="a")
        client = MultiComputerTargetAgentOS(agent_os_target_computers=[a])
        with client.temporary_select("a"):
            assert client.agent_os_target_computer_manager.active is a
        assert client.agent_os_target_computer_manager.active is a


class TestUsesAgentOsTargetComputerManager:
    def test_underlying_manager_is_an_agent_os_target_computer_manager(self) -> None:
        client = MultiComputerTargetAgentOS(
            agent_os_target_computers=[_make_local(computer_id="l")]
        )
        assert isinstance(client.agent_os_target_computer_manager, ComputerTargetPool)
