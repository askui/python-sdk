from collections.abc import Callable

import pytest

from askui.tools.askui.agent_os_target_computer import (
    ComputerTarget,
    LocalComputerTarget,
    RemoteComputerTarget,
)
from askui.tools.askui.computer_target_pool import (
    ComputerTargetPool,
)


def _make_remote(
    address: str = "1.2.3.4:23000",
    description: str = "remote",
    computer_id: str | None = None,
) -> RemoteComputerTarget:
    return RemoteComputerTarget(
        address=address, description=description, computer_id=computer_id
    )


def _make_local(computer_id: str | None = None) -> LocalComputerTarget:
    return LocalComputerTarget(discover_service=False, computer_id=computer_id)


@pytest.fixture(params=["local", "remote"])
def make_target(
    request: pytest.FixtureRequest,
) -> Callable[..., ComputerTarget]:
    """Build a single target of the parametrized kind so a test runs once per kind.

    Use for tests that register exactly one target and where the local/remote
    distinction is irrelevant to the behavior under test.
    """

    def _make(
        computer_id: str | None = None,
        address: str = "1.2.3.4:23000",
    ) -> ComputerTarget:
        if request.param == "local":
            return _make_local(computer_id=computer_id)
        return _make_remote(address=address, computer_id=computer_id)

    return _make


class TestConstruction:
    def test_empty_constructor_yields_empty_manager(self) -> None:
        m = ComputerTargetPool()
        assert m.list() == []
        assert m.active is None
        assert len(m) == 0

    def test_constructor_registers_initial_targets_in_order(self) -> None:
        a = _make_remote(address="1.1.1.1:23000", computer_id="a")
        b = _make_remote(address="2.2.2.2:23000", computer_id="b")
        m = ComputerTargetPool(agent_os_target_computers=[a, b])
        assert m.list() == [a, b]
        # First registered becomes active.
        assert m.active is a

    def test_first_added_becomes_active(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        a = make_target(computer_id="a")
        m.add(a)
        assert m.active is a


class TestAddConstraints:
    def test_rejects_second_local_target(self) -> None:
        m = ComputerTargetPool()
        m.add(_make_local(computer_id="first"))
        with pytest.raises(ValueError, match="second local Agent OS target computer"):
            m.add(_make_local(computer_id="second"))

    def test_rejects_duplicate_computer_id(self) -> None:
        m = ComputerTargetPool()
        m.add(_make_remote(address="1.1.1.1:23000", computer_id="rig"))
        with pytest.raises(ValueError, match="computer_id='rig'"):
            m.add(_make_remote(address="2.2.2.2:23000", computer_id="rig"))

    def test_rejects_duplicate_remote_address(self) -> None:
        m = ComputerTargetPool()
        m.add(_make_remote(address="1.1.1.1:23000", computer_id="a"))
        with pytest.raises(
            ValueError,
            match="remote Agent OS target computer with address '1.1.1.1:23000'",
        ):
            m.add(_make_remote(address="1.1.1.1:23000", computer_id="b"))

    def test_allows_local_plus_remote_with_same_address(self) -> None:
        m = ComputerTargetPool()
        m.add(_make_local(computer_id="local"))
        # Local target's default address is 'localhost:23000' but the local/remote
        # address-uniqueness rule only applies between remote targets.
        m.add(
            _make_remote(
                address="localhost:23000", description="remote", computer_id="remote"
            )
        )
        assert len(m) == 2


class TestGetAndSwitch:
    def test_get_returns_target_by_computer_id(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        a = make_target(address="1.1.1.1:23000", computer_id="a")
        m.add(a)
        assert m.get("a") is a

    def test_get_raises_keyerror_with_registered_ids(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        m.add(make_target(address="1.1.1.1:23000", computer_id="a"))
        with pytest.raises(KeyError) as exc_info:
            m.get("missing")
        message = str(exc_info.value)
        assert "missing" in message
        assert "'a'" in message  # registered id surfaced

    def test_switch_changes_active(self) -> None:
        m = ComputerTargetPool()
        a = _make_remote(address="1.1.1.1:23000", computer_id="a")
        b = _make_remote(address="2.2.2.2:23000", computer_id="b")
        m.add(a)
        m.add(b)
        assert m.active is a
        m.switch("b")
        assert m.active is b

    def test_switch_unknown_id_raises_keyerror(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        m.add(make_target(computer_id="a"))
        with pytest.raises(KeyError, match="missing"):
            m.switch("missing")


class TestRemove:
    def test_remove_drops_target(self) -> None:
        m = ComputerTargetPool()
        a = _make_remote(address="1.1.1.1:23000", computer_id="a")
        b = _make_remote(address="2.2.2.2:23000", computer_id="b")
        m.add(a)
        m.add(b)
        m.remove("a")
        assert m.list() == [b]

    def test_remove_active_falls_back_to_first_remaining(self) -> None:
        m = ComputerTargetPool()
        a = _make_remote(address="1.1.1.1:23000", computer_id="a")
        b = _make_remote(address="2.2.2.2:23000", computer_id="b")
        m.add(a)
        m.add(b)
        assert m.active is a
        m.remove("a")
        assert m.active is b

    def test_remove_last_clears_active(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        m.add(make_target(computer_id="a"))
        m.remove("a")
        assert m.active is None
        assert len(m) == 0

    def test_remove_inactive_keeps_active_unchanged(self) -> None:
        m = ComputerTargetPool()
        a = _make_remote(address="1.1.1.1:23000", computer_id="a")
        b = _make_remote(address="2.2.2.2:23000", computer_id="b")
        m.add(a)
        m.add(b)
        m.remove("b")
        assert m.active is a

    def test_remove_unknown_raises_keyerror(
        self, make_target: Callable[..., ComputerTarget]
    ) -> None:
        m = ComputerTargetPool()
        m.add(make_target(computer_id="a"))
        with pytest.raises(KeyError):
            m.remove("missing")


class TestReset:
    def test_reset_clears_all(self) -> None:
        m = ComputerTargetPool()
        m.add(_make_remote(computer_id="a"))
        m.add(_make_remote(address="2.2.2.2:23000", computer_id="b"))
        m.reset()
        assert m.list() == []
        assert m.active is None
        assert len(m) == 0
