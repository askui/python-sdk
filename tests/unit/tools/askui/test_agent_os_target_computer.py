from typing import Callable

import pytest

from askui.tools.askui.agent_os_target_computer import (
    ComputerTarget,
    LocalComputerTarget,
    RemoteComputerTarget,
)


class TestReplacePort:
    def test_replaces_port_on_bare_authority(self) -> None:
        assert (
            LocalComputerTarget.replace_port("example.com:1234", 23000)
            == "example.com:23000"
        )

    def test_replaces_port_on_url_with_scheme(self) -> None:
        assert (
            LocalComputerTarget.replace_port("http://example.com:1234", 23000)
            == "example.com:23000"
        )

    def test_falls_back_to_localhost_when_host_missing(self) -> None:
        # A bare ":1234" has no hostname, so the helper falls back to "localhost".
        assert LocalComputerTarget.replace_port(":1234", 23000) == "localhost:23000"


class TestAgentOsTargetComputer:
    def test_session_guid_unique_per_instance(self) -> None:
        a = RemoteComputerTarget(address="1.2.3.4:23000", description="a")
        b = RemoteComputerTarget(address="5.6.7.8:23000", description="b")
        assert a.session_guid != b.session_guid

    def test_computer_id_defaults_to_session_guid(self) -> None:
        s = RemoteComputerTarget(address="1.2.3.4:23000", description="a")
        assert s.computer_id == s.session_guid

    def test_explicit_computer_id_is_preserved(self) -> None:
        s = RemoteComputerTarget(
            address="1.2.3.4:23000", description="a", computer_id="laptop"
        )
        assert s.computer_id == "laptop"
        assert s.session_guid != "laptop"

    def test_display_defaults_to_one_and_is_settable(self) -> None:
        s = RemoteComputerTarget(address="1.2.3.4:23000", description="a")
        assert s.display == 1
        s.display = 3
        assert s.display == 3

    def test_explicit_display_is_preserved(self) -> None:
        s = RemoteComputerTarget(address="1.2.3.4:23000", description="a", display=2)
        assert s.display == 2

    def test_repr_contains_identity_fields(self) -> None:
        s = RemoteComputerTarget(
            address="1.2.3.4:23000",
            description="my rig",
            display=2,
            computer_id="rig",
        )
        r = repr(s)
        assert "RemoteComputerTarget" in r
        assert "computer_id='rig'" in r
        assert "description='my rig'" in r
        assert "display=2" in r

    def test_base_class_is_not_local(self) -> None:
        s = RemoteComputerTarget(address="1.2.3.4:23000", description="a")
        assert s.is_local is False

    def test_start_and_stop_are_no_ops_on_remote(self) -> None:
        s = RemoteComputerTarget(address="1.2.3.4:23000", description="a")
        s.start()
        s.stop()


class TestLocalAgentOsTargetComputer:
    def test_is_local(self) -> None:
        s = LocalComputerTarget(discover_service=False)
        assert s.is_local is True

    def test_default_description(self) -> None:
        s = LocalComputerTarget(discover_service=False)
        assert s.description == "Local computer target"

    def test_default_address(self) -> None:
        s = LocalComputerTarget(discover_service=False)
        assert s.address == "localhost:23000"

    def test_is_service_default_false(self) -> None:
        s = LocalComputerTarget(discover_service=False)
        assert s.is_service is False

    def test_explicit_computer_id(self) -> None:
        s = LocalComputerTarget(discover_service=False, computer_id="my-laptop")
        assert s.computer_id == "my-laptop"

    def test_parse_port_rejects_bad_address(self) -> None:
        s = LocalComputerTarget(discover_service=False, address="no-port-here")
        with pytest.raises(ValueError, match="Could not parse port"):
            s._parse_port()  # noqa: SLF001 - intentional unit test against helper

    def test_parse_port_extracts_port(self) -> None:
        s = LocalComputerTarget(discover_service=False, address="localhost:24567")
        assert s._parse_port() == 24567  # noqa: SLF001


class TestSubclassesPassThroughDisplayAndId:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: LocalComputerTarget(
                discover_service=False, display=4, computer_id="local"
            ),
            lambda: RemoteComputerTarget(
                address="1.2.3.4:23000",
                description="r",
                display=4,
                computer_id="remote",
            ),
        ],
    )
    def test_display_and_computer_id_round_trip(
        self, factory: Callable[[], ComputerTarget]
    ) -> None:
        s: ComputerTarget = factory()
        assert s.display == 4
        assert s.computer_id in {"local", "remote"}
