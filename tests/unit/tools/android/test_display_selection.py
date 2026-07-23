"""Unit tests for AndroidDisplay flag emission and PpadbAgentOs display pinning."""

import pytest

from askui.android_agent import AndroidAgent
from askui.tools.android.agent_os import (
    AndroidDisplay,
    SingleAndroidDisplay,
    UnknownAndroidDisplay,
)
from askui.tools.android.android_agent_os_error import AndroidAgentOsError
from askui.tools.android.ppadb_agent_os import PpadbAgentOs
from askui.tools.android.tools import (
    AndroidGetConnectedDisplaysInfosTool,
    AndroidSelectDeviceBySerialNumberTool,
    AndroidSelectDisplayByUniqueIDTool,
)


# --- AndroidDisplay flag emission ------------------------------------------


def test_both_ids_emit_flags() -> None:
    d = AndroidDisplay(1000000000000000002, "secondary", 2)
    assert d.get_display_id_flag() == "-d 2"
    assert d.get_display_unique_id_flag() == "-d 1000000000000000002"


def test_none_display_id_omits_input_flag_but_keeps_screencap() -> None:
    # primary case: input runs with no -d (default display), screencap keeps -d.
    d = AndroidDisplay(1000000000000000001, "primary", None)
    assert d.get_display_id_flag() == ""
    assert d.get_display_unique_id_flag() == "-d 1000000000000000001"


def test_none_unique_id_omits_screencap_flag() -> None:
    d = AndroidDisplay(None, "primary", 0)
    assert d.get_display_id_flag() == "-d 0"
    assert d.get_display_unique_id_flag() == ""


def test_single_display_omits_both_flags() -> None:
    d = SingleAndroidDisplay("primary")
    assert d.get_display_id_flag() == ""
    assert d.get_display_unique_id_flag() == ""


def test_unknown_display_omits_both_flags() -> None:
    d = UnknownAndroidDisplay()
    assert d.get_display_id_flag() == ""
    assert d.get_display_unique_id_flag() == ""


def test_str_hides_none_ids_from_model() -> None:
    # The model selects displays by unique id; a None id must not be presented
    # as a selectable value.
    with_id = AndroidDisplay(1000000000000000002, "secondary", 2)
    assert "unique_display_id=1000000000000000002" in str(with_id)

    without_id = AndroidDisplay(None, "primary", None)
    assert "None" not in str(without_id)
    assert "default display" in str(without_id)


# --- PpadbAgentOs display override -----------------------------------------


def test_single_display_becomes_override_list() -> None:
    d = AndroidDisplay(1000000000000000001, "primary", None)
    os = PpadbAgentOs(display=d)
    assert os.get_connected_displays() == [d]


def test_display_list_is_authoritative() -> None:
    d1 = AndroidDisplay(1000000000000000001, "primary", 0)
    d2 = AndroidDisplay(1000000000000000002, "secondary", 2)
    os = PpadbAgentOs(display=[d1, d2])
    assert os.get_connected_displays() == [d1, d2]


def test_empty_display_list_rejected() -> None:
    with pytest.raises(AndroidAgentOsError):
        PpadbAgentOs(display=[])


def test_set_display_by_unique_id_resolves_against_override() -> None:
    d1 = AndroidDisplay(1000000000000000001, "primary", 0)
    d2 = AndroidDisplay(1000000000000000002, "secondary", 2)
    os = PpadbAgentOs(display=[d1, d2])

    os.set_display_by_unique_id(1000000000000000002)
    assert os._selected_display is d2


def test_set_display_by_unique_id_unknown_raises() -> None:
    d1 = AndroidDisplay(1000000000000000001, "primary", 0)
    os = PpadbAgentOs(display=[d1])
    with pytest.raises(AndroidAgentOsError):
        os.set_display_by_unique_id(999999)


def test_no_override_when_selector_is_int_or_str() -> None:
    assert PpadbAgentOs(display=0)._display_override is None
    assert PpadbAgentOs(display="primary")._display_override is None
    assert PpadbAgentOs()._display_override is None


# --- AndroidAgent display-tool policy --------------------------------------


def test_default_tools_include_display_selectors() -> None:
    tools = AndroidAgent.get_default_tools()
    types = {type(t) for t in tools}
    assert AndroidSelectDisplayByUniqueIDTool in types
    assert AndroidSelectDeviceBySerialNumberTool in types


def test_disallowing_switching_drops_mutating_display_tools() -> None:
    tools = AndroidAgent.get_default_tools()
    filtered = AndroidAgent._apply_display_tool_policy(
        tools, display_allow_switching=False
    )
    types = {type(t) for t in filtered}
    assert AndroidSelectDisplayByUniqueIDTool not in types
    assert AndroidSelectDeviceBySerialNumberTool not in types
    # read-only info tool stays
    assert AndroidGetConnectedDisplaysInfosTool in types


def test_allowing_switching_keeps_all_tools() -> None:
    tools = AndroidAgent.get_default_tools()
    filtered = AndroidAgent._apply_display_tool_policy(
        tools, display_allow_switching=True
    )
    assert filtered == tools
