import base64
import io
from typing import Literal

import pytest
from PIL import Image

from askui.reporting import CompositeReporter
from askui.tools.agent_os import Coordinate
from askui.tools.askui import LocalComputerTarget
from askui.tools.askui.askui_controller import (
    MultiComputerTargetAgentOS,
    RenderObjectStyle,
)
from askui.tools.askui.askui_controller_settings import AskUiControllerSettings


@pytest.fixture
def agent_os_target_computer() -> LocalComputerTarget:
    return LocalComputerTarget(
        settings=AskUiControllerSettings(controller_args="--showOverlay true")
    )


@pytest.fixture
def controller_client(
    agent_os_target_computer: LocalComputerTarget,
) -> MultiComputerTargetAgentOS:
    return MultiComputerTargetAgentOS(
        reporter=CompositeReporter(),
        display=1,
        agent_os_target_computers=[agent_os_target_computer],
    )


def test_actions(controller_client: MultiComputerTargetAgentOS) -> None:
    with controller_client:
        controller_client.screenshot()
        controller_client.mouse_move(0, 0)
        controller_client.click()


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_click_all_buttons(
    controller_client: MultiComputerTargetAgentOS,
    button: Literal["left", "middle", "right"],
) -> None:
    """Test clicking each mouse button"""
    with controller_client:
        controller_client.click(button=button)


def test_mouse_multiple_clicks(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test click count parameter"""
    with controller_client:
        controller_client.click(count=3)


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_mouse_press_hold_release(
    controller_client: MultiComputerTargetAgentOS,
    button: Literal["left", "middle", "right"],
) -> None:
    """Test mouse_down() and mouse_up() operations"""
    with controller_client:
        controller_client.mouse_down(button=button)
        controller_client.mouse_up(button=button)


@pytest.mark.parametrize("x,y", [(0, 0), (100, 100), (500, 300)])
def test_mouse_move_coordinates(
    controller_client: MultiComputerTargetAgentOS, x: int, y: int
) -> None:
    """Test mouse movement to various coordinates"""
    with controller_client:
        controller_client.mouse_move(x, y)


def test_mouse_scroll_directions(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test horizontal and vertical scrolling"""
    with controller_client:
        controller_client.mouse_scroll(0, 5)  # Vertical scroll
        controller_client.mouse_scroll(5, 0)  # Horizontal scroll
        controller_client.mouse_scroll(3, -2)  # Combined scroll


def test_type_text_basic(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test typing simple text"""
    with controller_client:
        controller_client.type("Hello World")


def test_type_text_with_speed(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test typing with custom speed"""
    with controller_client:
        controller_client.type("Fast typing", typing_speed=100)
        controller_client.type("Slow typing", typing_speed=10)


def test_keyboard_tap_with_modifiers(
    controller_client: MultiComputerTargetAgentOS,
) -> None:
    """Test key combination like Ctrl+C"""
    with controller_client:
        controller_client.keyboard_tap("c", modifier_keys=["command"])
        controller_client.keyboard_tap("v", modifier_keys=["command"])


def test_keyboard_tap_multiple(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test multiple key taps"""
    with controller_client:
        controller_client.keyboard_tap("escape", count=3)


def test_keyboard_press_hold_release(
    controller_client: MultiComputerTargetAgentOS,
) -> None:
    """Test keyboard_pressed() and keyboard_release()"""
    with controller_client:
        controller_client.keyboard_pressed("escape")
        controller_client.keyboard_release("escape")


def test_screenshot_basic(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test taking screenshots with different report settings"""
    with controller_client:
        image_with_report = controller_client.screenshot()
        assert isinstance(image_with_report, Image.Image)


def test_get_display_information(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test retrieving display information"""
    with controller_client:
        display_info = controller_client.list_displays()
        assert display_info is not None


def test_get_process_list(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test retrieving running processes"""
    with controller_client:
        processes = controller_client.get_process_list()
        assert processes is not None

        processes_extended = controller_client.get_process_list(get_extended_info=True)
        assert processes_extended is not None


def test_get_automation_target_list(
    controller_client: MultiComputerTargetAgentOS,
) -> None:
    """Test retrieving automation targets"""
    with controller_client:
        targets = controller_client.get_automation_target_list()
        assert targets is not None


def test_set_display(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test changing active display"""
    with controller_client:
        controller_client.set_display(1)


def test_set_mouse_delay(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test configuring mouse action delay"""
    with controller_client:
        controller_client.set_mouse_delay(100)


def test_set_keyboard_delay(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test configuring keyboard action delay"""
    with controller_client:
        controller_client.set_keyboard_delay(50)


def test_run_command(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test executing shell commands"""
    with controller_client:
        controller_client.run_command("echo test", 0)


def test_get_action_count(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test getting count of batched actions"""
    with controller_client:
        count = controller_client.get_action_count()
        assert count is not None


def test_operations_before_connect() -> None:
    """Test calling methods before connect() raises appropriate errors"""
    client = MultiComputerTargetAgentOS(reporter=CompositeReporter(), display=1)

    with pytest.raises(
        AssertionError, match="Stub is not initialized. Call `connect()` first."
    ):
        client.screenshot()


def test_invalid_coordinates(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test mouse operations with potentially problematic coordinates"""
    with controller_client:
        controller_client.mouse_move(-1, -1)
        controller_client.mouse_move(9999, 9999)


def test_set_mouse_position(controller_client: MultiComputerTargetAgentOS) -> None:
    with controller_client:
        controller_client.set_mouse_position(100, 100)


def test_get_mouse_position(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test getting current mouse coordinates"""
    with controller_client:
        position = controller_client.get_mouse_position()
        assert position is not None
        assert hasattr(position, "x")
        assert hasattr(position, "y")


def test_render_quad(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test adding a quad render object to the display"""
    with controller_client:
        style = RenderObjectStyle(
            width=0.9,
            height=100,
            top="200px",
            left="10%",
            color="#ff0000",
            opacity=1,
        )

        response = controller_client.render_quad(style)

        assert response is not None


def test_render_line(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test rendering a line object to the display"""
    with controller_client:
        style = RenderObjectStyle(
            color="#00ff00",
            line_width=4,
            opacity=0.8,
        )
        points = [Coordinate(x=100, y=100), Coordinate(x=500, y=500)]

        response = controller_client.render_line(style, points)
        assert response is not None


def test_render_image(
    controller_client: MultiComputerTargetAgentOS,
    askui_logo_bmp: Image.Image,
) -> None:
    """Test rendering an image object to the display"""
    with controller_client:
        style = RenderObjectStyle(
            width=200,
            height=200,
            top=200,
            left=200,
            opacity=0.9,
        )

        img_buffer = io.BytesIO()
        askui_logo_bmp.save(img_buffer, format="BMP")
        img_bytes = img_buffer.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        response = controller_client.render_image(style, base64_image)
        assert response is not None


def test_render_text(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test rendering a text object to the display"""
    with controller_client:
        style = RenderObjectStyle(
            width=300,
            height=50,
            top=100,
            left=100,
            color="#0000ff",
            font_size=33,
            opacity=0.9,
        )

        response = controller_client.render_text(style, "Hello World!")
        assert response is not None


def test_update_render_object(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test updating an existing render object"""
    with controller_client:
        style = RenderObjectStyle(
            width=0.9,
            height=100,
            top="200px",
            left="10%",
            color="#ff0000",
            opacity=1,
        )

        object_id = controller_client.render_quad(style)
        assert object_id is not None

        update_style = RenderObjectStyle(
            width=0.5,
            height=100,
            top="200px",
            left="10%",
            color="#ff0000",
            opacity=1,
        )

        controller_client.update_render_object(object_id, update_style)


def test_update_text_object(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test updating an existing render object"""
    with controller_client:
        style = RenderObjectStyle(
            width=300,
            height=50,
            top=100,
            left=100,
            color="#0000ff",
            font_size=33,
            opacity=0.9,
        )

        object_id = controller_client.render_text(style, "Hello World!")
        assert object_id is not None

        update_style = RenderObjectStyle(
            width=0.5,
            height=100,
            top="200px",
            left="10%",
            color="#ff0000",
            opacity=1,
        )

        controller_client.update_render_object(object_id, update_style)


def test_delete_render_object(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test deleting an existing render object"""
    with controller_client:
        style = RenderObjectStyle(
            width=1.0,
            height=100,
            color="#ff0000",
            top=100,
            left=0,
        )
        quad_id = controller_client.render_quad(style)
        assert quad_id is not None

        controller_client.delete_render_object(quad_id)


def test_clear_render_objects(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test clearing all render objects"""
    with controller_client:
        style1 = RenderObjectStyle(
            width=100,
            height=50,
            color="#ff0000",
            top=100,
            left=100,
        )
        style2 = RenderObjectStyle(
            width=200,
            height=100,
            color="#00ff00",
            top=200,
            left=200,
        )

        controller_client.render_quad(style1)
        controller_client.render_quad(style2)

        controller_client.clear_render_objects()


def test_get_system_info(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test getting system information"""
    with controller_client:
        system_info = controller_client.get_system_info()
        assert system_info is not None
        assert system_info.platform is not None
        assert system_info.label is not None
        assert system_info.version is not None
        assert system_info.architecture is not None


def test_get_active_process(controller_client: MultiComputerTargetAgentOS) -> None:
    with controller_client:
        active_process = controller_client.get_active_process()

        assert active_process is not None
        assert active_process.process is not None
        assert active_process.process.name is not None
        assert active_process.process.id is not None


def test_set_active_process(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test setting the active process"""
    with controller_client:
        controller_client.set_active_process(1062)
        active_process = controller_client.get_active_process()
        assert active_process is not None
        assert active_process.process is not None


def test_get_active_window(controller_client: MultiComputerTargetAgentOS) -> None:
    """Test getting the active window"""
    with controller_client:
        active_window = controller_client.get_active_window()
        assert active_window is not None
        assert active_window.window is not None
        assert active_window.window.id is not None
        assert active_window.window.name is not None
        assert active_window.window.processId is not None
        assert active_window.window.processName is not None
