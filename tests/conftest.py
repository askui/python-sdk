import pathlib
from typing import cast

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from askui.tools.agent_os import ComputerAgentOS, Display, DisplaySize


@pytest.fixture
def path_fixtures() -> pathlib.Path:
    """Fixture providing the path to the fixtures directory."""
    return pathlib.Path().absolute() / "tests" / "fixtures"


@pytest.fixture
def path_fixtures_images(path_fixtures: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the images directory."""
    return path_fixtures / "images"


@pytest.fixture
def path_fixtures_screenshots(path_fixtures: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the screenshots directory."""
    return path_fixtures / "screenshots"


@pytest.fixture
def path_fixtures_pdf(path_fixtures: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the pdf directory."""
    return path_fixtures / "pdf"


@pytest.fixture
def path_fixtures_dummy_pdf(path_fixtures_pdf: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the dummy pdf."""
    return path_fixtures_pdf / "dummy.pdf"


@pytest.fixture
def path_fixtures_excel(path_fixtures: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the excel directory."""
    return path_fixtures / "excel"


@pytest.fixture
def path_fixtures_dummy_excel(path_fixtures_excel: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the dummy excel."""
    return path_fixtures_excel / "dummy.xlsx"


@pytest.fixture
def path_fixtures_docs(path_fixtures: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the docs directory."""
    return path_fixtures / "docs"


@pytest.fixture
def path_fixtures_dummy_doc(path_fixtures_docs: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the dummy doc."""
    return path_fixtures_docs / "dummy.docx"


@pytest.fixture
def github_login_screenshot(path_fixtures_screenshots: pathlib.Path) -> Image.Image:
    """Fixture providing the GitHub login screenshot."""
    screenshot_path = path_fixtures_screenshots / "macos__chrome__github_com__login.png"
    return Image.open(screenshot_path)


@pytest.fixture
def white_page_screenshot(path_fixtures_screenshots: pathlib.Path) -> Image.Image:
    """Fixture providing the white page screenshot."""
    screenshot_path = path_fixtures_screenshots / "white_page.png"
    return Image.open(screenshot_path)


@pytest.fixture
def path_fixtures_github_com__icon(path_fixtures_images: pathlib.Path) -> pathlib.Path:
    """Fixture providing the path to the github com icon image."""
    return path_fixtures_images / "github_com__icon.png"


@pytest.fixture
def agent_os_mock(mocker: MockerFixture) -> ComputerAgentOS:
    """Fixture providing a mock agent os."""
    mock = mocker.MagicMock(spec=ComputerAgentOS)
    mock.retrieve_active_display.return_value = Display(
        id=1,
        name="Display 1",
        size=DisplaySize(width=100, height=100),
    )
    mock.screenshot.return_value = Image.new("RGB", (100, 100), color="white")
    return cast("ComputerAgentOS", mock)


@pytest.fixture
def agent_os_mock_patch(
    mocker: MockerFixture, agent_os_mock: ComputerAgentOS
) -> ComputerAgentOS:
    """Patches `MultiComputerTargetAgentOS` so `ComputerAgent` uses `agent_os_mock`."""
    mocker.patch(
        "askui.computer_agent.MultiComputerTargetAgentOS",
        return_value=agent_os_mock,
    )
    return agent_os_mock


@pytest.fixture(autouse=True)
def disable_telemetry() -> None:
    from askui.container import telemetry

    telemetry.set_processors([])


@pytest.fixture
def askui_logo_bmp(path_fixtures: pathlib.Path) -> Image.Image:
    """Fixture providing askui logo as BMP."""
    screenshot_path = path_fixtures / "images" / "logo.bmp"
    return Image.open(screenshot_path)
