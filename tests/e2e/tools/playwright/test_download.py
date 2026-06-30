from pathlib import Path

import pytest

from askui.tools.playwright.agent_os import PlaywrightAgentOs

# A page with a link that downloads a small text file via a data URL. The
# ``download`` attribute makes the browser treat the navigation as a download
# and provides the suggested filename.
_DOWNLOAD_PAGE = (
    '<a id="dl" download="sample.txt" href="data:text/plain,Hello%20AskUI">download</a>'
)


def _trigger_download(agent_os: PlaywrightAgentOs) -> None:
    page = agent_os._page
    assert page is not None
    page.set_content(_DOWNLOAD_PAGE)
    page.click("#dl")
    # Give the download event time to fire and the file to be written.
    page.wait_for_timeout(2000)


@pytest.mark.timeout(60)
def test_download_is_copied_into_download_dir(tmp_path: Path) -> None:
    agent_os = PlaywrightAgentOs(
        headless=True, install_browser=False, download_dir=tmp_path
    )
    agent_os.connect()
    try:
        _trigger_download(agent_os)
    finally:
        agent_os.disconnect()

    saved = tmp_path / "sample.txt"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "Hello AskUI"
    assert agent_os.downloaded_files == [saved]


@pytest.mark.timeout(60)
def test_colliding_downloads_are_auto_renamed(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("pre-existing", encoding="utf-8")

    agent_os = PlaywrightAgentOs(
        headless=True, install_browser=False, download_dir=tmp_path
    )
    agent_os.connect()
    try:
        _trigger_download(agent_os)
    finally:
        agent_os.disconnect()

    renamed = tmp_path / "sample (1).txt"
    assert renamed.exists()
    assert renamed.read_text(encoding="utf-8") == "Hello AskUI"
    # The pre-existing file is left untouched.
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "pre-existing"


@pytest.mark.timeout(60)
def test_no_download_dir_leaves_files_in_temp(tmp_path: Path) -> None:
    agent_os = PlaywrightAgentOs(headless=True, install_browser=False)
    agent_os.connect()
    try:
        _trigger_download(agent_os)
    finally:
        agent_os.disconnect()

    assert agent_os.downloaded_files == []
    assert list(tmp_path.iterdir()) == []
