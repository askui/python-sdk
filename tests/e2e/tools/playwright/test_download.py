import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Download
from pytest_mock import MockerFixture

from askui.tools.playwright.agent_os import DownloadError, PlaywrightAgentOs

# A page with a link that downloads a small text file via a data URL. The
# ``download`` attribute makes the browser treat the navigation as a download
# and provides the suggested filename.
_DOWNLOAD_PAGE = (
    '<a id="dl" download="sample.txt" href="data:text/plain,Hello%20AskUI">download</a>'
)

# A file large enough that Playwright streams the artifact in multiple chunks,
# so a truncated save (the bug guarded against here) would cut it at a MiB
# boundary rather than producing a complete file.
_LARGE_FILE_SIZE = 20 * 1024 * 1024
_LARGE_FILE_NAME = "large.bin"
_LARGE_PAYLOAD = bytes((i * 2654435761) & 0xFF for i in range(_LARGE_FILE_SIZE))
_LARGE_PAYLOAD_SHA256 = hashlib.sha256(_LARGE_PAYLOAD).hexdigest()

_LARGE_DOWNLOAD_PAGE = (
    b"<html><body>"
    b'<a id="dl" href="/' + _LARGE_FILE_NAME.encode() + b'" download>Download</a>'
    b"</body></html>"
)


def _trigger_download(agent_os: PlaywrightAgentOs) -> None:
    page = agent_os._page
    assert page is not None
    page.set_content(_DOWNLOAD_PAGE)
    page.click("#dl")
    # Give the download event time to fire and the file to be written.
    page.wait_for_timeout(2000)


class _LargeDownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_LARGE_DOWNLOAD_PAGE)))
            self.end_headers()
            self.wfile.write(_LARGE_DOWNLOAD_PAGE)
            return

        if self.path == f"/{_LARGE_FILE_NAME}":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{_LARGE_FILE_NAME}"',
            )
            self.send_header("Content-Length", str(len(_LARGE_PAYLOAD)))
            self.end_headers()
            self.wfile.write(_LARGE_PAYLOAD)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr request logging."""


@pytest.fixture
def large_download_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LargeDownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.socket.getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join()


def _trigger_large_download(agent_os: PlaywrightAgentOs, base_url: str) -> None:
    agent_os.goto(base_url + "/")
    page = agent_os._page
    assert page is not None
    page.click("#dl")


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


@pytest.mark.timeout(60)
def test_large_download_is_complete_after_disconnect(
    tmp_path: Path, large_download_server: str
) -> None:
    agent_os = PlaywrightAgentOs(
        headless=True, install_browser=False, download_dir=tmp_path
    )
    agent_os.connect()
    _trigger_large_download(agent_os, large_download_server)
    # End the run immediately without pumping the event loop further; the
    # download must still be saved completely before the browser closes.
    agent_os.disconnect()

    saved = tmp_path / _LARGE_FILE_NAME
    assert saved.exists()
    content = saved.read_bytes()
    assert len(content) == _LARGE_FILE_SIZE
    assert hashlib.sha256(content).hexdigest() == _LARGE_PAYLOAD_SHA256


@pytest.mark.timeout(60)
def test_wait_until_downloads_complete_returns_saved_paths(
    tmp_path: Path, large_download_server: str
) -> None:
    agent_os = PlaywrightAgentOs(
        headless=True, install_browser=False, download_dir=tmp_path
    )
    agent_os.connect()
    try:
        _trigger_large_download(agent_os, large_download_server)
        saved_paths = agent_os.wait_until_downloads_complete()

        assert saved_paths == [tmp_path / _LARGE_FILE_NAME]
        assert (tmp_path / _LARGE_FILE_NAME).read_bytes() == _LARGE_PAYLOAD
    finally:
        agent_os.disconnect()


@pytest.mark.timeout(60)
def test_failed_save_raises_and_leaves_no_partial_file(
    tmp_path: Path, large_download_server: str, mocker: MockerFixture
) -> None:
    def _fail(_download: Download, path: object) -> None:
        # Emulate Playwright aborting the copy after writing a partial file.
        Path(str(path)).write_bytes(b"partial")
        error_msg = "Target page, context or browser has been closed"
        raise RuntimeError(error_msg)

    mocker.patch.object(Download, "save_as", _fail)

    agent_os = PlaywrightAgentOs(
        headless=True, install_browser=False, download_dir=tmp_path
    )
    agent_os.connect()
    _trigger_large_download(agent_os, large_download_server)

    with pytest.raises(DownloadError):
        agent_os.disconnect()

    # The truncated partial must be removed, not left masquerading as a
    # complete download.
    assert list(tmp_path.glob("*")) == []
