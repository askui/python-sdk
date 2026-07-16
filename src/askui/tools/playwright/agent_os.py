from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path
from typing import Literal

from PIL import Image
from playwright.sync_api import (
    Browser,
    BrowserContext,
    BrowserType,
    Download,
    Page,
    Playwright,
    ViewportSize,
    sync_playwright,
)
from typing_extensions import override

from askui.reporting import NULL_REPORTER, Reporter
from askui.utils.annotated_image import AnnotatedImage

from ..agent_os import (
    ComputerAgentOS,
    Display,
    DisplaySize,
    InputEvent,
    ModifierKey,
    PcKey,
)


class DownloadError(RuntimeError):
    """Raised when one or more browser downloads could not be saved completely.

    Surfaced instead of leaving silently truncated files on disk (e.g. when a
    download is still being copied while the browser is torn down).
    """


# Time to pump the Playwright event loop so a download that started right
# before teardown surfaces its ``download`` event before the queue is drained.
_DOWNLOAD_EVENT_GRACE_S = 0.1


def _to_unique_path(path: Path) -> Path:
    """Return ``path`` or, if it already exists, a counter-suffixed variant.

    For example, if ``report.pdf`` exists, returns ``report (1).pdf``; if that
    exists too, ``report (2).pdf``, and so on. This keeps existing files from
    being overwritten.

    Args:
        path (Path): The desired target path.

    Returns:
        Path: A path that does not currently exist on disk.
    """
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


class PlaywrightAgentOs(ComputerAgentOS):
    """Playwright-based implementation of `ComputerAgentOS`.

    This implementation uses Playwright's Python SDK to control browser automation
    and simulate user interactions. It provides mouse control, keyboard input,
    screen capture, and multi-tab management functionality through a browser context.

    Args:
        reporter (Reporter, optional): Reporter used for reporting. Defaults to
            `NULL_REPORTER`.
        browser_type (Literal["chromium", "firefox", "webkit"], optional): The browser
            type to use. Defaults to `"chromium"`.
        headless (bool, optional): Whether to run the browser in headless mode.
            Defaults to `False`.
        viewport_size (ViewportSize | None, optional): The viewport size. When
            ``None``, the browser inherits the system's native DPI and window
            size (``no_viewport=True``). Defaults to `None`.
        slow_mo (int, optional): Slows down Playwright operations by the specified
            amount of milliseconds. Defaults to `0`.
        install_browser (bool, optional): Whether to install browser on connection.
            Defaults to `True`.
        install_dependencies (bool, optional): Whether to install system dependencies
            (requires root permissions). Defaults to `False`.
        download_dir (str | Path | None, optional): Directory into which files
            downloaded by the browser are automatically copied once they finish.
            When ``None``, downloads are left in Playwright's temporary location
            (and deleted when the browser closes). The directory is created if it
            does not exist. Defaults to `None`.
        auto_follow_new_tab (bool, optional): When `True`, any new tab opened by the
            browser (e.g. via ``target="_blank"`` links or ``window.open()``)
            automatically becomes the active tab. When `False`, new tabs are tracked
            but the active tab does not change; use `switch_tab()` to move to them
            manually. Defaults to `True`.
    """

    _REPORTER_ROLE_NAME: str = "PlaywrightAgentOS"

    def __init__(
        self,
        reporter: Reporter = NULL_REPORTER,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
        viewport_size: ViewportSize | None = None,
        slow_mo: int = 0,
        install_browser: bool = True,
        install_dependencies: bool = False,
        download_dir: str | Path | None = None,
        auto_follow_new_tab: bool = True,
    ) -> None:
        self._browser_type = browser_type
        self._headless = headless
        self._viewport_size = viewport_size
        self._slow_mo = slow_mo
        self._install_browser = install_browser
        self._install_dependencies = install_dependencies
        self._download_dir = Path(download_dir) if download_dir is not None else None
        self._auto_follow_new_tab = auto_follow_new_tab

        # Playwright objects
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._pages: list[Page] = []
        self._reporter: Reporter = reporter

        # Set to True by _on_new_page when a new tab is followed; cleared by
        # _sync_pages() after calling bring_to_front() on the main thread.
        self._needs_bring_to_front: bool = False

        # Event listening state
        self._listening = False
        self._event_queue: list[InputEvent] = []

        # Download tracking state. `_pending_downloads` holds downloads whose
        # copy has not run yet; they are drained on the main thread (never
        # inside the event callback) so the copy runs deterministically and is
        # awaited before the browser is torn down. `_download_errors` collects
        # failures so they can be surfaced instead of leaving truncated files.
        self._download_lock = threading.Lock()
        self._pending_downloads: list[Download] = []
        self._downloaded_files: list[Path] = []
        self._download_errors: list[str] = []

    def _install_playwright_browser(self) -> None:
        """Install Playwright browser if requested."""
        if not self._install_browser:
            return

        try:
            # Install the specific browser type
            subprocess.run(
                ["playwright", "install", self._browser_type],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install {self._browser_type} browser: {e}"
            raise RuntimeError(error_msg) from e
        except FileNotFoundError as e:
            error_msg = (
                "Playwright CLI not found. Install with `pip install playwright`"
            )
            raise RuntimeError(error_msg) from e

    def _install_system_dependencies(self) -> None:
        """Install system dependencies if requested (requires root permissions)."""
        if not self._install_dependencies:
            return

        try:
            # Install system dependencies
            subprocess.run(
                ["playwright", "install-deps"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install system dependencies: {e}"
            raise RuntimeError(error_msg) from e
        except FileNotFoundError as e:
            error_msg = (
                "Playwright CLI not found. Install with `pip install playwright`"
            )
            raise RuntimeError(error_msg) from e

    def _annotated_screenshot(
        self,
        point_list: list[tuple[int, int]],
    ) -> AnnotatedImage:
        """Capture a screenshot and wrap it in an `AnnotatedImage` with annotations."""
        screenshot = self.screenshot(report=False)
        return AnnotatedImage(lambda: screenshot, point_list)

    @override
    def connect(self) -> None:
        """Establishes a synchronous connection to the browser."""

        # Install browser and dependencies if requested
        if self._install_dependencies:
            self._install_system_dependencies()

        if self._install_browser:
            self._install_playwright_browser()

        self._playwright = sync_playwright().start()
        browser_launcher: BrowserType = getattr(self._playwright, self._browser_type)
        self._browser = browser_launcher.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
        )
        if self._viewport_size is not None:
            self._context = self._browser.new_context(
                viewport=self._viewport_size,
            )
        else:
            # Use no_viewport to inherit the system's native DPI and window
            # size.  Without this, Playwright defaults to 1280x720 with
            # deviceScaleFactor=1.  On high-DPI screens (e.g. macOS Retina)
            # Chromium compensates by zooming the page 2x and briefly
            # un-zooming every time a screenshot is captured, causing a
            # visible flicker.
            self._context = self._browser.new_context(
                no_viewport=True,
            )

        self._pages = []
        self._page = self._context.new_page()
        self._pages.append(self._page)
        self._page.on("download", self._on_download)
        self._context.on("page", self._on_new_page)
        # Navigate to a blank page to ensure we have a working page
        self._page.goto("data:text/html,<html><body><h1>Starting...</h1></body></html>")
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "Connected to playwright browser",
        )

    def _on_new_page(self, page: Page) -> None:
        """Track a newly opened browser tab.

        Registered as a ``page`` event handler on the `BrowserContext`. Called
        from Playwright's background asyncio thread whenever a new tab is opened
        (e.g. via ``target="_blank"`` links or ``window.open()``).

        Only thread-safe operations are performed here: appending to the tracked
        list and reassigning ``self._page``. Sync Playwright methods such as
        ``bring_to_front()`` must NOT be called here — they require the main
        greenlet context and deadlock when invoked from the background thread.
        Auto-follow (including ``bring_to_front()``) is completed on the main
        thread by `_sync_pages`, which is called at the start of every
        `screenshot`.

        Args:
            page (Page): The newly opened Playwright page.
        """
        page.on("download", self._on_download)
        self._pages.append(page)
        if self._auto_follow_new_tab:
            self._page = page
            # Signal _sync_pages() to call bring_to_front() on the main thread.
            # bring_to_front() is deliberately NOT called here: even though
            # Playwright's EventGreenlet supports sync calls, any exception it
            # raises would be silently deferred and re-raised at the next API
            # call, which could fail an unrelated operation such as screenshot().
            self._needs_bring_to_front = True
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"New tab opened: '{page.url}'",
        )

    def _sync_pages(self) -> None:
        """Sync tracked pages with the live browser context, main-thread safe.

        Called at the start of every ``screenshot`` (after ``_pump_event_loop``
        has flushed all pending page events). Does three things:

        1. Picks up any pages that ``_on_new_page`` could not track because of
           the race between the dispatcher firing the event and ``screenshot``
           being called — after the pump the event has already fired, so this
           acts as a safety net rather than the primary detection path.
        2. Prunes pages that have been closed since the last call.
        3. Calls ``bring_to_front()`` on the main thread when
           ``_needs_bring_to_front`` is set. The flag is raised by
           ``_on_new_page`` instead of calling ``bring_to_front()`` there
           directly, because a deferred exception from inside an EventGreenlet
           would be re-raised at the next API call and silently break
           unrelated operations.
        """
        if self._context is None:
            return

        known_ids = {id(p) for p in self._pages}
        for page in self._context.pages:
            if id(page) not in known_ids:
                page.on("download", self._on_download)
                self._pages.append(page)
                known_ids.add(id(page))
                if self._auto_follow_new_tab:
                    self._page = page
                    self._needs_bring_to_front = True

        # Prune closed pages
        self._pages = [p for p in self._pages if not p.is_closed()]
        if self._page is not None and self._page.is_closed():
            self._page = self._pages[-1] if self._pages else None

        if self._needs_bring_to_front and self._page is not None:
            self._needs_bring_to_front = False
            self._page.bring_to_front()

    def _on_download(self, download: Download) -> None:
        """Register a started download for deterministic saving.

        Registered as the page's ``download`` event handler. The download is
        only recorded here; the actual copy runs in `_flush_downloads` on the
        main thread. Calling the blocking `save_as` from inside this sync-API
        event callback is not guaranteed to run to completion, which truncated
        large files when the browser was closed mid-copy.

        Args:
            download (Download): The Playwright download to persist.
        """
        if self._download_dir is None:
            return
        with self._download_lock:
            self._pending_downloads.append(download)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"Download started: '{Path(download.suggested_filename).name}'",
        )

    def _pump_event_loop(self, seconds: float) -> None:
        """Pump the Playwright event loop for ``seconds``.

        Lets queued events (such as a just-started ``download``) be delivered
        to their handlers. Best effort: swallows errors as the page may already
        be closing.
        """
        if self._page is None:
            return
        try:
            self._page.wait_for_timeout(seconds * 1000)
        except Exception:  # noqa: BLE001 - best effort; page may be closing
            pass

    def _deliver_and_flush_downloads(self) -> None:
        """Deliver any just-started download events, then save all downloads.

        Used at points where no other Playwright call is pumping the event loop
        (teardown and the explicit wait), so a download triggered by the last
        action is not missed.
        """
        if self._download_dir is None:
            return
        self._pump_event_loop(_DOWNLOAD_EVENT_GRACE_S)
        self._flush_downloads()

    def _flush_downloads(self) -> None:
        """Copy all pending downloads into `download_dir`, blocking until done.

        Runs on the main (Playwright) thread. Each failure is collected in
        `_download_errors` rather than raised here, so a single failing download
        neither aborts the remaining ones nor the teardown of other resources.
        """
        if self._download_dir is None:
            return
        with self._download_lock:
            pending = self._pending_downloads
            self._pending_downloads = []
        for download in pending:
            self._save_download(download)

    def _save_download(self, download: Download) -> None:
        assert self._download_dir is not None
        # Use only the filename component to avoid path traversal from a
        # server-suggested name such as "../../etc/passwd".
        suggested_name = Path(download.suggested_filename).name
        target = _to_unique_path(self._download_dir / suggested_name)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # `save_as` blocks until the download has fully finished and been
            # copied to `target`, so running it here (on the main thread, before
            # teardown) guarantees a complete file.
            download.save_as(target)
        except Exception as e:  # noqa: BLE001 - collect, don't let one break the run
            # Do not leave a partially written file masquerading as a complete
            # download; remove it and surface the failure.
            target.unlink(missing_ok=True)
            error_msg = f"Failed to save download '{suggested_name}': {e}"
            self._download_errors.append(error_msg)
            self._reporter.add_message(self._REPORTER_ROLE_NAME, error_msg)
            return
        self._downloaded_files.append(target)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"Downloaded file saved to {target}",
        )

    def wait_until_downloads_complete(self) -> list[Path]:
        """Block until all downloads started so far are fully saved to disk.

        `save_as` blocks until each download has finished, so callers can use
        this to obtain complete files mid-run without having to keep the
        Playwright event loop alive themselves.

        Returns:
            list[Path]: Absolute paths of all downloads saved so far this
                session.

        Raises:
            DownloadError: If one or more downloads could not be saved
                completely.
        """
        self._deliver_and_flush_downloads()
        self._raise_download_errors()
        return list(self._downloaded_files)

    def _raise_download_errors(self) -> None:
        if not self._download_errors:
            return
        errors = self._download_errors
        self._download_errors = []
        error_msg = "Failed to complete downloads:\n" + "\n".join(errors)
        raise DownloadError(error_msg)

    @property
    def downloaded_files(self) -> list[Path]:
        """Files copied into `download_dir`, in the order they finished.

        Returns:
            list[Path]: Absolute paths of downloads saved so far this session.
                Empty when no `download_dir` was configured or nothing was
                downloaded yet.
        """
        return list(self._downloaded_files)

    @override
    def disconnect(self) -> None:
        """Terminates the connection to the browser.

        Any download triggered during the run is written to `download_dir`
        completely before the page/context/browser is closed. If a download
        cannot be saved, its partial file is removed and a `DownloadError` is
        raised after teardown instead of leaving a silently truncated file.

        Raises:
            DownloadError: If one or more downloads could not be saved
                completely.
        """
        if self._listening:
            self.stop_listening()

        # Drain in-flight downloads while the page/context/browser are still
        # alive so `save_as` can run to completion. Otherwise Playwright aborts
        # the copy and leaves a truncated file on disk.
        self._deliver_and_flush_downloads()

        # Clear page tracking; context.close() handles actual page teardown.
        self._pages.clear()
        self._page = None

        if self._context:
            self._context.close()
            self._context = None

        if self._browser:
            self._browser.close()
            self._browser = None

        if self._playwright:
            self._playwright.stop()
            self._playwright = None

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "Disconnected from playwright os",
        )

        self._raise_download_errors()

    @override
    def screenshot(self, report: bool = True, unscaled: bool = False) -> Image.Image:
        """Capture a screenshot of the current page.

        Args:
            report (bool, optional): Whether to include the screenshot in
                reporting. Defaults to `True`.
            unscaled (bool, optional): Accepted for interface compatibility. This
                agent OS always returns the native page resolution, so it has no
                effect. Defaults to `False`.

        Returns:
            Image.Image: A PIL Image object containing the screenshot.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        # Pump the event loop before syncing so any pending "page" events
        # (new tabs opened by the previous action) are processed first.
        # Without this, the new-page event fires during page.screenshot()
        # after the CDP command has already been sent to the old page.
        self._pump_event_loop(0)
        self._sync_pages()

        screenshot_bytes = self._page.screenshot(scale="css")
        screenshot = Image.open(io.BytesIO(screenshot_bytes))
        # Taking the screenshot pumps the event loop, so any download that
        # started since the last interaction has now surfaced its event.
        # Persist it so it is available to the caller during the run, not only
        # at teardown.
        self._flush_downloads()
        if report:
            self._reporter.add_message(
                self._REPORTER_ROLE_NAME, "screenshot()", screenshot
            )
        return screenshot

    @override
    def mouse_move(self, x: int, y: int, duration: int = 500) -> None:
        """Move the mouse cursor to specified coordinates on the page.

        Args:
            x (int): The horizontal coordinate (in pixels) to move to.
            y (int): The vertical coordinate (in pixels) to move to.
            duration (int, optional): Ignored — Playwright moves the mouse
                instantly. Kept for compatibility with the base class.
                Defaults to `500`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"mouse_move(x={x}, y={y})",
            self._annotated_screenshot([(x, y)]),
        )
        self._page.mouse.move(x, y)

    @override
    def type(self, text: str, typing_speed: int = 50) -> None:
        """
        Simulates typing text as if entered on a keyboard.

        Args:
            text (str): The text to be typed.
            typing_speed (int, optional): The speed of typing in characters per
                second. Defaults to `50`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"Typing text: '{text}'",
            self.screenshot(report=False),
        )
        # Convert typing speed from CPM to delay between characters
        delay = 1000 / typing_speed if typing_speed > 0 else 0
        self._page.keyboard.type(text, delay=delay)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"After typing text: '{text}'",
            self.screenshot(report=False),
        )

    @override
    def click(
        self, button: Literal["left", "middle", "right"] = "left", count: int = 1
    ) -> None:
        """
        Simulates clicking a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse
                button to click. Defaults to `"left"`.
            count (int, optional): Number of times to click. Defaults to `1`.
        """
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"click(button={button}, count={count})",
            self.screenshot(report=False),
        )
        for _ in range(count):
            self.mouse_down(button)
            self.mouse_up(button)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"After click(button={button}, count={count})",
            self.screenshot(report=False),
        )

    @override
    def mouse_down(self, button: Literal["left", "middle", "right"] = "left") -> None:
        """
        Simulates pressing and holding a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse
                button to press. Defaults to `"left"`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"mouse_down(button={button})",
        )
        self._page.mouse.down(button=button)

    @override
    def mouse_up(self, button: Literal["left", "middle", "right"] = "left") -> None:
        """
        Simulates releasing a mouse button.

        Args:
            button (Literal["left", "middle", "right"], optional): The mouse
                button to release. Defaults to `"left"`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._page.mouse.up(button=button)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"mouse_up(button={button})",
        )

    @override
    def mouse_scroll(self, dx: int, dy: int) -> None:
        """
        Simulates scrolling the mouse wheel.

        Args:
            dx (int): The horizontal scroll amount. Positive values scroll right,
                negative values scroll left.
            dy (int): The vertical scroll amount. Positive values scroll down,
                negative values scroll up.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"mouse_scroll(dx={dx}, dy={dy})",
            self.screenshot(report=False),
        )
        self._page.mouse.wheel(delta_x=dx, delta_y=dy)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"After mouse_scroll(dx={dx}, dy={dy})",
            self.screenshot(report=False),
        )

    @override
    def keyboard_pressed(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Simulates pressing and holding a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to press.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"keyboard_pressed(key={key}, modifier_keys={modifier_keys})",
            self.screenshot(report=False),
        )
        # Press modifier keys first
        if modifier_keys:
            for modifier in modifier_keys:
                self._page.keyboard.down(self._convert_key(modifier))

        # Press the main key
        self._page.keyboard.down(self._convert_key(key))

    @override
    def keyboard_release(
        self, key: PcKey | ModifierKey, modifier_keys: list[ModifierKey] | None = None
    ) -> None:
        """
        Simulates releasing a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to release.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                release along with the main key. Defaults to `None`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        # Release the main key first
        self._page.keyboard.up(self._convert_key(key))

        # Release modifier keys
        if modifier_keys:
            for modifier in modifier_keys:
                self._page.keyboard.up(self._convert_key(modifier))

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"keyboard_release(key={key}, modifier_keys={modifier_keys})",
            self.screenshot(report=False),
        )

    @override
    def keyboard_tap(
        self,
        key: PcKey | ModifierKey,
        modifier_keys: list[ModifierKey] | None = None,
        count: int = 1,
    ) -> None:
        """
        Simulates pressing and immediately releasing a keyboard key.

        Args:
            key (PcKey | ModifierKey): The key to tap.
            modifier_keys (list[ModifierKey] | None, optional): List of modifier keys to
                press along with the main key. Defaults to `None`.
            count (int, optional): The number of times to tap the key. Defaults to `1`.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            (f"keyboard_tap(key={key}, modifier_keys={modifier_keys}, count={count})"),
            self.screenshot(report=False),
        )
        for _ in range(count):
            # Press modifier keys first
            if modifier_keys:
                for modifier in modifier_keys:
                    self._page.keyboard.down(self._convert_key(modifier))

            # Press and release the main key
            self._page.keyboard.press(self._convert_key(key))

            # Release modifier keys
            if modifier_keys:
                for modifier in modifier_keys:
                    self._page.keyboard.up(self._convert_key(modifier))

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            (
                f"After keyboard_tap(key={key}, "
                f"modifier_keys={modifier_keys}, count={count})"
            ),
            self.screenshot(report=False),
        )

    @override
    def retrieve_active_display(self) -> Display:
        """
        Retrieve the currently active display/screen.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        viewport_size = self._page.viewport_size
        if viewport_size is None:
            error_msg = "No viewport size."
            raise RuntimeError(error_msg)

        return Display(
            id=1,
            name="Display",
            size=DisplaySize(
                width=viewport_size["width"],
                height=viewport_size["height"],
            ),
        )

    # --- Tab management ---

    def list_tabs(self) -> list[dict[str, int | str]]:
        """Return metadata for every currently open browser tab.

        Returns:
            list[dict[str, int | str]]: One entry per open tab, each containing:

                - ``index`` (int): Zero-based tab index used by `switch_tab` and
                  `close_tab`.
                - ``title`` (str): Page title (empty string if unavailable).
                - ``url`` (str): Current URL of the page.
        """
        tabs: list[dict[str, int | str]] = []
        for i, page in enumerate(self._pages):
            try:
                title: str = page.title()
            except Exception:  # noqa: BLE001
                title = ""
            tabs.append({"index": i, "title": title, "url": page.url})
        return tabs

    def switch_tab(self, index: int) -> None:
        """Switch the active browser tab to the tab at ``index``.

        Args:
            index (int): Zero-based index of the tab to activate (see
                `list_tabs` for available indices).

        Raises:
            RuntimeError: If no browser session is active.
            IndexError: If ``index`` is out of range.
        """
        if not self._pages:
            error_msg = "No open tabs. Call connect() first."
            raise RuntimeError(error_msg)
        if index < 0 or index >= len(self._pages):
            error_msg = (
                f"Tab index {index} is out of range. "
                f"Available indices: 0-{len(self._pages) - 1}."
            )
            raise IndexError(error_msg)
        self._page = self._pages[index]
        self._page.bring_to_front()
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"switch_tab(index={index}) -> '{self._page.url}'",
        )

    def close_tab(self, index: int) -> None:
        """Close the browser tab at ``index``.

        When the closed tab was the active one, the agent automatically
        switches to the nearest remaining tab (the tab at ``index - 1``, or
        the new last tab when ``index`` was the last one).

        Args:
            index (int): Zero-based index of the tab to close (see `list_tabs`
                for available indices).

        Raises:
            RuntimeError: If no browser session is active or if only one tab
                remains (closing it would leave the browser with no pages).
            IndexError: If ``index`` is out of range.
        """
        if not self._pages:
            error_msg = "No open tabs. Call connect() first."
            raise RuntimeError(error_msg)
        if len(self._pages) <= 1:
            error_msg = "Cannot close the last remaining tab."
            raise RuntimeError(error_msg)
        if index < 0 or index >= len(self._pages):
            error_msg = (
                f"Tab index {index} is out of range. "
                f"Available indices: 0-{len(self._pages) - 1}."
            )
            raise IndexError(error_msg)
        page_to_close = self._pages.pop(index)
        was_active = self._page is page_to_close
        page_to_close.close()
        if was_active:
            new_index = min(index, len(self._pages) - 1)
            self._page = self._pages[new_index]
            self._page.bring_to_front()
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"close_tab(index={index})",
        )

    def _convert_key(self, key: PcKey | ModifierKey) -> str:
        """
        Convert our key format to Playwright's key format.

        Args:
            key (PcKey | ModifierKey): The key to convert.

        Returns:
            str: The Playwright-compatible key string.
        """
        # Map our modifier keys to Playwright format
        modifier_map: dict[PcKey | ModifierKey, str] = {
            "command": "Meta",
            "alt": "Alt",
            "control": "Control",
            "shift": "Shift",
            "right_shift": "Shift",
        }

        if key in modifier_map:
            return modifier_map[key]

        # For regular keys, Playwright uses similar format
        # but some keys might need conversion
        key_map: dict[PcKey | ModifierKey, str] = {
            "backspace": "Backspace",
            "delete": "Delete",
            "enter": "Enter",
            "tab": "Tab",
            "escape": "Escape",
            "up": "ArrowUp",
            "down": "ArrowDown",
            "right": "ArrowRight",
            "left": "ArrowLeft",
            "home": "Home",
            "end": "End",
            "pageup": "PageUp",
            "pagedown": "PageDown",
            "numpad_lock": "NumLock",
            "numpad_0": "Numpad0",
            "numpad_1": "Numpad1",
            "numpad_2": "Numpad2",
            "numpad_3": "Numpad3",
            "numpad_4": "Numpad4",
            "numpad_5": "Numpad5",
            "numpad_6": "Numpad6",
            "numpad_7": "Numpad7",
            "numpad_8": "Numpad8",
            "numpad_9": "Numpad9",
            "numpad_+": "NumpadAdd",
            "numpad_-": "NumpadSubtract",
            "numpad_*": "NumpadMultiply",
            "numpad_/": "NumpadDivide",
            "numpad_.": "NumpadDecimal",
            "space": " ",
        }

        if key in key_map:
            return key_map[key]

        # Function keys
        if key.startswith("f") and key[1:].isdigit():
            return key.upper()

        # For most other keys, return as-is
        return key

    # --- Extra browser-oriented actions ---
    def goto(self, url: str) -> None:
        """
        Navigate to a specific URL.

        Args:
            url (str): The URL to navigate to.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"goto(url='{url}')",
        )
        self._page.goto(url)
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"After goto(url='{url}')",
            self.screenshot(report=False),
        )

    def back(self) -> None:
        """Navigate back to the previous page in the browser history."""
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "back()",
        )
        self._page.go_back()
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "After back()",
            self.screenshot(report=False),
        )

    def forward(self) -> None:
        """Navigate forward to the next page in the browser history."""
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "forward()",
        )
        self._page.go_forward()
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            "After forward()",
            self.screenshot(report=False),
        )

    def get_page_title(self) -> str:
        """
        Get the title of the current page.

        Returns:
            str: The page title.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        title = self._page.title()
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"get_page_title() -> '{title}'",
        )
        return title

    def get_page_url(self) -> str:
        """
        Get the URL of the current page.

        Returns:
            str: The current page URL.
        """
        if not self._page:
            error_msg = "No active page. Call connect() first."
            raise RuntimeError(error_msg)

        url = self._page.url
        self._reporter.add_message(
            self._REPORTER_ROLE_NAME,
            f"get_page_url() -> '{url}'",
        )
        return url

    @property
    def tags(self) -> list[str]:
        """Get the tags for this agent OS.

        Returns:
            list[str]: A list of tags that identify this agent OS type.
        """
        if not hasattr(self, "_tags"):
            self._tags = ["playwright"]
        return self._tags

    @tags.setter
    def tags(self, tags: list[str]) -> None:
        """Set the tags for this agent OS.

        Args:
            tags (list[str]): A list of tags that identify this agent OS type.
        """
        self._tags = tags
