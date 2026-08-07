from __future__ import annotations

import base64
import io
import json
import logging
import platform
import random
import shutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from importlib.metadata import distributions
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from jinja2 import Template
from typing_extensions import TypedDict, override

from askui.utils.annotated_image import AnnotatedImage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

    from askui.callbacks.conversation_statistics_callback import (
        ConversationUsageSummary,
        UsageSummary,
    )


def normalize_to_pil_images(
    image: Image.Image | list[Image.Image] | AnnotatedImage | None,
) -> list[Image.Image]:
    """Normalize various image input types to a list of PIL images.

    Zero-size images (width or height of 0) are filtered out because PIL
    cannot encode them to PNG, which would crash any reporter that tries.
    """
    if image is None:
        return []
    if isinstance(image, AnnotatedImage):
        images: list[Image.Image] = image.get_images()
    elif isinstance(image, list):
        images = image
    else:
        images = [image]
    return [img for img in images if img.width > 0 and img.height > 0]


def _format_duration(seconds: float) -> str:
    """Format a duration given in seconds as ``HH:MM:SS`` or
    ``HH:MM:SS.mmm`` for sub-second precision.

    Used by `SimpleHtmlReporter` to render both the overall execution time and
    per-conversation durations consistently.
    """
    total_seconds = max(float(seconds), 0.0)
    whole_seconds = int(total_seconds)
    millis = int(round((total_seconds - whole_seconds) * 1000))
    if millis == 1000:
        whole_seconds += 1
        millis = 0
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if whole_seconds == 0 and millis > 0:
        return f"{base}.{millis:03d}"
    return base


def _base64_media_label(media_type: str) -> str:
    """Return a human-friendly label for a base64 source's ``media_type``."""
    if media_type == "application/pdf":
        return "PDF"
    if media_type.startswith("image/"):
        return "image"
    return media_type


def truncate_base64_media(content: Any) -> Any:
    """Replace base64 media data with a placeholder to keep reports readable.

    Walks the message content recursively and replaces the ``data`` field of
    any base64 source block (matching the schema of ``Base64ImageSourceParam``
    / ``Base64PdfSourceParam``, i.e. ``{"type": "base64", "data": "...",
    "media_type": "..."}``) with a placeholder string. This covers screenshots,
    images, and PDF documents alike, so large binary blobs never bloat the
    report. All other content (including regular long strings like prompts or
    tool outputs) is left untouched.
    """
    if isinstance(content, dict):
        content_dict: dict[Any, Any] = content
        media_type = content_dict.get("media_type")
        if (
            isinstance(media_type, str)
            and content_dict.get("type") == "base64"
            and "data" in content_dict
        ):
            label = _base64_media_label(media_type)
            return {**content_dict, "data": f"[Base64 {label} data truncated]"}
        return {
            key: truncate_base64_media(value) for key, value in content_dict.items()
        }

    if isinstance(content, list):
        content_list: list[Any] = content
        return [truncate_base64_media(item) for item in content_list]

    return content


class Reporter(ABC):
    """Abstract base class for reporters. Cannot be instantiated directly.

    Defines the interface that all reporters must implement to be used with `askui.VisionAgent`.
    """

    @abstractmethod
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        """Add a message to the report.

        Args:
            role (str): The role of the message sender (e.g., `"User"`, `"Assistant"`,
                `"System"`)
            content (str | dict | list): The message content, which can be a string,
                dictionary, or list, e.g. `'click 2x times on text "Edit"'`
            image (PIL.Image.Image | list[PIL.Image.Image], optional): PIL Image or
                list of PIL Images to include with the message
        """
        raise NotImplementedError

    @abstractmethod
    def add_usage_summary(self, usage: UsageSummary) -> None:
        """Add usage statistics summary to the report.

        Called at the end of an ``act()`` execution with accumulated token
        usage and optional cost breakdown.

        Args:
            usage (UsageSummary): Accumulated usage statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """Add cache execution statistics showing token savings.

        Called when a cached trajectory is executed. The original_usage contains
        the token usage from when the cache was originally recorded.

        Args:
            original_usage (dict[str, int | None]): Token usage from cache recording:
                - input_tokens: Input tokens used during original recording
                - output_tokens: Output tokens used during original recording
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self) -> None:
        """Generates the final report.

        Implementing this method is only required if the report is not generated
        in "real-time", e.g., on calls of `add_message()`, but must be generated
        at the end of the execution.

        This method is called when the `askui.VisionAgent` context is exited or
        `askui.VisionAgent.close()` is called.
        """


class NullReporter(Reporter):
    """A reporter that does nothing."""

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        pass

    @override
    def add_usage_summary(self, usage: UsageSummary) -> None:
        pass

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        pass

    @override
    def generate(self) -> None:
        pass


NULL_REPORTER = NullReporter()


class ReporterErrorHandler(Reporter):
    """A reporter that handles errors by logging them and skipping the reporter."""

    def __init__(self, reporter: Reporter) -> None:
        self._reporter = reporter
        self._error_occurred = False

    def _log_on_exception(self, error: Exception) -> None:
        error_message = getattr(error, "message", str(error))
        logger.exception(
            "Skipping the usage of reporter %s due to the following exception: %s",
            self._reporter,
            error_message,
        )
        self._error_occurred = True

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        if self._error_occurred:
            logger.debug("Skipping reporter due to previous error")
            return
        try:
            self._reporter.add_message(role, content, image)
        except Exception as e:  # noqa: BLE001
            self._log_on_exception(e)

    @override
    def add_usage_summary(self, usage: UsageSummary) -> None:
        if self._error_occurred:
            logger.debug("Skipping reporter due to previous error")
            return
        try:
            self._reporter.add_usage_summary(usage)
        except Exception as e:  # noqa: BLE001
            self._log_on_exception(e)

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        if self._error_occurred:
            logger.debug("Skipping reporter due to previous error")
            return
        try:
            self._reporter.add_cache_execution_statistics(original_usage)
        except Exception as e:  # noqa: BLE001
            self._log_on_exception(e)

    @override
    def generate(self) -> None:
        if self._error_occurred:
            logger.debug("Skipping reporter due to previous error")
            return
        try:
            self._reporter.generate()
        except Exception as e:  # noqa: BLE001
            self._log_on_exception(e)


class CompositeReporter(Reporter):
    """A reporter that combines multiple reporters.

    Allows generating different reports simultaneously. Each message added will be forwarded to all
        reporters passed to the constructor. The reporters are called (`add_message()`, `generate()`) in
        the order they are ordered in the `reporters` list.

    Args:
        reporters (list[Reporter] | None, optional): List of reporters to combine
    """

    def __init__(self, reporters: list[Reporter] | None = None) -> None:
        self._reporters = [
            ReporterErrorHandler(reporter) for reporter in reporters or []
        ]

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        """Add a message to the report."""
        for reporter in self._reporters:
            reporter.add_message(role, content, image)

    @override
    def add_usage_summary(self, usage: UsageSummary) -> None:
        """Add usage summary to all reporters."""
        for reporter in self._reporters:
            reporter.add_usage_summary(usage)

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """Add cache execution statistics to all reporters."""
        for reporter in self._reporters:
            reporter.add_cache_execution_statistics(original_usage)

    @override
    def generate(self) -> None:
        """Generates the final report."""
        for report in self._reporters:
            report.generate()


class SystemInfo(TypedDict):
    platform: str
    python_version: str
    packages: list[str]


class SimpleHtmlReporter(Reporter):
    """A reporter that generates HTML reports with conversation logs and system information.

    Messages are streamed to a temporary file as they arrive so that base64-encoded
    screenshots are never held in memory all at once. The final report is assembled
    as a single self-contained HTML file on `generate()`.

    Args:
        report_dir (str, optional): Directory where reports will be saved.
            Defaults to `reports`.
    """

    def __init__(self, report_dir: str = "reports") -> None:
        self.report_dir = Path(report_dir)
        self._temp_messages_file: Path | None = None
        self.system_info = self._collect_system_info()
        self.usage_summary: UsageSummary | None = None
        self.cache_original_usage: dict[str, int | None] | None = None
        self._start_time: datetime | None = None

    def _collect_system_info(self) -> SystemInfo:
        """Collect system and Python information"""
        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "packages": sorted(
                [f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()]
            ),
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _format_content(self, content: Union[str, dict[str, Any], list[Any]]) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)
        return str(content)

    def _get_temp_messages_file(self) -> Path:
        """Return the path to the temporary messages file, creating it if needed."""
        if self._temp_messages_file is None or not self._temp_messages_file.exists():
            self.report_dir.mkdir(parents=True, exist_ok=True)
            _report_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            self._temp_messages_file = (
                self.report_dir / f"AskUI_report_{_report_ts}.tmp"
            )
        return self._temp_messages_file

    _MESSAGE_ROW_TEMPLATE = Template(
        '<tr class="{{ role_lower }}">'
        '<td class="timestamp">{{ ts_str }} UTC</td>'
        '<td><span class="role-badge role-{{ role_lower }}">{{ role }}</span></td>'
        '<td class="content-cell">'
        "{% if is_json %}"
        '<div class="json-content">'
        '<pre><code class="json">{{ content }}</code></pre>'
        "</div>"
        "{% else %}"
        "{{ content }}"
        "{% endif %}"
        "{% for image in images %}"
        '<br><img src="data:image/png;base64,{{ image }}" '
        'class="message-image" alt="Message image">'
        "{% endfor %}"
        "</td>"
        "</tr>\n"
    )

    def _render_message_row(
        self,
        timestamp: datetime,
        role: str,
        content: str,
        is_json: bool,
        images: list[str],
    ) -> str:
        """Render a single conversation message as an HTML table row."""
        return self._MESSAGE_ROW_TEMPLATE.render(
            role_lower=role.lower(),
            ts_str=timestamp.strftime("%H:%M:%S.%f")[:-3],
            role=role,
            content=content,
            is_json=is_json,
            images=images,
        )

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        """Add a message to the report.

        The rendered HTML row is written directly to a temporary file so that
        base64 image data is not accumulated in memory during long runs.
        """
        if self._start_time is None:
            self._start_time = datetime.now(tz=timezone.utc)

        _images = normalize_to_pil_images(image)
        _content = truncate_base64_media(content)

        timestamp = datetime.now(tz=timezone.utc)
        formatted_content = self._format_content(_content)
        is_json = isinstance(_content, (dict, list))
        image_b64s = [self._image_to_base64(img) for img in _images]

        row_html = self._render_message_row(
            timestamp, role, formatted_content, is_json, image_b64s
        )
        with self._get_temp_messages_file().open(mode="a", encoding="utf-8") as f:
            f.write(row_html)

    @override
    def add_usage_summary(self, usage: UsageSummary) -> None:
        """Store usage summary for inclusion in the report."""
        self.usage_summary = usage

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """Accumulate original cache usage for calculating savings.

        Called once per cache file. Token counts are summed so that the
        savings percentage reflects all cache files used during execution.
        """
        if self.cache_original_usage is None:
            self.cache_original_usage = dict(original_usage)
            return
        for key, value in original_usage.items():
            if value is not None:
                self.cache_original_usage[key] = (
                    self.cache_original_usage.get(key) or 0
                ) + value

    @override
    def generate(self) -> None:
        """Generate an HTML report file.

        Creates a timestamped HTML file in the `report_dir` containing:
        - System information
        - All collected messages with their content and images
        - Syntax-highlighted JSON content

        Message rows are streamed from a temporary file so that the full set of
        base64 images is never held in memory simultaneously.
        """
        _HEADER_TEMPLATE = """
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AskUI Python SDK Report - {{ timestamp }}</title>
                <link rel="stylesheet"
                    href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
                <script
                    src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js">
                </script>
                <style>
                    * {
                        box-sizing: border-box;
                    }

                    :root {
                        --bg-gradient-start: #1a1a2e;
                        --bg-gradient-end: #16213e;
                        --text-primary: #e0e0e0;
                        --text-secondary: #b0b0b0;
                        --text-muted: #888;
                        --accent-primary: rgb(34, 197, 94);
                        --accent-secondary: rgb(22, 163, 74);
                        --section-bg: rgba(30, 30, 46, 0.8);
                        --section-border: rgba(34, 197, 94, 0.3);
                        --table-bg: rgba(20, 20, 35, 0.6);
                        --table-border: rgba(34, 197, 94, 0.2);
                        --table-header-text: #fff;
                        --header-bg-start: rgba(34, 197, 94, 0.2);
                        --header-bg-end: rgba(34, 197, 94, 0.1);
                        --scrollbar-track: rgba(20, 20, 35, 0.5);
                        --scrollbar-thumb: rgba(34, 197, 94, 0.5);
                        --scrollbar-thumb-hover: rgb(34, 197, 94);
                        --code-bg: rgba(0, 0, 0, 0.4);
                        --shadow-color: rgba(0, 0, 0, 0.3);
                    }

                    [data-theme="light"] {
                        --bg-gradient-start: #ffffff;
                        --bg-gradient-end: #f4f4f5;
                        --text-primary: #0a0a0b;
                        --text-secondary: #52525b;
                        --text-muted: #71717a;
                        --accent-primary: rgb(34, 197, 94);
                        --accent-secondary: rgb(22, 163, 74);
                        --section-bg: #ffffff;
                        --section-border: #e4e4e7;
                        --table-bg: #f4f4f5;
                        --table-border: #e4e4e7;
                        --table-header-text: #0a0a0b;
                        --header-bg-start: rgba(34, 197, 94, 0.1);
                        --header-bg-end: rgba(34, 197, 94, 0.05);
                        --scrollbar-track: #f4f4f5;
                        --scrollbar-thumb: rgba(34, 197, 94, 0.3);
                        --scrollbar-thumb-hover: rgb(34, 197, 94);
                        --code-bg: #f1f5f9;
                        --shadow-color: rgba(0, 0, 0, 0.1);
                    }

                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                            'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                            sans-serif;
                        margin: 0;
                        padding: 0;
                        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
                        color: var(--text-primary);
                        line-height: 1.6;
                        min-height: 100vh;
                        transition: background 0.3s ease, color 0.3s ease;
                    }

                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 40px 20px;
                    }

                    .header {
                        background: linear-gradient(135deg, var(--header-bg-start) 0%, var(--header-bg-end) 100%);
                        border: 2px solid var(--accent-primary);
                        border-radius: 16px;
                        padding: 30px 40px;
                        margin-bottom: 40px;
                        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.3);
                        backdrop-filter: blur(10px);
                        position: relative;
                        transition: background 0.3s ease, border-color 0.3s ease;
                    }

                    .header h1 {
                        margin: 0 0 10px 0;
                        font-size: 2.5em;
                        font-weight: 700;
                        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    }

                    .header p {
                        margin: 0;
                        color: var(--text-secondary);
                        font-size: 0.95em;
                    }

                    .theme-toggle {
                        position: absolute;
                        top: 30px;
                        right: 40px;
                        background: var(--section-bg);
                        border: 2px solid var(--accent-primary);
                        border-radius: 25px;
                        width: 60px;
                        height: 30px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 2px 8px var(--shadow-color);
                    }

                    .theme-toggle:hover {
                        border-color: var(--accent-secondary);
                        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
                    }

                    .theme-toggle-slider {
                        position: absolute;
                        top: 3px;
                        left: 3px;
                        width: 24px;
                        height: 24px;
                        background: var(--accent-primary);
                        border-radius: 50%;
                        transition: all 0.3s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 14px;
                    }

                    [data-theme="light"] .theme-toggle-slider {
                        transform: translateX(30px);
                        background: var(--accent-secondary);
                    }

                    .theme-toggle-slider::before {
                        content: '🌙';
                        position: absolute;
                    }

                    [data-theme="light"] .theme-toggle-slider::before {
                        content: '☀️';
                    }

                    .section {
                        background: var(--section-bg);
                        border: 1px solid var(--section-border);
                        border-radius: 12px;
                        padding: 30px;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 20px var(--shadow-color);
                        backdrop-filter: blur(10px);
                        transition: background 0.3s ease, border-color 0.3s ease;
                    }

                    .section h2 {
                        margin: 0 0 20px 0;
                        font-size: 1.8em;
                        font-weight: 600;
                        color: var(--accent-primary);
                        border-bottom: 2px solid var(--section-border);
                        padding-bottom: 10px;
                        transition: color 0.3s ease, border-color 0.3s ease;
                    }

                    table {
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        margin-bottom: 20px;
                        background: var(--table-bg);
                        border-radius: 8px;
                        overflow: hidden;
                        transition: background 0.3s ease;
                    }

                    th, td {
                        padding: 16px;
                        text-align: left;
                        border-bottom: 1px solid var(--table-border);
                        transition: border-color 0.3s ease;
                    }

                    th {
                        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3) 0%, rgba(34, 197, 94, 0.2) 100%);
                        color: var(--table-header-text);
                        font-weight: 600;
                        text-transform: uppercase;
                        font-size: 0.85em;
                        letter-spacing: 0.5px;
                    }

                    tr:last-child td {
                        border-bottom: none;
                    }

                    .assistant {
                        background-color: rgba(34, 197, 94, 0.05);
                    }

                    .assistant:hover {
                        background-color: rgba(34, 197, 94, 0.1);
                        transition: background-color 0.2s ease;
                    }

                    .user {
                        background-color: rgba(20, 20, 35, 0.4);
                    }

                    .user:hover {
                        background-color: rgba(20, 20, 35, 0.6);
                        transition: background-color 0.2s ease;
                    }

                    .system {
                        background-color: rgba(22, 163, 74, 0.05);
                    }

                    .system:hover {
                        background-color: rgba(22, 163, 74, 0.1);
                        transition: background-color 0.2s ease;
                    }

                    .system-info {
                        width: auto;
                        min-width: 50%;
                    }

                    .system-info td {
                        color: var(--text-primary);
                    }

                    .package-list {
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono',
                            'Source Code Pro', monospace;
                        font-size: 0.9em;
                        line-height: 1.8;
                        color: var(--text-secondary);
                    }

                    .hidden-packages {
                        display: none !important;
                    }

                    .visible-packages {
                        display: block !important;
                    }

                    .show-more {
                        color: rgb(34, 197, 94);
                        cursor: pointer;
                        text-decoration: none;
                        margin-top: 10px;
                        display: inline-block;
                        padding: 8px 16px;
                        background: rgba(34, 197, 94, 0.1);
                        border: 1px solid #e4e4e7;
                        border-radius: 6px;
                        transition: all 0.2s ease;
                        font-weight: 500;
                    }

                    .show-more:hover {
                        background: rgba(34, 197, 94, 0.2);
                        border-color: #d4d4d8;
                        color: rgb(22, 163, 74);
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
                    }

                    .message-image {
                        max-width: 100%;
                        max-height: 600px;
                        margin: 15px 0;
                        border-radius: 8px;
                        border: 2px solid #e4e4e7;
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
                        transition: all 0.3s ease;
                        cursor: pointer;
                    }

                    .message-image:hover {
                        border-color: rgb(34, 197, 94);
                        box-shadow: 0 6px 24px rgba(34, 197, 94, 0.5);
                        transform: scale(1.02);
                    }

                    pre {
                        margin: 0;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }

                    pre code {
                        padding: 20px !important;
                        border-radius: 8px;
                        font-size: 14px;
                        display: block;
                        background: var(--code-bg) !important;
                        border: 1px solid var(--section-border);
                        transition: background 0.3s ease, border-color 0.3s ease;
                    }

                    .json-content {
                        background: rgba(0, 0, 0, 0.3);
                        border: 1px solid #e4e4e7;
                        border-radius: 8px;
                        margin: 10px 0;
                        overflow: hidden;
                    }

                    .role-badge {
                        display: inline-block;
                        padding: 6px 12px;
                        border-radius: 6px;
                        font-size: 0.85em;
                        font-weight: 600;
                        letter-spacing: 0.5px;
                    }

                    .role-assistant {
                        background: rgba(34, 197, 94, 0.2);
                        color: rgb(34, 197, 94);
                        border: 1px solid rgba(34, 197, 94, 0.4);
                    }

                    .role-user {
                        background: rgba(255, 255, 255, 0.1);
                        color: #fff;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                    }

                    .role-system {
                        background: rgba(22, 163, 74, 0.15);
                        color: rgb(22, 163, 74);
                        border: 1px solid rgba(22, 163, 74, 0.3);
                    }

                    .timestamp {
                        color: var(--text-muted);
                        font-size: 0.9em;
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono',
                            'Source Code Pro', monospace;
                    }

                    .content-cell {
                        color: var(--text-primary);
                        max-width: 800px;
                    }


                    .usage-subtitle {
                        color: var(--text-secondary);
                        margin-top: -8px;
                        margin-bottom: 16px;
                        font-size: 0.95em;
                    }

                    .nested-table {
                        margin: 12px 0 0 0;
                        width: 100%;
                    }

                    .nested-table th, .nested-table td {
                        padding: 10px 12px;
                        font-size: 0.9em;
                    }

                    .usage-breakdown-list {
                        display: grid;
                        gap: 12px;
                    }

                    .usage-breakdown-item {
                        border: 1px solid var(--section-border);
                        border-radius: 10px;
                        background: rgba(255, 255, 255, 0.02);
                        overflow: hidden;
                    }

                    .usage-breakdown-item summary {
                        list-style: none;
                        cursor: pointer;
                        padding: 14px 16px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        gap: 12px;
                        color: var(--text-primary);
                    }

                    .usage-breakdown-item summary::-webkit-details-marker {
                        display: none;
                    }

                    .usage-breakdown-item summary::after {
                        content: "▸";
                        color: var(--text-muted);
                        font-size: 0.95em;
                        transition: transform 0.2s ease;
                    }

                    .usage-breakdown-item[open] summary::after {
                        transform: rotate(90deg);
                    }

                    .usage-breakdown-title {
                        font-weight: 600;
                    }

                    .usage-breakdown-meta {
                        color: var(--text-secondary);
                        font-size: 0.9em;
                    }

                    .usage-breakdown-content {
                        padding: 0 16px 16px 16px;
                    }

                    .usage-breakdown-content table {
                        margin-top: 8px;
                    }

                    .mono {
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono',
                            'Source Code Pro', monospace;
                    }

                    @media (max-width: 768px) {
                        .container {
                            padding: 20px 10px;
                        }

                        .header {
                            padding: 20px;
                        }

                        .header h1 {
                            font-size: 1.8em;
                        }

                        .section {
                            padding: 20px;
                        }

                        th, td {
                            padding: 12px 8px;
                            font-size: 0.9em;
                        }

                        .message-image {
                            max-width: 100%;
                        }
                    }

                    /* Scrollbar styling */
                    ::-webkit-scrollbar {
                        width: 12px;
                        height: 12px;
                    }

                    ::-webkit-scrollbar-track {
                        background: var(--scrollbar-track);
                    }

                    ::-webkit-scrollbar-thumb {
                        background: var(--scrollbar-thumb);
                        border-radius: 6px;
                    }

                    ::-webkit-scrollbar-thumb:hover {
                        background: var(--scrollbar-thumb-hover);
                    }
                </style>
                <script>
                    function togglePackages() {
                        const hiddenPackages = document.getElementById(
                            'hiddenPackages'
                        );
                        const toggleButton = document.getElementById(
                            'toggleButton'
                        );

                        if (hiddenPackages.classList.contains('hidden-packages')) {
                            hiddenPackages.classList.remove('hidden-packages');
                            hiddenPackages.classList.add('visible-packages');
                            toggleButton.textContent = 'Show less';
                        } else {
                            hiddenPackages.classList.remove('visible-packages');
                            hiddenPackages.classList.add('hidden-packages');
                            toggleButton.textContent = 'Show more...';
                        }
                    }

                    function toggleTheme() {
                        const html = document.documentElement;
                        const currentTheme = html.getAttribute('data-theme');
                        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                        html.setAttribute('data-theme', newTheme);

                        // Update highlight.js theme
                        const link = document.querySelector('link[href*="highlight.js"]');
                        if (link) {
                            link.href = newTheme === 'light'
                                ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css'
                                : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css';
                            setTimeout(() => {
                                document.querySelectorAll('pre code').forEach((block) => {
                                    hljs.highlightElement(block);
                                });
                            }, 100);
                        }
                    }

                    document.addEventListener('DOMContentLoaded', () => {
                        document.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    });
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="theme-toggle" onclick="toggleTheme()" title="Toggle dark/light mode">
                            <div class="theme-toggle-slider"></div>
                        </div>
                        <h1>AskUI Python SDK Report</h1>
                        <p>Generated: {{ timestamp }}</p>
                    </div>

                    <div class="section">
                        <h2>System Information</h2>
                        <table class="system-info">
                            <tr>
                                <th>Platform</th>
                                <td>{{ system_info.platform }}</td>
                            </tr>
                            <tr>
                                <th>Python Version</th>
                                <td>{{ system_info.python_version }}</td>
                            </tr>
                            <tr>
                                <th>Installed Packages</th>
                                <td class="package-list">
                                    {% for package in system_info.packages[:5] %}
                                    {{ package }}<br>
                                    {% endfor %}
                                    {% if system_info.packages|length > 5 %}
                                        <div id="hiddenPackages" class="hidden-packages">
                                        {% for package in system_info.packages[5:] %}
                                            {{ package }}<br>
                                        {% endfor %}
                                        </div>
                                        <span id="toggleButton" class="show-more"
                                            onclick="togglePackages()">Show more...</span>
                                    {% endif %}
                                </td>
                            </tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Execution Statistics</h2>
                        <table class="system-info">
                            {% if execution_time_formatted is not none %}
                            <tr>
                                <th>Execution Time</th>
                                <td>{{ execution_time_formatted }}</td>
                            </tr>
                            {% endif %}
                            <tr>
                                <th>Estimated Token Cost <span style="font-weight:normal;color:var(--text-muted);">(actual cost may differ)</span></th>
                                <td>
                                    {% if usage_summary is not none and usage_summary.total_cost is not none %}
                                        ${{ "%.6f"|format(usage_summary.total_cost) }} {{ usage_summary.currency or 'USD' }}
                                        {% if cache_original_usage is not none and usage_summary.total_cost > 0
                                            and usage_summary.input_cost_per_million_tokens is not none
                                            and usage_summary.output_cost_per_million_tokens is not none %}
                                            {% set original_input = cache_original_usage.get('input_tokens') or 0 %}
                                            {% set original_output = cache_original_usage.get('output_tokens') or 0 %}
                                            {% set original_cache_read = cache_original_usage.get('cache_read_input_tokens') or 0 %}
                                            {% set original_cache_write = cache_original_usage.get('cache_creation_input_tokens') or 0 %}
                                            {% set original_cost = (original_input * usage_summary.input_cost_per_million_tokens + original_output * usage_summary.output_cost_per_million_tokens + original_cache_read * (usage_summary.cache_read_cost_per_million_tokens or 0) + original_cache_write * (usage_summary.cache_write_cost_per_million_tokens or 0)) / 1000000 %}
                                            {% if original_cost > 0 %}
                                                {% set savings_pct = ((usage_summary.total_cost - original_cost) / original_cost * 100) %}
                                                <span style="color: #22c55e; margin-left: 8px; font-weight: 600;">
                                                    ({{ "%.0f"|format(savings_pct) }}%)
                                                </span>
                                            {% endif %}
                                        {% endif %}
                                    {% else %}
                                        -
                                        <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                            (pricing not configured)
                                        </span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% if usage_summary is not none %}
                                {% if usage_summary.input_tokens is not none %}
                                <tr>
                                    <th>Input Tokens</th>
                                    <td>
                                        {{ "{:,}".format(usage_summary.input_tokens) }}
                                        {% if usage_summary.input_token_cost is not none %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (${{ "%.6f"|format(usage_summary.input_token_cost) }} {{ usage_summary.currency or 'USD' }})
                                            </span>
                                        {% endif %}
                                        {% if usage_summary.input_cost_per_million_tokens is not none and usage_summary.input_cost_per_million_tokens > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (Rate: ${{ "%.2f"|format(usage_summary.input_cost_per_million_tokens) }}/1M tokens)
                                            </span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endif %}
                                {% if usage_summary.output_tokens is not none %}
                                <tr>
                                    <th>Output Tokens</th>
                                    <td>
                                        {{ "{:,}".format(usage_summary.output_tokens) }}
                                        {% if usage_summary.output_token_cost is not none %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (${{ "%.6f"|format(usage_summary.output_token_cost) }} {{ usage_summary.currency or 'USD' }})
                                            </span>
                                        {% endif %}
                                        {% if usage_summary.output_cost_per_million_tokens is not none and usage_summary.output_cost_per_million_tokens > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (Rate: ${{ "%.2f"|format(usage_summary.output_cost_per_million_tokens) }}/1M tokens)
                                            </span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endif %}
                                {% if usage_summary.cache_creation_input_tokens is not none %}
                                <tr>
                                    <th>Cache Create Tokens</th>
                                    <td>
                                        {{ "{:,}".format(usage_summary.cache_creation_input_tokens) }}
                                        {% if usage_summary.cache_write_token_cost is not none and usage_summary.cache_write_token_cost > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (${{ "%.6f"|format(usage_summary.cache_write_token_cost) }} {{ usage_summary.currency or 'USD' }})
                                            </span>
                                        {% endif %}
                                        {% if usage_summary.cache_write_cost_per_million_tokens is not none and usage_summary.cache_write_cost_per_million_tokens > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (Rate: ${{ "%.2f"|format(usage_summary.cache_write_cost_per_million_tokens) }}/1M tokens)
                                            </span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endif %}
                                {% if usage_summary.cache_read_input_tokens is not none %}
                                <tr>
                                    <th>Cache Read Tokens</th>
                                    <td>
                                        {{ "{:,}".format(usage_summary.cache_read_input_tokens) }}
                                        {% if usage_summary.cache_read_token_cost is not none and usage_summary.cache_read_token_cost > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (${{ "%.6f"|format(usage_summary.cache_read_token_cost) }} {{ usage_summary.currency or 'USD' }})
                                            </span>
                                        {% endif %}
                                        {% if usage_summary.cache_read_cost_per_million_tokens is not none and usage_summary.cache_read_cost_per_million_tokens > 0 %}
                                            <span style="color: var(--text-muted); margin-left: 8px; font-size: 0.85em;">
                                                (Rate: ${{ "%.2f"|format(usage_summary.cache_read_cost_per_million_tokens) }}/1M tokens)
                                            </span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endif %}
                            {% endif %}
                        </table>
                    </div>
                    {% if usage_summary is not none and usage_summary.per_conversation_summaries %}
                    <div class="section">
                        <h2>Usage Breakdown</h2>
                        <p class="usage-subtitle">
                            Total usage grouped by conversation, with per-step token and cost details.
                        </p>
                        <div class="usage-breakdown-list">
                            {% for conversation_usage in usage_summary.per_conversation_summaries %}
                            {% set conversation_duration = format_conversation_duration(conversation_usage) %}
                            <details class="usage-breakdown-item">
                                <summary>
                                    <span class="usage-breakdown-title">
                                        Conversation #{{ conversation_usage.conversation_index }}
                                    </span>
                                    <span class="usage-breakdown-meta">
                                        {{ conversation_usage.step_summaries | length }} step(s),
                                        {% if conversation_duration is not none %}
                                            Duration: {{ conversation_duration }},
                                        {% endif %}
                                        Input {{ "{:,}".format(conversation_usage.input_tokens or 0) }},
                                        Output {{ "{:,}".format(conversation_usage.output_tokens or 0) }},
                                        Cache Create {{ "{:,}".format(conversation_usage.cache_creation_input_tokens or 0) }},
                                        Cache Read {{ "{:,}".format(conversation_usage.cache_read_input_tokens or 0) }}
                                        {% if conversation_usage.total_cost is not none and conversation_usage.total_cost > 0 %}
                                            , Cost:
                                            ${{ "%.6f"|format(conversation_usage.total_cost) }}
                                            {{ conversation_usage.currency or "USD" }}
                                        {% endif %}
                                    </span>
                                </summary>
                                <div class="usage-breakdown-content">
                                    {% set show_conversation_cost = conversation_usage.total_cost is not none and conversation_usage.total_cost > 0 %}
                                    <table class="nested-table">
                                        <tr>
                                            <th>Conversation ID</th>
                                            {% if conversation_duration is not none %}
                                            <th>Duration</th>
                                            {% endif %}
                                            <th>Input Tokens</th>
                                            <th>Output Tokens</th>
                                            <th>Cache Create</th>
                                            <th>Cache Read</th>
                                            {% if show_conversation_cost %}
                                            <th>Estimated Cost</th>
                                            {% endif %}
                                        </tr>
                                        <tr class="system">
                                            <td class="mono">{{ conversation_usage.conversation_id }}</td>
                                            {% if conversation_duration is not none %}
                                            <td>{{ conversation_duration }}</td>
                                            {% endif %}
                                            <td>{{ "{:,}".format(conversation_usage.input_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(conversation_usage.output_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(conversation_usage.cache_creation_input_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(conversation_usage.cache_read_input_tokens or 0) }}</td>
                                            {% if show_conversation_cost %}
                                            <td>
                                                ${{ "%.6f"|format(conversation_usage.total_cost or 0) }}
                                                {{ conversation_usage.currency or "USD" }}
                                            </td>
                                            {% endif %}
                                        </tr>
                                    </table>
                                    {% if conversation_usage.step_summaries %}
                                    {% set step_cost = namespace(show=false) %}
                                    {% for step_usage in conversation_usage.step_summaries %}
                                        {% if step_usage.total_cost is not none and step_usage.total_cost > 0 %}
                                            {% set step_cost.show = true %}
                                        {% endif %}
                                    {% endfor %}
                                    <table class="nested-table">
                                        <tr>
                                            <th>Step</th>
                                            <th>Input Tokens</th>
                                            <th>Output Tokens</th>
                                            <th>Cache Create</th>
                                            <th>Cache Read</th>
                                            {% if step_cost.show %}
                                            <th>Estimated Cost</th>
                                            {% endif %}
                                        </tr>
                                        {% for step_usage in conversation_usage.step_summaries %}
                                        <tr>
                                            <td>#{{ step_usage.step_index }}</td>
                                            <td>{{ "{:,}".format(step_usage.input_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(step_usage.output_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(step_usage.cache_creation_input_tokens or 0) }}</td>
                                            <td>{{ "{:,}".format(step_usage.cache_read_input_tokens or 0) }}</td>
                                            {% if step_cost.show %}
                                            <td>
                                                {% if step_usage.total_cost is not none and step_usage.total_cost > 0 %}
                                                    ${{ "%.6f"|format(step_usage.total_cost) }}
                                                    {{ step_usage.currency or "USD" }}
                                                {% endif %}
                                            </td>
                                            {% endif %}
                                        </tr>
                                        {% endfor %}
                                    </table>
                                    {% else %}
                                        <span style="color: var(--text-muted);">No step usage entries</span>
                                    {% endif %}
                                </div>
                            </details>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}


                    <div class="section">
                        <h2>Conversation Log</h2>
                        <table>
                            <tr>
                                <th>Time</th>
                                <th>Role</th>
                                <th>Content</th>
                            </tr>
        """

        _FOOTER = "            </table>\n                    </div>\n                </div>\n            </body>\n        </html>"

        # Calculate execution time
        end_time = datetime.now(tz=timezone.utc)
        execution_time_formatted: str | None = None
        if self._start_time is not None:
            execution_time_formatted = _format_duration(
                (end_time - self._start_time).total_seconds()
            )

        def _format_conversation_duration(
            conversation_usage: "ConversationUsageSummary",
        ) -> str | None:
            """Derive the formatted conversation duration from stored timestamps.

            Returns ``None`` if either ``started_at`` or ``ended_at`` is missing
            so the template can skip rendering.
            """
            if (
                conversation_usage.started_at is None
                or conversation_usage.ended_at is None
            ):
                return None
            return _format_duration(
                (
                    conversation_usage.ended_at - conversation_usage.started_at
                ).total_seconds()
            )

        header_html = Template(_HEADER_TEMPLATE).render(
            timestamp=end_time,
            system_info=self.system_info,
            usage_summary=self.usage_summary,
            cache_original_usage=self.cache_original_usage,
            execution_time_formatted=execution_time_formatted,
            format_conversation_duration=_format_conversation_duration,
        )

        report_path = (
            self.report_dir / f"report_{end_time:%Y%m%d%H%M%S%f}"
            f"{random.randint(0, 1000):03}.html"
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)

        with report_path.open(mode="w", encoding="utf-8") as out:
            out.write(header_html)
            try:
                if (
                    self._temp_messages_file is not None
                    and self._temp_messages_file.exists()
                ):
                    with self._temp_messages_file.open(
                        mode="r", encoding="utf-8"
                    ) as tmp:
                        shutil.copyfileobj(tmp, out)
                    self._temp_messages_file.unlink()
                    self._temp_messages_file = None
            finally:
                out.write(_FOOTER)


class AllureReporter(Reporter):
    """A reporter that integrates with Allure Framework for test reporting.

    This reporter creates Allure test reports by recording agent interactions as test steps
    and attaching screenshots. It requires one of the allure Python packages to be installed.

    The AllureReporter uses eager loading - it immediately checks for the allure dependency
    during initialization and raises an ImportError if not found.

    Raises:
        ImportError: If none of the required allure packages are installed during initialization.

    Example:
        ```python
        from askui import VisionAgent
        from askui.reporting import AllureReporter

        with VisionAgent(reporter=[AllureReporter()]) as agent:
            agent.act("Click the login button")
            # Each action becomes an allure step with screenshots attached
        ```

    Note:
        This reporter requires one of the following packages to be installed:
        - allure-python-commons
        - allure-pytest
        - allure-behave

        Install via: `pip install allure-python-commons`
    """

    def __init__(self) -> None:
        """Initialize the AllureReporter and import the allure module.

        Performs eager loading of the allure module. If the module is not available,
        raises ImportError immediately during initialization.

        Raises:
            ImportError: If the allure module cannot be imported. The error message
                provides installation instructions.
        """
        try:
            import allure  # type: ignore
        except ImportError:
            msg = (
                "AllureReporter requires the allure-python-commons', 'allure-pytest' or 'allure-behave' package. "
                "Please install it via 'pip install allure-python-commons'."
            )
            raise ImportError(msg) from None

        self.allure = allure

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        """Add a message as an Allure step with optional screenshots."""
        with self.allure.step(f"{role}: {str(content)}"):
            if image:
                _images = normalize_to_pil_images(image)
                for img in _images:
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    self.allure.attach(
                        img_bytes.getvalue(),
                        name="screenshot",
                        attachment_type=self.allure.attachment_type.PNG,
                    )

    @override
    def add_usage_summary(self, usage: UsageSummary) -> None:
        """No-op for AllureReporter - usage is not tracked."""

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """No-op for AllureReporter - cache statistics are not tracked."""

    @override
    def generate(self) -> None:
        """No-op for AllureReporter as reports are generated in real-time."""
