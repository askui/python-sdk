import base64
import io
import json
import platform
import random
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from importlib.metadata import distributions
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from jinja2 import Template
from PIL import Image
from typing_extensions import TypedDict, override

from askui.utils.annotated_image import AnnotatedImage


def normalize_to_pil_images(
    image: Image.Image | list[Image.Image] | AnnotatedImage | None,
) -> list[Image.Image]:
    """Normalize various image input types to a list of PIL images."""
    if image is None:
        return []
    if isinstance(image, AnnotatedImage):
        return image.get_images()
    if isinstance(image, list):
        return image
    return [image]


def truncate_content(
    content: Any,
    max_string_length: int = 100000,
) -> Any:
    """Filter out long strings (i.e. the base64 image data) to keep reports readable."""
    if isinstance(content, str):
        if len(content) > max_string_length:
            return f"[truncated: {len(content)} characters]"
        return content

    if isinstance(content, dict):
        return {
            key: truncate_content(value, max_string_length)
            for key, value in content.items()
        }

    if isinstance(content, list):
        return [truncate_content(item, max_string_length) for item in content]

    # For other types (int, float, bool, None), return as-is
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
    def add_usage_summary(self, usage: dict[str, int | None]) -> None:
        """Add usage statistics summary to the report.

        Called at the end of an act() execution with accumulated token usage.

        Args:
            usage (dict[str, int | None]): Accumulated usage statistics containing:
                - input_tokens: Total input tokens sent to API
                - output_tokens: Total output tokens generated
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
    def add_usage_summary(self, usage: dict[str, int | None]) -> None:
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


class CompositeReporter(Reporter):
    """A reporter that combines multiple reporters.

    Allows generating different reports simultaneously. Each message added will be forwarded to all
        reporters passed to the constructor. The reporters are called (`add_message()`, `generate()`) in
        the order they are ordered in the `reporters` list.

    Args:
        reporters (list[Reporter] | None, optional): List of reporters to combine
    """

    def __init__(self, reporters: list[Reporter] | None = None) -> None:
        self._reporters = reporters or []

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
    def add_usage_summary(self, usage: dict[str, int | None]) -> None:
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

    Args:
        report_dir (str, optional): Directory where reports will be saved.
            Defaults to `reports`.
    """

    def __init__(self, report_dir: str = "reports") -> None:
        self.report_dir = Path(report_dir)
        self.messages: list[dict[str, Any]] = []
        self.system_info = self._collect_system_info()
        self.usage_summary: dict[str, int | None] | None = None
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

    @override
    def add_message(
        self,
        role: str,
        content: Union[str, dict[str, Any], list[Any]],
        image: Optional[Image.Image | list[Image.Image] | AnnotatedImage] = None,
    ) -> None:
        """Add a message to the report."""
        # Track start time from first message
        if self._start_time is None:
            self._start_time = datetime.now(tz=timezone.utc)

        _images = normalize_to_pil_images(image)
        _content = truncate_content(content)

        message = {
            "timestamp": datetime.now(tz=timezone.utc),
            "role": role,
            "content": self._format_content(_content),
            "is_json": isinstance(_content, (dict, list)),
            "images": [self._image_to_base64(img) for img in _images],
        }
        self.messages.append(message)

    @override
    def add_usage_summary(self, usage: dict[str, int | None]) -> None:
        """Store usage summary for inclusion in the report."""
        self.usage_summary = usage

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """Store original cache usage for calculating savings."""
        self.cache_original_usage = original_usage

    @override
    def generate(self) -> None:
        """Generate an HTML report file.

        Creates a timestamped HTML file in the `report_dir` containing:
        - System information
        - All collected messages with their content and images
        - Syntax-highlighted JSON content
        """
        template_str = """
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
                            {% if execution_time_seconds is not none %}
                            <tr>
                                <th>Execution Time</th>
                                <td>{{ "%.2f"|format(execution_time_seconds) }} seconds</td>
                            </tr>
                            {% endif %}
                            {% if usage_summary.get('input_tokens') is not none %}
                            <tr>
                                <th>Input Tokens</th>
                                <td>
                                    {{ "{:,}".format(usage_summary.get('input_tokens')) }}
                                    {% if cache_original_usage and cache_original_usage.get('input_tokens') %}
                                        {% set original = cache_original_usage.get('input_tokens') %}
                                        {% set current = usage_summary.get('input_tokens') %}
                                        {% set saved = original - current %}
                                        {% if saved > 0 and original > 0 %}
                                            {% set savings_pct = (saved / original * 100) %}
                                            <span style="color: #22c55e; margin-left: 8px;">({{ "%.1f"|format(savings_pct) }}% saved via trajectory caching)</span>
                                        {% endif %}
                                    {% endif %}
                                </td>
                            </tr>
                            {% endif %}
                            {% if usage_summary.get('output_tokens') is not none %}
                            <tr>
                                <th>Output Tokens</th>
                                <td>
                                    {{ "{:,}".format(usage_summary.get('output_tokens')) }}
                                    {% if cache_original_usage and cache_original_usage.get('output_tokens') %}
                                        {% set original = cache_original_usage.get('output_tokens') %}
                                        {% set current = usage_summary.get('output_tokens') %}
                                        {% set saved = original - current %}
                                        {% if saved > 0 and original > 0 %}
                                            {% set savings_pct = (saved / original * 100) %}
                                            <span style="color: #22c55e; margin-left: 8px;">({{ "%.1f"|format(savings_pct) }}% saved via trajectory caching)</span>
                                        {% endif %}
                                    {% endif %}
                                </td>
                            </tr>
                            {% endif %}
                            {% if cache_original_usage.get('input_tokens') is not none %}
                            <tr>
                                <th>Original Input Tokens</th>
                                <td>{{ "{:,}".format(cache_original_usage.get('input_tokens')) }}</td>
                            </tr>
                            {% endif %}
                            {% if cache_original_usage.get('output_tokens') is not none %}
                            <tr>
                                <th>Original Output Tokens</th>
                                <td>{{ "{:,}".format(cache_original_usage.get('output_tokens')) }}</td>
                            </tr>
                            {% endif %}
                        </table>
                    </div>

                    <div class="section">
                        <h2>Conversation Log</h2>
                        <table>
                            <tr>
                                <th>Time</th>
                                <th>Role</th>
                                <th>Content</th>
                            </tr>
                            {% for msg in messages %}
                                <tr class="{{ msg.role.lower() }}">
                                    <td class="timestamp">{{ msg.timestamp.strftime('%H:%M:%S.%f')[:-3] }} UTC</td>
                                    <td>
                                        <span class="role-badge role-{{ msg.role.lower() }}">
                                            {{ msg.role }}
                                        </span>
                                    </td>
                                    <td class="content-cell">
                                        {% if msg.is_json %}
                                            <div class="json-content">
                                                <pre><code class="json">{{ msg.content }}</code></pre>
                                            </div>
                                        {% else %}
                                            {{ msg.content }}
                                        {% endif %}
                                        {% for image in msg.images %}
                                            <br>
                                            <img src="data:image/png;base64,{{ image }}"
                                                class="message-image"
                                                alt="Message image">
                                        {% endfor %}
                                    </td>
                                </tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>
            </body>
        </html>
        """

        template = Template(template_str)

        # Calculate execution time
        end_time = datetime.now(tz=timezone.utc)
        execution_time_seconds: float | None = None
        if self._start_time is not None:
            execution_time_seconds = (end_time - self._start_time).total_seconds()

        html = template.render(
            timestamp=end_time,
            messages=self.messages,
            system_info=self.system_info,
            usage_summary=self.usage_summary,
            cache_original_usage=self.cache_original_usage,
            execution_time_seconds=execution_time_seconds,
        )

        report_path = (
            self.report_dir / f"report_{datetime.now(tz=timezone.utc):%Y%m%d%H%M%S%f}"
            f"{random.randint(0, 1000):03}.html"
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")


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
    def add_usage_summary(self, usage: dict[str, int | None]) -> None:
        """No-op for AllureReporter - usage is not tracked."""

    @override
    def add_cache_execution_statistics(
        self, original_usage: dict[str, int | None]
    ) -> None:
        """No-op for AllureReporter - cache statistics are not tracked."""

    @override
    def generate(self) -> None:
        """No-op for AllureReporter as reports are generated in real-time."""
