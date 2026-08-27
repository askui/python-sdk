"""Tests for caching observability (logs + reporter events)."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from askui.speaker.cache_executor import CacheExecutor, ExecutionResult
from askui.tools.caching_tools import VerifyCacheExecution
from askui.utils.caching.cache_manager import CacheManager
from askui.utils.caching.reporting_utils import (
    CACHE_REPORTER_SOURCE,
    report_cache_event,
)

logger = logging.getLogger(__name__)


class _CapturingReporter:
    """Minimal reporter capturing (role, content) pairs."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []

    def add_message(self, role: str, content: Any, _image: Any = None) -> None:
        self.messages.append((role, content))


def test_report_cache_event_logs_and_reports(caplog: Any) -> None:
    reporter = _CapturingReporter()
    with caplog.at_level(logging.INFO):
        report_cache_event(reporter, "something happened", log=logger)  # type: ignore[arg-type]
    assert ("Cache", "something happened") in [(r, c) for r, c in reporter.messages]
    assert any("something happened" in rec.message for rec in caplog.records)
    assert reporter.messages[0][0] == CACHE_REPORTER_SOURCE


def test_report_cache_event_without_reporter_only_logs(caplog: Any) -> None:
    with caplog.at_level(logging.WARNING):
        report_cache_event(None, "no reporter here", log=logger, level=logging.WARNING)
    assert any("no reporter here" in rec.message for rec in caplog.records)


def test_report_cache_event_detail_headline_warning_reason_info(caplog: Any) -> None:
    reporter = _CapturingReporter()
    with caplog.at_level(logging.INFO):
        report_cache_event(
            reporter,  # type: ignore[arg-type]
            "Cache invalidated and will not be reused.",
            log=logger,
            level=logging.WARNING,
            detail="Invalidation reason: something very long",
        )
    # Headline is a WARNING; the full reason is an INFO record.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("Cache invalidated" in r.message for r in warnings)
    assert any("something very long" in r.message for r in infos)
    # The reporter receives the combined message.
    assert "something very long" in reporter.messages[0][1]
    assert "Cache invalidated" in reporter.messages[0][1]


def _write_valid_cache(path: Path) -> None:
    path.write_text(
        '{"metadata": {"version": "0.3", "created_at": "2025-01-01T00:00:00Z", '
        '"is_valid": true, "execution_attempts": 0, "failures": []}, '
        '"trajectory": [], "cache_parameters": {}}',
        encoding="utf-8",
    )


def test_verify_failure_reports_reason_and_cache_name() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "login.json"
        _write_valid_cache(cache_path)

        executor = CacheExecutor()
        executor._cache_file = CacheManager.read_cache_file(cache_path)
        executor._cache_file_path = str(cache_path)

        reporter = _CapturingReporter()
        tool = VerifyCacheExecution(
            cache_executor=executor,
            cache_manager=CacheManager(),
            reporter=reporter,  # type: ignore[arg-type]
        )
        tool(success=False, verification_notes="button was missing")

        reported = " ".join(str(c) for _, c in reporter.messages)
        assert "login.json" in reported
        assert "button was missing" in reported
        assert "FAILED" in reported


def test_replay_failure_message_handles_missing_error_message() -> None:
    executor = CacheExecutor()
    reporter = _CapturingReporter()
    executor._reporter = reporter  # type: ignore[assignment]

    result = ExecutionResult(status="FAILED", step_index=2, error_message=None)
    executor._handle_failed(CacheManager(), result)

    reported = " ".join(str(c) for _, c in reporter.messages)
    assert "unknown error" in reported
    assert "None" not in reported
