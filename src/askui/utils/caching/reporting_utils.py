"""Helpers for surfacing caching activity to both logs and the reporter.

Caching decisions (cache hit/miss, replay progress, pauses, verification
outcomes, invalidation, recording) are relevant to users trying to understand
"what and why" the cache is doing. These helpers emit a single message to the
standard logger AND, when a reporter is available, to the reporter so the
information also shows up in the HTML report / attached reporters instead of only
on stderr.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askui.reporting import Reporter

# Source/role used for caching messages in reporters.
CACHE_REPORTER_SOURCE = "Cache"


def report_cache_event(
    reporter: "Reporter | None",
    message: str,
    *,
    log: logging.Logger,
    level: int = logging.INFO,
) -> None:
    """Emit a caching event to the logger and (if present) the reporter.

    Args:
        reporter: The reporter to forward the message to, or ``None``.
        message: Human-readable description of what/why the cache is doing.
        log: The module logger to write to.
        level: Logging level for the log record (default ``logging.INFO``).
    """
    log.log(level, message)
    if reporter is not None:
        try:
            reporter.add_message(CACHE_REPORTER_SOURCE, message)
        except Exception:  # noqa: BLE001 - reporting must never break caching
            log.debug("Failed to forward cache event to reporter", exc_info=True)
