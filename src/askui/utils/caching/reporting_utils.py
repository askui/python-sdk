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
    detail: str | None = None,
) -> None:
    """Emit a caching event to the logger and (if present) the reporter.

    Args:
        reporter: The reporter to forward the message to, or ``None``.
        message: Concise headline of what/why the cache is doing.
        log: The module logger to write to.
        level: Logging level for the headline record (default ``logging.INFO``).
        detail: Optional longer description. It is always logged at ``INFO`` so a
            verbose reason does not bloat a higher-level (e.g. WARNING) headline,
            and it is appended to the reporter message so the full context is
            still surfaced there.
    """
    log.log(level, message)
    if detail:
        log.info(detail)
    if reporter is not None:
        reporter_message = message if not detail else f"{message} {detail}"
        try:
            reporter.add_message(CACHE_REPORTER_SOURCE, reporter_message)
        except Exception:  # noqa: BLE001 - reporting must never break caching
            log.debug("Failed to forward cache event to reporter", exc_info=True)
