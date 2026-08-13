from askui.models.exceptions import AutomationError


class DesktopAgentOsError(AutomationError):
    """Unfixable error raised by the Desktop Agent OS.

    Raised when the Desktop Agent OS returns a response that violates the
    expected protocol (e.g. an unexpected response type or a response missing
    both an error and a payload). These indicate a broken controller or
    connection rather than something the agent can recover from, so - like
    other `AutomationError`s - they are re-raised by the tool-calling loop and
    terminate the run.

    For failures the agent can react to and work around (e.g. a path that does
    not exist), raise `DesktopAgentOsException` instead.
    """


class DesktopAgentOsException(Exception):  # noqa: N818
    """Recoverable error raised by the Desktop Agent OS.

    Raised when an operation on the Desktop Agent OS fails in a way the agent
    can react to and work around - for example, reading a file or directory
    that does not exist, or a file whose contents cannot be decoded. Because it
    derives from `Exception` (and not `AutomationError`), the tool-calling loop
    catches it and surfaces it to the agent as a tool error result instead of
    terminating the run.

    For unfixable protocol violations, raise `DesktopAgentOsError` instead.
    """
