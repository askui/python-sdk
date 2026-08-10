class DesktopAgentOsError(Exception):
    """Base class for Desktop Agent OS errors.

    This error is raised when an error occurs in the Desktop Agent OS.

    Inherits from `Exception` (not `BaseException`) so that the standard
    `except Exception` handlers in the tool-calling loop can catch it and
    surface it to the agent as a tool error result instead of crashing.
    """
