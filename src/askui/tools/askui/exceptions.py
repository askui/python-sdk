class AskUiControllerError(Exception):
    """Base exception for AskUI controller errors.

    This exception is raised when there is an error in the AskUI controller
    client, which handles the communication with the AskUI controller process
    running on the target computer.

    Args:
        message (str): The error message.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AskUiControllerOperationFailedError(AskUiControllerError):
    """Exception raised when a controller operation fails.

    This exception is raised when a specific operation in the AskUI controller
    fails, such as starting/stopping the controller or executing an action.

    Args:
        operation (str): The operation that failed.
        error (Exception): The original error that caused the failure.
    """

    def __init__(self, operation: str, error: Exception):
        super().__init__(f"Failed to {operation}: {error}")
        self.operation = operation
        self.original_error = error


class AskUiControllerOperationTimeoutError(AskUiControllerError):
    """Exception raised when a controller action times out.

    This exception is raised when an action in the AskUI controller takes longer
    than the expected timeout period to complete.

    Args:
        message (str): The error message.
        timeout_seconds (float | None): Optional timeout duration in seconds.
    """

    def __init__(
        self,
        message: str = (
            "Controller action did not finish within the expected time window."
        ),
        timeout_seconds: float | None = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class AskUiControllerInvalidCommandError(AskUiControllerError):
    """Exception raised when a command sent to the controller is invalid.

    This exception is raised when a command fails schema validation on the
    target computer side, typically due to malformed command structure or
    invalid parameters.

    Args:
        details (str | None): Optional additional error details from the target
            computer.
    """

    def __init__(self, details: str | None = None):
        error_msg = (
            "AgentOS: command validation failed on the target computer. "
            "This is typically caused by a malformed command or a version "
            "mismatch; updating the AskUI controller to the latest version "
            "may resolve it."
        )
        if details:
            error_msg += f"\nController details: {details}"
        super().__init__(error_msg)
        self.details = details


__all__ = [
    "AskUiControllerError",
    "AskUiControllerOperationFailedError",
    "AskUiControllerOperationTimeoutError",
    "AskUiControllerInvalidCommandError",
]
