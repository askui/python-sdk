"""Tool that waits for a specified duration while showing a progress bar."""

from askui.models.shared.tools import Tool
from askui.tools.utils import wait_with_progress


class WaitWithProgressTool(Tool):
    """
    Tool for waiting a specified number of seconds with a console progress bar.

    Use when the agent needs to wait for something (e.g. page load, animation)
    and the user should see progress. For short silent pauses, use `WaitTool`
    instead.

    Args:
        max_wait_time (int, optional): Maximum allowed wait duration in seconds.
            Defaults to 3600 (1 hour).

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import WaitWithProgressTool

        with VisionAgent() as agent:
            agent.act(
                "Submit the form and wait 10 seconds for the page to load",
                tools=[WaitWithProgressTool(max_wait_time=120)]
            )
        ```
    """

    def __init__(self, max_wait_time: int = 10 * 60) -> None:
        if max_wait_time < 1:
            msg = "Max wait time must be at least 1 second"
            raise ValueError(msg)
        super().__init__(
            name="wait_with_progress_tool",
            description=(
                "Waits for a specified number of seconds (e.g. page load, "
                "animation). Displays a progress bar in the console during "
                "the wait. Use when the user should see that the agent is "
                "waiting; for short silent pauses use wait_tool instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wait_duration": {
                        "type": "integer",
                        "description": (
                            "Duration of the wait in seconds "
                            "(must be an integer, e.g. 5 for 5 seconds)."
                        ),
                        "minimum": 1,
                        "maximum": max_wait_time,
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Optional short label shown next to the progress bar "
                            "(e.g. 'Waiting for page load'). If omitted, "
                            "'Waiting' is used."
                        ),
                    },
                },
                "required": ["wait_duration"],
            },
        )
        self._max_wait_time = max_wait_time
        self.is_cacheable = True

    def __call__(self, wait_duration: int, message: str = "Waiting") -> str:
        """
        Wait for the specified duration and show a progress bar.

        Args:
            wait_duration (int): Duration to wait in seconds (at least 1, at
                most the `max_wait_time` set at construction).
            message (str, optional): Label shown next to the progress bar.
                Defaults to "Waiting".

        Returns:
            str: Confirmation message after the wait completes.

        Raises:
            ValueError: If `wait_duration` is less than 1 or exceeds
                `max_wait_time`.
        """
        if wait_duration < 1:
            msg = "Wait duration must be at least 1 second"
            raise ValueError(msg)
        if wait_duration > self._max_wait_time:
            msg = f"Wait duration must not exceed {self._max_wait_time} seconds"
            raise ValueError(msg)
        wait_with_progress(wait_duration, message)
        return f"Finished waiting for {wait_duration} seconds."
