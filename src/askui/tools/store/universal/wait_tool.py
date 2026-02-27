"""Tool that waits for a specified duration without console output."""

import time

from askui.models.shared.tools import Tool


class WaitTool(Tool):
    """
    Tool for waiting a specified number of seconds without any console output.

    Use when a short, silent pause is needed between actions (e.g. after
    clicking before taking a screenshot). For longer waits or when the user
    should see progress, prefer `WaitWithProgressTool`.

    Args:
        max_wait_time (int, optional): Maximum allowed wait duration in seconds.
            Defaults to 3600 (1 hour).

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import WaitTool

        with VisionAgent() as agent:
            agent.act(
                "Click the button then wait 2 seconds and take a screenshot",
                tools=[WaitTool(max_wait_time=60)]
            )
        ```
    """

    def __init__(self, max_wait_time: int = 10 * 60) -> None:
        if max_wait_time < 1:
            msg = "Max wait time must be at least 1 second"
            raise ValueError(msg)
        super().__init__(
            name="wait_tool",
            description=(
                "Waits for a specified number of seconds without any console output. "
                "Use for short, silent waits (e.g. brief pause between actions). "
                "For longer waits or visible progress, use wait_with_progress_tool."
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
                },
                "required": ["wait_duration"],
            },
        )
        self._max_wait_time = max_wait_time
        self.is_cacheable = True

    def __call__(self, wait_duration: int) -> str:
        """
        Wait for the specified number of seconds.

        Args:
            wait_duration (int): Duration to wait in seconds (at least 1, at
                most the `max_wait_time` set at construction).

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
        time.sleep(wait_duration)
        return f"Finished waiting for {wait_duration} seconds."
