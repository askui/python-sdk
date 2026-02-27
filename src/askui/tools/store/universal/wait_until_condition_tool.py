"""Tool that waits until a condition is met or a maximum wait time is reached."""

import time
from typing import Callable

from askui.models.shared.tools import Tool
from askui.tools.utils import wait_with_progress


class WaitUntilConditionTool(Tool):
    """
    Tool for waiting until a condition is met or a timeout is reached.

    Polls a callable at a fixed interval. Returns as soon as the condition
    returns `True`, or when the maximum wait time is reached. During each
    interval between checks, a progress bar is shown in the console.

    Args:
        condition_check (Callable[[], bool]): Callable with no arguments
            invoked at each poll; return `True` when the condition is met,
            `False` otherwise.
        description (str): Short description of what the condition checks for,
            used in the tool description for the agent.
        max_wait_time (float, optional): Maximum time to wait in seconds.
            Defaults to 3600 (1 hour).

    Example:
        ```python
        from pathlib import Path
        from askui import VisionAgent
        from askui.tools.store.universal import WaitUntilConditionTool

        def file_ready() -> bool:
            return Path("output/result.json").exists()

        with VisionAgent() as agent:
            agent.act(
                "Wait until the result file appears",
                tools=[WaitUntilConditionTool(
                    condition_check=file_ready,
                    description="result file exists",
                    max_wait_time=300
                )]
            )
        ```
    """

    def __init__(
        self,
        condition_check: Callable[[], bool],
        description: str,
        max_wait_time: int = 60 * 60,
    ) -> None:
        if max_wait_time < 1:
            msg = "Max wait time must be at least 1 second"
            raise ValueError(msg)
        super().__init__(
            name="wait_until_condition_tool",
            description=(
                f"Waits for: {description}. "
                "Polls a condition at a given interval up to a maximum time; "
                "returns early if the condition is met, otherwise after timeout."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "max_wait_time": {
                        "type": "integer",
                        "description": (
                            "Maximum time to wait in seconds before giving up."
                        ),
                        "minimum": 1,
                        "maximum": int(max_wait_time),
                    },
                    "check_interval": {
                        "type": "integer",
                        "description": (
                            "Interval in seconds between condition checks "
                            "(e.g. 5 for every 5 seconds). Must be at least 1."
                        ),
                        "minimum": 1,
                        "maximum": int(max_wait_time),
                    },
                },
                "required": ["max_wait_time"],
            },
        )
        self._condition_check = condition_check
        self._max_wait_time = max_wait_time
        self.is_cacheable = True

    def __call__(self, max_wait_time: int, check_interval: int = 1) -> str:
        """
        Wait until the condition is met or the given timeout is reached.

        Args:
            max_wait_time (int): Maximum time to wait in seconds (must not
                exceed the limit set at construction).
            check_interval (int, optional): Seconds between condition checks.
                Defaults to 1. Must be at least 1 and not greater than
                `max_wait_time`.

        Returns:
            str: Message indicating either that the condition was met (with
                elapsed time) or that the timeout was reached.

        Raises:
            ValueError: If `max_wait_time` or `check_interval` are out of
                valid range.
        """
        if max_wait_time > self._max_wait_time:
            msg = f"max_wait_time must not exceed {self._max_wait_time} seconds"
            raise ValueError(msg)
        if check_interval < 1:
            msg = "check_interval must be at least 1 second"
            raise ValueError(msg)
        if check_interval > max_wait_time:
            msg = "check_interval must not exceed max_wait_time"
            raise ValueError(msg)

        start = time.monotonic()
        num_checks = 0
        while True:
            num_checks += 1
            if self._condition_check():
                elapsed = time.monotonic() - start
                return (
                    f"Condition met after {elapsed:.1f} seconds ({num_checks} checks)."
                )
            elapsed = time.monotonic() - start
            if elapsed >= max_wait_time:
                return (
                    f"Timeout after {max_wait_time} seconds "
                    f"(condition not met after {num_checks} checks)."
                )
            sleep_for = min(check_interval, max_wait_time - elapsed)
            if sleep_for > 0:
                wait_with_progress(
                    sleep_for,
                    f"Waiting for condition (check {num_checks})",
                )
