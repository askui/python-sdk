"""Tool that returns the current date and time in the local timezone."""

from datetime import datetime

from askui.models.shared.tools import Tool


class GetCurrentTimeTool(Tool):
    """
    Tool for returning the current date and time in the local timezone.

    This tool allows the agent to know the current time when scheduling,
    logging, or time-dependent decisions. The time is formatted in a
    human-readable way including the timezone.

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import GetCurrentTimeTool

        with VisionAgent() as agent:
            agent.act(
                "What time is it? Plan the next step based on current time.",
                tools=[GetCurrentTimeTool()]
            )
        ```
    """

    def __init__(self) -> None:
        super().__init__(
            name="get_current_time_tool",
            description=(
                "Returns the current date and time in the local timezone. "
                "Use this tool when you need to know the current time for scheduling, "
                "logging, or time-dependent decisions."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        self.is_cacheable = True

    def __call__(self) -> str:
        """
        Return the current date and time in the local timezone.

        Returns:
            str: Human-readable string with today's date, current time, and
                timezone (e.g. "Today is February 18, 2025 and currently
                it is 14:30:00 CET").
        """
        now = datetime.now().astimezone()
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%H:%M:%S")
        tz_str = now.strftime("%Z") or now.strftime("%z")
        return f"Today is {date_str} and currently it is {time_str} {tz_str}"
