from askui.models.shared.tools import Tool


class PrintToConsoleTool(Tool):
    """
    Tool for printing messages to the console to communicate with the user.

    This tool allows the agent to provide real-time updates, status information,
    progress notifications, or any other messages directly to the user's console
    output. It is particularly useful for keeping users informed about the current
    state of execution, intermediate results, or important notifications during
    long-running operations.

    Args:
        source_name (str | None, optional): An optional prefix to prepend to all
            printed messages. If provided, messages will be formatted as
            "[source_name]: content". This helps identify the source of messages
            when multiple tools or components are printing to the console.

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import PrintToConsoleTool

        with VisionAgent() as agent:
            agent.act(
                "Click the login button and print a status message",
                tools=[PrintToConsoleTool(source_name="Agent")]
            )
        ```
    """

    def __init__(self, source_name: str | None = None):
        super().__init__(
            name="print_to_console_tool",
            description=(
                "Prints a message to the console to communicate with the user. "
                "Use this tool to provide updates about the current execution state, "
                "progress information, intermediate results, or important "
                "notifications. This is particularly useful for long-running "
                "operations or when you need to inform the user about what actions "
                "are being performed or what has been discovered during execution. "
                "The message will be displayed in the user's console output "
                "immediately."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "The message content to print to the console. This should "
                            "be a clear, informative message that helps the user "
                            "understand the current state of execution, progress made, "
                            "or any important information discovered. Keep messages "
                            "concise but descriptive."
                        ),
                    },
                },
                "required": ["content"],
            },
        )
        self._source_name = source_name
        self.is_cacheable = False

    def __call__(self, content: str) -> str:
        """
        Print the provided content to the console.

        Args:
            content (str): The message content to print to the console.

        Returns:
            str: A confirmation message indicating that the content was successfully
                printed to the console.
        """
        if self._source_name is not None:
            formatted_content = f"[{self._source_name}]: {content}"
        else:
            formatted_content = content
        print(formatted_content, flush=True)
        return "Message was successfully printed to the console."
