from PIL import Image

from askui.models.shared import ComputerBaseTool, ToolTags
from askui.tools.agent_os import ComputerAgentOS


class ComputerGetFileTool(ComputerBaseTool):
    """
    Reads a file at an absolute path on the computer under automation.

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.computer.experimental import ComputerGetFileTool

        with ComputerAgent(act_tools=[ComputerGetFileTool()]) as agent:
            agent.act("Read /home/user/notes.txt and summarize the contents")

        with ComputerAgent() as agent:
            agent.act(
                "Read /home/user/notes.txt and summarize the contents",
                tools=[ComputerGetFileTool()],
            )
        ```
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="get_file_tool",
            description=(
                "Reads a file at an absolute path on the computer under automation. "
                "Returns UTF-8 text as a string, or a decoded image for "
                "supported image formats. Unsupported binary types are rejected."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "absolute_file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the file on the target machine (for "
                            "example 'C:\\\\path\\\\notes.txt' on Windows or "
                            "'/home/user/notes.txt' on Linux/macOS)."
                        ),
                    },
                },
                "required": ["absolute_file_path"],
            },
            agent_os=agent_os,
            required_tags=[ToolTags.SCALED_AGENT_OS.value],
        )
        self.is_cacheable = True

    def __call__(self, absolute_file_path: str) -> Image.Image | str:
        return self.agent_os.get_file(absolute_file_path)
