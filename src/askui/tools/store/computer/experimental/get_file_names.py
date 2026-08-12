from askui.models.shared import ComputerBaseTool
from askui.tools.agent_os import ComputerAgentOS


class ComputerGetFileNamesTool(ComputerBaseTool):
    """
    Lists regular files (not subdirectories) in a directory on the computer under
    automation.

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.computer.experimental import ComputerGetFileNamesTool

        with ComputerAgent(act_tools=[ComputerGetFileNamesTool()]) as agent:
            agent.act("List the regular files in /home/user/Documents")

        with ComputerAgent() as agent:
            agent.act(
                "List the regular files in /home/user/Documents",
                tools=[ComputerGetFileNamesTool()],
            )
        ```
    """

    def __init__(self, agent_os: ComputerAgentOS | None = None) -> None:
        super().__init__(
            name="get_file_names_tool",
            description=(
                "Lists the names of regular files in an absolute directory on the "
                "computer under automation. IMPORTANT: only files are returned - "
                "folders and subdirectories are NOT included in the result and this "
                "tool cannot be used to discover or navigate them. Use absolute "
                "paths as on the target machine. Returns names only; use "
                "get_file_tool to read a file's contents."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "absolute_directory_path": {
                        "type": "string",
                        "description": (
                            "Absolute path of the directory to scan (for example "
                            "'C:\\\\Users\\\\Public' on Windows or '/home/user/Public' "
                            "on Linux/macOS)."
                        ),
                    },
                },
                "required": ["absolute_directory_path"],
            },
            agent_os=agent_os,
        )
        self.is_cacheable = True

    def __call__(self, absolute_directory_path: str) -> str:
        names = self.agent_os.get_file_names(absolute_directory_path)
        file_names = ",".join(f"'{n}'" for n in names)
        return (
            f"Files in '{absolute_directory_path}' ({len(names)} files): {file_names} "
        )
