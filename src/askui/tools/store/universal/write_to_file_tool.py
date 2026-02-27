from pathlib import Path

from askui.models.shared.tools import Tool


class WriteToFileTool(Tool):
    """
    Tool for writing content to files on the filesystem.

    This tool allows the agent to save text content to files, creating the
    necessary directory structure automatically if it doesn't exist. It supports
    both overwrite and append modes, making it useful for logging, saving results,
    creating reports, or storing any text-based data during execution.

    Args:
        base_dir (str | Path): The base directory path where files will be written.
            All file paths will be relative to this directory. The base directory
            will be created if it doesn't exist.

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import WriteToFileTool

        with VisionAgent() as agent:
            agent.act(
                "Extract the text from the page and save it to results/output.txt",
                tools=[WriteToFileTool(base_dir="/path/to/output")]
            )
        ```

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import WriteToFileTool

        with VisionAgent(
            act_tools=[WriteToFileTool(base_dir="/path/to/logs")]
        ) as agent:
            agent.act("Log the current state to logs/execution.log with append mode")
        ```
    """

    def __init__(self, base_dir: str | Path) -> None:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        absolute = base_dir.absolute()
        super().__init__(
            name="write_to_file_tool",
            description=(
                "Writes text content to a file on the filesystem. The file path is "
                f"relative to the base directory '{absolute}' specified during tool "
                "initialization. "
                "The directory structure will be created automatically if it doesn't "
                "exist. You can choose to overwrite existing files or append to them "
                "by setting the append parameter. Use this tool to save results, "
                "create reports, log information, or store any text-based data "
                "discovered or generated during execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "The relative path of the file to write to. The path is "
                            f"relative to the base directory '{absolute}' specified "
                            "during tool initialization. For example, if file_path "
                            "is 'config/settings.txt', the file will be written to "
                            f"'{absolute}/config/settings.txt'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The text content to write to the file. This can be any "
                            "string content including plain text, structured data, "
                            "logs, reports, or any other text-based information that "
                            "needs to be persisted to disk."
                        ),
                    },
                    "append": {
                        "type": "boolean",
                        "description": (
                            "Whether to append content to the existing file or "
                            "overwrite it. If `True`, the content will be appended to "
                            "the end of the file (useful for logging or accumulating "
                            "data). If `False` or not specified, any existing file "
                            "will be completely overwritten with the new content "
                            "(useful for saving results or reports)."
                        ),
                    },
                },
                "required": ["file_path", "content", "append"],
            },
        )
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.is_cacheable = False

    def __call__(self, file_path: str, content: str, append: bool) -> str:
        """
        Write the provided content to the specified file.

        Args:
            file_path (str): The relative path of the file where content should be
                written, relative to the base directory.
            content (str): The text content to write to the file.
            append (bool, optional): If `True`, appends content to the existing file.
                If `False` , overwrites the existing file.

        Returns:
            str: A confirmation message indicating where the file was written,
                including the full absolute path and whether content was appended
                or overwritten.

        Raises:
            OSError: If the file cannot be written due to filesystem errors.
        """
        absolute_file_path = self._base_dir / file_path
        absolute_file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with absolute_file_path.open(mode=mode, encoding="utf-8") as f:
            f.write(content)

        action = "appended to" if append else "written to"
        return f"Content was successfully {action} {absolute_file_path}."
