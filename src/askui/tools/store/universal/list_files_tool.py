from pathlib import Path

from askui.models.shared.tools import Tool


class ListFilesTool(Tool):
    """
    Tool for listing files and directories in a directory on the filesystem.

    This tool allows the agent to explore the filesystem structure, discover
    available files and directories, and navigate the directory tree. It's useful
    for finding files, understanding project structure, or exploring data
    directories during execution.

    Args:
        base_dir (str | Path): The base directory path where file listing will start.
            All directory paths will be relative to this directory.

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import ListFilesTool

        with VisionAgent() as agent:
            agent.act(
                "List all files in the output directory",
                tools=[ListFilesTool(base_dir="/path/to/project")]
            )
        ```
    """

    def __init__(self, base_dir: str | Path) -> None:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        absolute = base_dir.absolute()
        super().__init__(
            name="list_files_tool",
            description=(
                "Lists files and directories in a directory on the filesystem. The "
                f"base directory is set to '{absolute}' during tool initialization. "
                "All directory paths are relative to this base directory. Use this "
                "tool to explore the filesystem structure, discover available files "
                "and directories, or navigate the directory tree during execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": (
                            "The relative path of the directory to list. The path is "
                            f"relative to the base directory '{absolute}' specified "
                            "during tool initialization. For example, if "
                            "directory_path is 'output', the directory will be listed "
                            f"from '{absolute}/output'. If not specified or empty, "
                            f"lists the base directory '{absolute}' itself."
                        ),
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": (
                            "Whether to list files recursively in subdirectories. If "
                            "`True`, all files and directories in subdirectories will "
                            "be included. If `False` or not specified, only files and "
                            "directories in the specified directory will be listed."
                        ),
                    },
                },
                "required": [],
            },
        )
        self._base_dir = Path(base_dir)
        self.is_cacheable = True

    def __call__(self, directory_path: str = "", recursive: bool = False) -> str:
        """
        List files and directories in the specified directory.

        Args:
            directory_path (str, optional): The relative path of the directory to list,
                relative to the base directory. If not specified or empty, lists the
                base directory itself.
            recursive (bool, optional): If `True`, lists files recursively in
                subdirectories. If `False` or not specified, only lists files in the
                specified directory.

        Returns:
            str: A formatted string listing all files and directories found, including
                their paths and types (file or directory), or an error message if the
                directory cannot be accessed.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path exists but is not a directory.
            OSError: If the directory cannot be accessed due to filesystem errors.
        """
        if directory_path:
            absolute_dir_path = self._base_dir / directory_path
        else:
            absolute_dir_path = self._base_dir

        if not absolute_dir_path.exists():
            error_msg = f"Directory not found: {absolute_dir_path}"
            raise FileNotFoundError(error_msg)

        if not absolute_dir_path.is_dir():
            error_msg = f"Path is not a directory: {absolute_dir_path}"
            raise NotADirectoryError(error_msg)

        items: list[str] = []
        if recursive:
            for item in absolute_dir_path.rglob("*"):
                relative_path = item.relative_to(self._base_dir)
                item_type = "directory" if item.is_dir() else "file"
                items.append(f"{item_type}: {relative_path}")
        else:
            for item in sorted(absolute_dir_path.iterdir()):
                relative_path = item.relative_to(self._base_dir)
                item_type = "directory" if item.is_dir() else "file"
                items.append(f"{item_type}: {relative_path}")

        if not items:
            return f"Directory {absolute_dir_path} is empty."

        result = f"Contents of {absolute_dir_path}:\n\n"
        result += "\n".join(items)
        return result
