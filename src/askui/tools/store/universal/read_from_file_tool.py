from pathlib import Path

from askui.models.shared.tools import Tool


class ReadFromFileTool(Tool):
    """
    Tool for reading content from files on the filesystem.

    This tool allows the agent to read text content from files, making it useful
    for loading configuration files, reading data files, processing logs, or
    accessing any text-based data stored on the filesystem during execution.

    Args:
        base_dir (str | Path): The base directory path where files will be read from.
            All file paths will be relative to this directory.

        encodings (list[str]): The list of encodings to try to read the file.
            If not provided, the default encodings will be used. The default encodings
            are "utf-8" and "latin-1".

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import ReadFromFileTool

        with VisionAgent() as agent:
            agent.act(
                "Read the configuration from config/settings.txt and use it",
                tools=[ReadFromFileTool(base_dir="/path/to/project")]
            )
        ```
    """

    def __init__(
        self,
        base_dir: str | Path,
        encodings: list[str] | None = None,
    ) -> None:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        absolute = base_dir.absolute()
        super().__init__(
            name="read_from_file_tool",
            description=(
                "Reads text content from a file on the filesystem. The base directory "
                f"is set to '{absolute}' during tool initialization. All file paths are"
                " relative to this base directory. Use this tool to load configuration "
                "files, read data files, process logs, or access any text-based data "
                "stored on the filesystem during execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "The relative path of the file to read from. The path is "
                            f"relative to the base directory '{absolute}' specified "
                            "during tool initialization. For example, if file_path "
                            "is 'config/settings.txt', the file will be read from "
                            f"'{absolute}/config/settings.txt'."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        )
        self._encodings = encodings or ["utf-8", "latin-1"]
        self._base_dir = Path(base_dir)
        self.is_cacheable = True

    def __call__(self, file_path: str) -> str:
        """
        Read the content from the specified file.

        Args:
            file_path (str): The relative path of the file to read from, relative to
                the base directory.

        Returns:
            str: The content of the file as a string, or an error message if the file
                cannot be read.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read due to filesystem errors.
            RuntimeError: If the file cannot be read due to encoding errors.
        """
        absolute_file_path = self._base_dir / file_path

        if not absolute_file_path.exists():
            error_msg = f"File not found: {absolute_file_path}"
            raise FileNotFoundError(error_msg)

        if not absolute_file_path.is_file():
            error_msg = f"Path is not a file: {absolute_file_path}"
            raise ValueError(error_msg)

        content = None
        for encoding in self._encodings:
            try:
                content = absolute_file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if not content:
            error_msg = (
                f"Failed to read file {absolute_file_path} with any"
                f" of the encodings: {', '.join(self._encodings)}"
            )
            raise RuntimeError(error_msg)
        return f"Content of {absolute_file_path}:\n\n{content}"
