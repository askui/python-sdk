from pathlib import Path
from typing import Tuple

from PIL import Image

from askui.models.shared.tools import Tool
from askui.utils.llm_image_utils import compute_contained_size, resize_image


class LoadImageTool(Tool):
    """
    Tool for loading images from a directory on the filesystem.

    This tool allows the agent to load and process images from the filesystem,
    making them available for analysis, comparison, or visual inspection. It
    supports common image formats (PNG, JPEG, BMP, GIF, etc.) and automatically
    scales images to fit within a target size for efficient processing. This is
    useful for tasks like analyzing screenshots, comparing visual elements,
    verifying image content, or providing visual context during execution.

    Args:
        base_dir (str | Path): The base directory path where images will be loaded
            from. All image paths will be relative to this directory.

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import LoadImageTool

        with VisionAgent() as agent:
            agent.act(
                "Describe the logo image called 'logo.png'",
                tools=[LoadImageTool(base_dir="images")]
            )
        ```

    Example:
        ```python
        from askui import VisionAgent
        from askui.tools.store.universal import LoadImageTool

        with VisionAgent(
            act_tools=[LoadImageTool(base_dir="images")]
        ) as agent:
            agent.act("Describe the logo image called 'logo.png'")
        ```
    """

    def __init__(self, base_dir: str | Path) -> None:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        absolute = base_dir.absolute()
        super().__init__(
            name="load_image_tool",
            description=(
                "Loads an image from the filesystem and returns it for analysis or "
                f"processing. The base directory is set to '{absolute}' during tool "
                "initialization. All image paths are relative to this base directory. "
                "Supports common image formats (PNG, JPEG, BMP, GIF, etc.). Images are "
                "automatically scaled to fit within a target size for efficient "
                "processing. Use this tool to analyze screenshots, compare visual "
                "elements, verify image content, inspect UI elements, or provide "
                "visual context during execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "The relative path of the image file to load. The path is "
                            f"relative to the base directory '{absolute}' specified "
                            "during tool initialization. For example, if image_path "
                            "is 'screenshots/login.png', the image will be loaded from "
                            f"'{absolute}/screenshots/login.png'."
                        ),
                    },
                },
                "required": [
                    "image_path",
                ],
            },
        )
        self._base_dir = base_dir
        self._target_size = (1024, 768)

    def __call__(self, image_path: str = "") -> Tuple[str, Image.Image]:
        """
        Load an image from the specified path and return it for processing.

        The image is automatically scaled to fit within the target size (1024x768)
        while preserving aspect ratio, ensuring efficient processing without
        losing important visual details.

        Args:
            image_path (str): The relative path of the image file to load, relative
                to the base directory specified during tool initialization.

        Returns:
            Tuple[str, Image.Image]: A tuple containing a confirmation message
                indicating the image was successfully loaded (including the full
                absolute path) and the loaded PIL Image object respectively.

        Raises:
            FileNotFoundError: If the image file does not exist at the specified path.
            FileExistsError: If the path exists but is not a file (e.g., a directory).
        """
        absolute_image_path = self._base_dir / image_path

        if not absolute_image_path.exists():
            error_msg = f"Image not found: {absolute_image_path}"
            raise FileNotFoundError(error_msg)

        if not absolute_image_path.is_file():
            error_msg = f"Path is not a file: {absolute_image_path}"
            raise FileExistsError(error_msg)

        image = Image.open(absolute_image_path)
        target_size = compute_contained_size(
            image.width,
            image.height,
            self._target_size[0],
            self._target_size[1],
        )
        image = resize_image(image, target_size)

        return (
            f"Image was successfully loaded from {absolute_image_path}",
            image,
        )
