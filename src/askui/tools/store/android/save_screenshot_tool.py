from pathlib import Path

from askui.models.shared import AndroidBaseTool


class AndroidSaveScreenshotTool(AndroidBaseTool):
    """
    Tool for saving screenshots from the currently connected Android device to disk.

    This tool captures a screenshot of the current Android device screen and saves
    it to a specified location on the filesystem. The screenshot is saved as a PNG
    image file. The directory structure will be created automatically if it doesn't
    exist.

    Args:
        base_dir (str): The base directory path where screenshots will be saved.
            All screenshot paths will be relative to this directory.

    Example:
        ```python
        from askui import AndroidAgent
        from askui.tools.store.android import AndroidSaveScreenshotTool

        with AndroidAgent() as agent:
            agent.act(
                "Take a screenshot and save it as demo/demo.png",
                tools=[AndroidSaveScreenshotTool(base_dir="/path/to/screenshots")]
            )
        ```

    Example
    ```python
    from askui import AndroidAgent
    from askui.tools.store.android import AndroidSaveScreenshotTool

    with AndroidAgent(
        act_tools=[AndroidSaveScreenshotTool(base_dir="/path/to/screenshots")]
    ) as agent:
        agent.act("Take a screenshot and save it as demo/demo.png")
    """

    def __init__(self, base_dir: str) -> None:
        super().__init__(
            name="save_screenshot_tool",
            description=(
                "Saves a screenshot of the currently connected Android device screen "
                "to disk as a PNG image file. The screenshot is captured from the "
                "currently active display. The directory structure for the specified "
                "path will be created automatically if it doesn't exist. The PNG "
                "extension is automatically appended to the provided path."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "The relative path where the screenshot should be saved, "
                            "without the PNG extension. The path is relative to the "
                            "base directory specified during tool initialization. "
                            "For example, if base_dir is '/screenshots' and "
                            "image_path is 'test/my_screenshot', the file will be "
                            "saved as '/screenshots/test/my_screenshot.png'. "
                            "Subdirectories will be created automatically if needed."
                        ),
                    },
                },
                "required": ["image_path"],
            },
        )
        self._base_dir = base_dir
        self.is_cacheable = True

    def __call__(self, image_path: str) -> str:
        """
        Save a screenshot of the current Android device screen to disk.

        Args:
            image_path (str): The relative path where the screenshot should be saved,
                without the PNG extension. The path is relative to the base directory
                specified during tool initialization.

        Returns:
            str: A confirmation message indicating where the screenshot was saved,
                including the full absolute path.
        """
        absolute_image_path = Path(self._base_dir) / f"{image_path}.png"
        absolute_image_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.agent_os.screenshot()
        image.save(absolute_image_path, format="PNG")
        return (
            f"Screenshot of the current Android device screen saved to "
            f"{absolute_image_path}."
        )
