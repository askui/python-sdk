from pathlib import Path

from askui.models.shared.tools import Tool
from askui.utils.pdf_utils import PdfSource


class LoadPdfTool(Tool):
    """
    Tool for loading PDF documents from a directory on the filesystem.

    This tool allows the agent to load PDF files and hand them to the model for
    analysis. The PDF is passed through unchanged as a document, so the model
    can reason about text, tables, charts, and layout. Use it to read reports,
    contracts, forms, or any other PDF-based content during execution.

    Args:
        base_dir (str | Path): The base directory path where PDFs will be loaded
            from. All PDF paths will be relative to this directory.

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.universal import LoadPdfTool

        with ComputerAgent() as agent:
            agent.act(
                "Summarize the key findings in 'report.pdf'",
                tools=[LoadPdfTool(base_dir="documents")],
            )
        ```

    Example:
        ```python
        from askui import ComputerAgent
        from askui.tools.store.universal import LoadPdfTool

        with ComputerAgent(
            act_tools=[LoadPdfTool(base_dir="documents")]
        ) as agent:
            agent.act("Summarize the key findings in 'report.pdf'")
        ```
    """

    def __init__(self, base_dir: str | Path) -> None:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        absolute = base_dir.absolute()
        super().__init__(
            name="load_pdf_tool",
            description=(
                "Loads a PDF document from the filesystem and returns it for "
                f"analysis. The base directory is set to '{absolute}' during tool "
                "initialization. All PDF paths are relative to this base directory. "
                "The document is passed to the model in full (every page as both "
                "text and image), so use this tool to read reports, contracts, "
                "forms, or any PDF-based content, and to reason about its text, "
                "tables, charts, and layout."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": (
                            "The relative path of the PDF file to load. The path is "
                            f"relative to the base directory '{absolute}' specified "
                            "during tool initialization. For example, if pdf_path is "
                            "'reports/q4.pdf', the PDF will be loaded from "
                            f"'{absolute}/reports/q4.pdf'."
                        ),
                    },
                },
                "required": [
                    "pdf_path",
                ],
            },
        )
        self._base_dir = base_dir
        self.is_cacheable = True

    def __call__(self, pdf_path: str = "") -> tuple[str, PdfSource]:
        """
        Load a PDF from the specified path and return it for processing.

        Args:
            pdf_path (str): The relative path of the PDF file to load, relative to
                the base directory specified during tool initialization.

        Returns:
            tuple[str, PdfSource]: A tuple containing a confirmation message
                indicating the PDF was successfully loaded (including the full
                absolute path) and the loaded `PdfSource` respectively.

        Raises:
            FileNotFoundError: If the PDF file does not exist at the specified path.
            FileExistsError: If the path exists but is not a file (e.g., a directory).
        """
        absolute_pdf_path = self._base_dir / pdf_path

        if not absolute_pdf_path.exists():
            error_msg = f"PDF not found: {absolute_pdf_path}"
            raise FileNotFoundError(error_msg)

        if not absolute_pdf_path.is_file():
            error_msg = f"Path is not a file: {absolute_pdf_path}"
            raise FileExistsError(error_msg)

        return (
            f"PDF was successfully loaded from {absolute_pdf_path}",
            PdfSource(absolute_pdf_path),
        )
