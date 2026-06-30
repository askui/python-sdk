"""Example demonstrating PDF document support.

The agent can read PDF documents and reason about their text, tables, charts, and
layout. A PDF is forwarded to the model unchanged - as a base64 ``document`` block to
Anthropic Claude and as a ``file`` content part to OpenAI - so no Markdown conversion
is performed. PDFs must not exceed 32MB; a larger file raises ``PdfTooLargeError``.

Three entry points are shown:
1. `agent.get(source=...)` - extract information directly from a PDF on disk.
2. `LoadPdfTool` - let `act()` load a PDF from disk during execution.
3. `ComputerGetFileTool` - read a file off the computer under automation; PDFs come
   back as a `PdfSource` (text files as `str`, images as `PIL.Image.Image`).

A custom tool returning a `PdfSource` is also shown - any tool may hand a PDF to the
model the same way returning a `PIL.Image.Image` produces an image.

Required environment variables (see .env):
- ASKUI_WORKSPACE_ID, ASKUI_TOKEN - for the default AskUI model stack

Drop a `sample.pdf` next to this file (or change `PDF_PATH`) before running the
on-disk examples.
"""

import logging
from pathlib import Path

from askui import ComputerAgent
from askui.models.shared.tools import Tool
from askui.tools.store.computer.experimental import ComputerGetFileTool
from askui.tools.store.universal import LoadPdfTool
from askui.utils.pdf_utils import PdfSource

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
PDF_PATH = HERE / "sample.pdf"


def extract_from_pdf_file() -> None:
    """Extract information straight from a PDF on disk via `get()`.

    No screen interaction is needed - the PDF itself is the source.
    """
    if not PDF_PATH.exists():
        logger.warning("No PDF at %s - skipping extract_from_pdf_file()", PDF_PATH)
        return

    with ComputerAgent() as agent:
        summary = agent.get(
            "Summarize the key points of this document in 3 bullet points",
            source=str(PDF_PATH),
        )
        logger.info("PDF summary:\n%s", summary)


def load_pdf_during_act() -> None:
    """Let `act()` load a PDF from disk through `LoadPdfTool`.

    `LoadPdfTool` resolves paths relative to its `base_dir`; the loaded PDF is handed
    to the model in full (every page as both text and image).
    """
    if not PDF_PATH.exists():
        logger.warning("No PDF at %s - skipping load_pdf_during_act()", PDF_PATH)
        return

    with ComputerAgent(act_tools=[LoadPdfTool(base_dir=str(HERE))]) as agent:
        agent.act(
            f"Load '{PDF_PATH.name}', tell me what it is about, and list any headings "
            "you find."
        )


def read_pdf_from_target_machine() -> None:
    """Read a PDF off the computer under automation with `ComputerGetFileTool`.

    The controller returns the file decoded as a `PdfSource`, which the SDK forwards
    to the model as a document block so it can reason over the full PDF.
    """
    with ComputerAgent(act_tools=[ComputerGetFileTool()]) as agent:
        agent.act(
            "Read the PDF at '/home/user/report.pdf' on this machine and summarize "
            "its first page."
        )


class LoadInvoiceTool(Tool):
    """Custom tool that hands a fixed invoice PDF to the model.

    Returning a `PdfSource` from a tool mirrors returning a `PIL.Image.Image`: the PDF
    is rendered as a document block in the tool result. Pass a `Path` (or raw bytes) -
    a plain `str` is interpreted as PDF bytes, not a file path.
    """

    def __init__(self, pdf_path: Path) -> None:
        super().__init__(
            name="load_invoice",
            description="Loads the current invoice PDF for analysis.",
            input_schema={"type": "object", "properties": {}},
        )
        self._pdf_path = pdf_path

    def __call__(self) -> PdfSource:
        return PdfSource(self._pdf_path)


def custom_pdf_returning_tool() -> None:
    """Use a custom tool that returns a `PdfSource`."""
    if not PDF_PATH.exists():
        logger.warning("No PDF at %s - skipping custom_pdf_returning_tool()", PDF_PATH)
        return

    with ComputerAgent(act_tools=[LoadInvoiceTool(pdf_path=PDF_PATH)]) as agent:
        agent.act("Load the invoice and tell me its total amount.")


if __name__ == "__main__":
    # Pick the scenario you want to try.
    extract_from_pdf_file()
    # load_pdf_during_act()
    # read_pdf_from_target_machine()
    # custom_pdf_returning_tool()

    logger.info("Done!")
