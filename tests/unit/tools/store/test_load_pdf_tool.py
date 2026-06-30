"""Tests for `LoadPdfTool`.

The tool loads a PDF from a base directory and returns it as a `PdfSource`, the
PDF counterpart to `LoadImageTool`. When the tool result is converted into
content blocks the `PdfSource` becomes a `document` block, so the model receives
the PDF in full.
"""

import base64
from pathlib import Path

import pytest

from askui.models.shared.agent_message_param import (
    Base64PdfSourceParam,
    DocumentBlockParam,
)
from askui.models.shared.tools import _convert_to_content
from askui.tools.store.universal import LoadPdfTool
from askui.utils.pdf_utils import PdfSource

_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


class TestLoadPdfTool:
    def test_loads_pdf_relative_to_base_dir(self, tmp_path: Path) -> None:
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "q4.pdf").write_bytes(_PDF_BYTES)
        tool = LoadPdfTool(base_dir=tmp_path)

        message, source = tool(pdf_path="reports/q4.pdf")

        assert isinstance(source, PdfSource)
        assert source.to_bytes() == _PDF_BYTES
        assert "reports/q4.pdf" in message.replace("\\", "/")

    def test_result_converts_to_document_block(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").write_bytes(_PDF_BYTES)
        tool = LoadPdfTool(base_dir=tmp_path)

        blocks = _convert_to_content(tool(pdf_path="doc.pdf"))

        # tuple result -> confirmation text block + document block
        document = next(b for b in blocks if isinstance(b, DocumentBlockParam))
        assert isinstance(document.source, Base64PdfSourceParam)
        assert base64.b64decode(document.source.data) == _PDF_BYTES

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = LoadPdfTool(base_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            tool(pdf_path="nope.pdf")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        tool = LoadPdfTool(base_dir=tmp_path)

        with pytest.raises(IsADirectoryError):
            tool(pdf_path="sub")

    def test_is_cacheable(self, tmp_path: Path) -> None:
        assert LoadPdfTool(base_dir=tmp_path).is_cacheable is True
