"""Unit tests for `PdfSource` size guarding and metadata."""

from pathlib import Path

import pytest

from askui.utils.pdf_utils import (
    MAX_PDF_SIZE_BYTES,
    PdfSource,
    PdfTooLargeError,
)

_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


class TestSizeBytes:
    def test_size_from_bytes(self) -> None:
        assert PdfSource(_PDF_BYTES).size_bytes == len(_PDF_BYTES)

    def test_size_from_path(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(_PDF_BYTES)
        assert PdfSource(pdf).size_bytes == len(_PDF_BYTES)


class TestValidateSize:
    def test_within_limit_passes(self) -> None:
        PdfSource(_PDF_BYTES).validate_size()  # does not raise

    def test_over_limit_raises(self) -> None:
        oversized = PdfSource(b"x" * (MAX_PDF_SIZE_BYTES + 1))
        with pytest.raises(PdfTooLargeError):
            oversized.validate_size()

    def test_custom_limit(self) -> None:
        with pytest.raises(PdfTooLargeError):
            PdfSource(_PDF_BYTES).validate_size(max_size_bytes=1)

    def test_to_base64_enforces_limit(self) -> None:
        oversized = PdfSource(b"x" * (MAX_PDF_SIZE_BYTES + 1))
        with pytest.raises(PdfTooLargeError):
            oversized.to_base64()

    def test_error_reports_sizes(self) -> None:
        oversized = PdfSource(b"x" * (MAX_PDF_SIZE_BYTES + 1))
        with pytest.raises(PdfTooLargeError) as exc_info:
            oversized.validate_size()
        assert exc_info.value.size_bytes == MAX_PDF_SIZE_BYTES + 1
        assert exc_info.value.max_size_bytes == MAX_PDF_SIZE_BYTES


class TestFilename:
    def test_filename_from_path(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(_PDF_BYTES)
        assert PdfSource(pdf).filename == "report.pdf"

    def test_filename_none_for_bytes(self) -> None:
        assert PdfSource(_PDF_BYTES).filename is None
