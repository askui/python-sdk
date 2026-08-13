"""Tests for `AskUiControllerClient._decode_file_payload`.

`get_file` decodes a Base64 payload from the controller and dispatches by the
detected MIME type (via ``filetype.guess``) rather than by trying to parse it as
each type in turn: images become `PIL.Image.Image`, PDFs become `PdfSource`,
and anything that decodes cleanly as UTF-8 becomes a string.
"""

import base64
import io

import pytest
from PIL import Image

from askui.tools.askui.askui_controller import (
    AskUiControllerClient,
    DesktopAgentOsException,
)
from askui.utils.pdf_utils import PdfSource

_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return _b64(buffer.getvalue())


class TestDecodeFilePayload:
    def test_decodes_image(self) -> None:
        result = AskUiControllerClient._decode_file_payload(_png_b64())
        assert isinstance(result, Image.Image)
        assert result.format == "PNG"

    def test_decodes_pdf(self) -> None:
        result = AskUiControllerClient._decode_file_payload(_b64(_PDF_BYTES))
        assert isinstance(result, PdfSource)
        assert result.to_bytes() == _PDF_BYTES

    def test_decodes_utf8_text(self) -> None:
        result = AskUiControllerClient._decode_file_payload(_b64(b"hello world"))
        assert result == "hello world"

    def test_rejects_unsupported_binary(self) -> None:
        with pytest.raises(DesktopAgentOsException):
            AskUiControllerClient._decode_file_payload(_b64(b"\x00\x01\x02\x03"))
