"""Tests for converting tool-call results into provider-neutral content blocks.

Tools may return a `PdfSource` to hand a PDF document to the model, mirroring
how returning a `PIL.Image` produces an image block. Anthropic accepts
`document` blocks inside `tool_result` content (base64 PDF, no beta header), so
a returned PDF is converted into a `DocumentBlockParam`.
"""

import base64
from pathlib import Path
from typing import cast

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    TextResourceContents,
)
from PIL import Image

from askui.models.anthropic.messages_api import from_content_block
from askui.models.shared.agent_message_param import (
    Base64PdfSourceParam,
    DocumentBlockParam,
    ImageBlockParam,
    TextBlockParam,
    ToolResultBlockParam,
)
from askui.models.shared.tools import (
    McpToolAdapterException,
    _convert_call_tool_result,
    _convert_from_mcp_tool_call_result,
    _convert_mcp_resource,
    _convert_to_content,
    _convert_to_mcp_content,
)
from askui.utils.pdf_utils import MAX_PDF_SIZE_BYTES, PdfSource

# Smallest payload that is unambiguously a PDF; we only base64-encode the bytes,
# never parse them.
_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


class TestConvertPdfToolResult:
    def test_pdf_source_becomes_document_block(self) -> None:
        result = _convert_to_content(PdfSource(_PDF_BYTES))

        assert len(result) == 1
        block = result[0]
        assert isinstance(block, DocumentBlockParam)
        assert isinstance(block.source, Base64PdfSourceParam)
        assert block.source.media_type == "application/pdf"
        assert base64.b64decode(block.source.data) == _PDF_BYTES

    def test_pdf_alongside_text_and_image_preserves_order(self) -> None:
        result = _convert_to_content(
            [
                "see the attached report",
                PdfSource(_PDF_BYTES),
                Image.new("RGB", (2, 2)),
            ]
        )

        expected_types: list[type] = [
            TextBlockParam,
            DocumentBlockParam,
            ImageBlockParam,
        ]
        assert [type(block) for block in result] == expected_types

    def test_pdf_source_from_path_sets_title(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(_PDF_BYTES)

        result = _convert_to_content(PdfSource(pdf))

        block = result[0]
        assert isinstance(block, DocumentBlockParam)
        assert block.title == "report.pdf"

    def test_pdf_source_from_bytes_has_no_title(self) -> None:
        result = _convert_to_content(PdfSource(_PDF_BYTES))

        block = result[0]
        assert isinstance(block, DocumentBlockParam)
        assert block.title is None

    def test_document_in_tool_result_serializes_for_anthropic(self) -> None:
        block = ToolResultBlockParam(
            tool_use_id="toolu_1",
            content=[
                DocumentBlockParam(source=Base64PdfSourceParam(data="cGRm")),
            ],
        )

        dumped = cast("dict", from_content_block(block))

        document = dumped["content"][0]
        assert document["type"] == "document"
        assert document["source"] == {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "cGRm",
        }


class TestConvertMcpResource:
    """An MCP tool returns a PDF as an embedded blob resource, not an image."""

    def test_pdf_blob_resource_becomes_document_block(self) -> None:
        resource = BlobResourceContents(
            uri="file:///doc.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(_PDF_BYTES).decode(),
        )

        block = _convert_mcp_resource(resource)

        assert isinstance(block, DocumentBlockParam)
        assert base64.b64decode(block.source.data) == _PDF_BYTES

    def test_text_resource_becomes_text_block(self) -> None:
        resource = TextResourceContents(uri="file:///a.txt", text="hello")

        block = _convert_mcp_resource(resource)

        assert isinstance(block, TextBlockParam)
        assert block.text == "hello"

    def test_unsupported_blob_resource_is_dropped(self) -> None:
        resource = BlobResourceContents(
            uri="file:///a.bin",
            mimeType="application/octet-stream",
            blob="QUJD",
        )

        assert _convert_mcp_resource(resource) is None

    def test_embedded_pdf_resource_in_call_tool_result(self) -> None:
        result = CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///doc.pdf",
                        mimeType="application/pdf",
                        blob=base64.b64encode(_PDF_BYTES).decode(),
                    ),
                )
            ],
            structured_content=None,
            meta=None,
        )

        blocks = _convert_call_tool_result(result)

        assert len(blocks) == 1
        assert isinstance(blocks[0], DocumentBlockParam)
        assert base64.b64decode(blocks[0].source.data) == _PDF_BYTES

    def test_oversized_pdf_resource_is_dropped(self) -> None:
        # ``blob`` length implies a decoded size above the limit, so the
        # resource is dropped instead of being forwarded to the provider.
        oversized_blob = "A" * ((MAX_PDF_SIZE_BYTES + 1024) * 4 // 3)
        resource = BlobResourceContents(
            uri="file:///big.pdf",
            mimeType="application/pdf",
            blob=oversized_blob,
        )

        assert _convert_mcp_resource(resource) is None


class TestMcpPdfRoundTrip:
    """PDFs survive the outbound (`to_mcp`) and inbound (`from_mcp`) MCP paths."""

    def test_pdf_source_serialized_to_embedded_resource(self) -> None:
        converted = _convert_to_mcp_content(PdfSource(_PDF_BYTES))

        assert isinstance(converted, EmbeddedResource)
        assert isinstance(converted.resource, BlobResourceContents)
        assert converted.resource.mimeType == "application/pdf"
        assert base64.b64decode(converted.resource.blob) == _PDF_BYTES

    def test_embedded_pdf_resource_becomes_pdf_source(self) -> None:
        resource = EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///doc.pdf",
                mimeType="application/pdf",
                blob=base64.b64encode(_PDF_BYTES).decode(),
            ),
        )

        result = _convert_from_mcp_tool_call_result("tool", resource)

        assert isinstance(result, PdfSource)
        assert result.to_bytes() == _PDF_BYTES

    def test_text_embedded_resource_becomes_string(self) -> None:
        resource = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(uri="file:///a.txt", text="hello"),
        )

        assert _convert_from_mcp_tool_call_result("tool", resource) == "hello"

    def test_unsupported_embedded_resource_raises(self) -> None:
        resource = EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///a.bin",
                mimeType="application/octet-stream",
                blob="QUJD",
            ),
        )

        with pytest.raises(McpToolAdapterException):
            _convert_from_mcp_tool_call_result("tool", resource)
