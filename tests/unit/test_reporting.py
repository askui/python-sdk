"""Tests for reporting helpers.

`truncate_base64_media` keeps HTML/text reports readable by replacing large
base64 blobs (screenshots, images, PDF documents) with a short placeholder,
rather than dumping the full encoded payload into the report.
"""

from typing import Any

from askui.reporting import truncate_base64_media


def _base64_source(media_type: str) -> dict[str, Any]:
    return {"type": "base64", "media_type": media_type, "data": "QUJD" * 5000}


class TestTruncateBase64Media:
    def test_truncates_image_with_friendly_label(self) -> None:
        result = truncate_base64_media(
            {"type": "image", "source": _base64_source("image/png")}
        )
        assert result["source"]["data"] == "[Base64 image data truncated]"

    def test_truncates_pdf_with_friendly_label(self) -> None:
        result = truncate_base64_media(
            {"type": "document", "source": _base64_source("application/pdf")}
        )
        assert result["source"]["data"] == "[Base64 PDF data truncated]"

    def test_truncates_pdf_inside_tool_result(self) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"type": "text", "text": "PDF loaded"},
                    {"type": "document", "source": _base64_source("application/pdf")},
                ],
            }
        ]

        result = truncate_base64_media(content)

        document = result[0]["content"][1]
        assert document["source"]["data"] == "[Base64 PDF data truncated]"
        # Non-media content is left untouched.
        assert result[0]["content"][0]["text"] == "PDF loaded"

    def test_unknown_media_type_falls_back_to_raw_type(self) -> None:
        result = truncate_base64_media(
            {"type": "input_audio", "source": _base64_source("audio/wav")}
        )
        assert result["source"]["data"] == "[Base64 audio/wav data truncated]"

    def test_leaves_plain_content_untouched(self) -> None:
        assert truncate_base64_media("just a prompt") == "just a prompt"
        assert truncate_base64_media({"type": "text", "text": "hello"}) == {
            "type": "text",
            "text": "hello",
        }
