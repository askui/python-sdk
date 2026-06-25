"""Unit tests for `AnthropicGetModel`.

Claude processes PDFs server-side (each page as text + image), so `get()` sends
a PDF as a base64 `document` block rather than rasterising it client-side.
Images keep the existing resize-and-image-block path; Office documents remain
unsupported.
"""

import base64
from unittest.mock import MagicMock

import pytest
from PIL import Image

from askui.models.anthropic.get_model import AnthropicGetModel
from askui.models.shared.agent_message_param import (
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.settings import GetSettings
from askui.utils.excel_utils import OfficeDocumentSource
from askui.utils.image_utils import ImageSource
from askui.utils.pdf_utils import PdfSource

_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


def _model_returning(text: str) -> tuple[AnthropicGetModel, MagicMock]:
    messages_api = MagicMock(spec=MessagesApi)
    messages_api.create_message.return_value = MessageParam(
        role="assistant", content=[TextBlockParam(text=text)]
    )
    model = AnthropicGetModel(model_id="claude-sonnet-4-6", messages_api=messages_api)
    return model, messages_api


class TestAnthropicGetModel:
    def test_pdf_source_sends_document_block(self) -> None:
        model, messages_api = _model_returning("42 pages")

        result = model.get(
            query="How many pages?",
            source=PdfSource(_PDF_BYTES),
            response_schema=None,
            get_settings=GetSettings(),
        )

        assert result == "42 pages"
        blocks = messages_api.create_message.call_args.kwargs["messages"][0].content
        assert isinstance(blocks[0], DocumentBlockParam)
        assert blocks[0].source.media_type == "application/pdf"
        assert base64.b64decode(blocks[0].source.data) == _PDF_BYTES
        assert isinstance(blocks[1], TextBlockParam)
        assert blocks[1].text == "How many pages?"

    def test_image_source_still_sends_image_block(self) -> None:
        model, messages_api = _model_returning("a submit button")

        result = model.get(
            query="What is shown?",
            source=ImageSource(Image.new("RGB", (10, 10))),
            response_schema=None,
            get_settings=GetSettings(),
        )

        assert result == "a submit button"
        blocks = messages_api.create_message.call_args.kwargs["messages"][0].content
        assert isinstance(blocks[0], ImageBlockParam)

    def test_office_document_remains_unsupported(self) -> None:
        model, _ = _model_returning("unused")

        with pytest.raises(NotImplementedError, match="Office Document"):
            model.get(
                query="Describe",
                source=MagicMock(spec=OfficeDocumentSource),
                response_schema=None,
                get_settings=GetSettings(),
            )
