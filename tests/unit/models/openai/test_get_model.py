"""Unit tests for OpenAIGetModel."""

from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from askui.models.exceptions import QueryNoResponseError
from askui.models.openai.get_model import OpenAIGetModel
from askui.models.shared.settings import GetSettings
from askui.utils.excel_utils import OfficeDocumentSource
from askui.utils.image_utils import ImageSource
from askui.utils.pdf_utils import PdfSource


def _make_completion(content: str | None) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=1234567890,
        model="qwen2.5vl",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


class TestOpenAIGetModel:
    def test_basic_query_returns_string(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            "The button says Submit"
        )

        source = MagicMock(spec=ImageSource)
        source.to_data_url.return_value = "data:image/png;base64,abc"

        model = OpenAIGetModel(model_id="qwen2.5vl", client=mock_client)
        result = model.get(
            query="What does the button say?",
            source=source,
            response_schema=None,
            get_settings=GetSettings(),
        )

        assert result == "The button says Submit"
        mock_client.chat.completions.create.assert_called_once()

    def test_no_response_raises_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(None)

        source = MagicMock(spec=ImageSource)
        source.to_data_url.return_value = "data:image/png;base64,abc"

        model = OpenAIGetModel(model_id="qwen2.5vl", client=mock_client)
        with pytest.raises(QueryNoResponseError):
            model.get(
                query="Describe",
                source=source,
                response_schema=None,
                get_settings=GetSettings(),
            )

    def test_pdf_source_sends_file_part(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion("3 pages")

        source = MagicMock(spec=PdfSource)
        source.to_data_url.return_value = "data:application/pdf;base64,abc"
        source.filename = "report.pdf"

        model = OpenAIGetModel(model_id="gpt-4o", client=mock_client)
        result = model.get(
            query="How many pages?",
            source=source,
            response_schema=None,
            get_settings=GetSettings(),
        )

        assert result == "3 pages"
        content = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        file_part = next(part for part in content if part["type"] == "file")
        assert file_part["file"]["file_data"] == "data:application/pdf;base64,abc"
        # The PDF's real file name is forwarded to OpenAI's ``file`` part.
        assert file_part["file"]["filename"] == "report.pdf"

    def test_office_document_source_not_supported(self) -> None:
        mock_client = MagicMock()
        source = MagicMock(spec=OfficeDocumentSource)

        model = OpenAIGetModel(model_id="qwen2.5vl", client=mock_client)
        with pytest.raises(NotImplementedError, match="Office Document"):
            model.get(
                query="Describe",
                source=source,
                response_schema=None,
                get_settings=GetSettings(),
            )
