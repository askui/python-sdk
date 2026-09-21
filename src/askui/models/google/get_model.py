"""GoogleGetModel — GetModel implementation for Google Gemini models."""

import json as json_lib
import logging
from typing import Type

import google.genai as genai
import google.genai.types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from askui.models.exceptions import (
    QueryNoResponseError,
    QueryUnexpectedResponseError,
)
from askui.models.models import GetModel
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema, to_response_schema
from askui.prompts.get_prompts import SYSTEM_PROMPT_GET
from askui.utils.excel_utils import OfficeDocumentSource
from askui.utils.image_utils import ImageSource
from askui.utils.source_utils import Source

logger = logging.getLogger(__name__)

# Maximum File Size for Inline Data PDF
# https://ai.google.dev/gemini-api/docs/file-input-methods
# last checked: February 17, 2026
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


class GoogleGetModel(GetModel):
    """GetModel implementation for Google Gemini models.

    Args:
        model_id (str): The model identifier (e.g., "gemini-3.5-flash").
        client (genai.Client): The Google GenAI client for creating messages.
    """

    def __init__(
        self,
        model_id: str,
        client: genai.Client,
    ) -> None:
        self._model_id = model_id
        self._client = client

    def _source_to_part(self, source: Source) -> genai_types.Part:
        """Convert a Source to a Google GenAI Part.

        Args:
            source (Source): The image, PDF, or office document source.

        Returns:
            genai_types.Part: The corresponding GenAI part.

        Raises:
            ValueError: If the source data exceeds the size limit.
        """
        if isinstance(source, ImageSource):
            data = source.to_bytes()
            if len(data) > _MAX_FILE_SIZE_BYTES:
                limit = _MAX_FILE_SIZE_BYTES
                err_msg = f"Image file size exceeds the limit of {limit} bytes."
                raise ValueError(err_msg)
            return genai_types.Part.from_bytes(data=data, mime_type="image/png")
        if isinstance(source, OfficeDocumentSource):
            with source.reader as r:
                return genai_types.Part.from_text(text=r.read().decode())
        # PDF and other sources
        with source.reader as r:
            data = r.read()
            if len(data) > _MAX_FILE_SIZE_BYTES:
                err_msg = (
                    f"PDF file size exceeds the limit of {_MAX_FILE_SIZE_BYTES} bytes."
                )
                raise ValueError(err_msg)
            return genai_types.Part.from_bytes(data=data, mime_type="application/pdf")

    @override
    def get(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        """Extract information from a source based on a query.

        Args:
            query (str): A description of what information to extract.
            source (Source): The source to analyze (screenshot, image, or PDF).
            response_schema (Type[ResponseSchema] | None): Optional Pydantic model
                class defining the expected response structure.
            get_settings (GetSettings): The settings for this get operation.

        Returns:
            Either a string response or a Pydantic model instance if response_schema
            is provided.

        Raises:
            QueryNoResponseError: If the model returns no response.
            QueryUnexpectedResponseError: If the response cannot be parsed.
            NotImplementedError: If a recursive response schema is provided.
        """
        try:
            _response_schema = to_response_schema(response_schema)
            json_schema = json_lib.dumps(_response_schema.model_json_schema())
            logger.debug(
                "Json schema used for response",
                extra={"json_schema": json_schema},
            )

            content = genai_types.Content(
                parts=[
                    self._source_to_part(source),
                    genai_types.Part.from_text(text=query),
                ],
                role="user",
            )

            system_prompt = (
                str(get_settings.system_prompt)
                if get_settings.system_prompt
                else str(SYSTEM_PROMPT_GET)
            )

            response = self._client.models.generate_content(
                model=f"models/{self._model_id}",
                contents=content,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": _response_schema,
                    "system_instruction": system_prompt,
                },
            )
            json_str = response.text
            if json_str is None:
                raise QueryNoResponseError(
                    message="No response from the model", query=query
                )
            try:
                return _response_schema.model_validate_json(json_str).root
            except ValidationError as e:
                error_message = str(e.errors())
                raise QueryUnexpectedResponseError(
                    message=f"Unexpected response from the model: {error_message}",
                    query=query,
                    response=json_str,
                ) from e
        except RecursionError as e:
            error_message = (
                "Recursive response schemas are not supported by GoogleGetModel"
            )
            raise NotImplementedError(error_message) from e
