"""OpenAIGetModel — GetModel for any OpenAI-compatible API."""

import json
import logging
from typing import TYPE_CHECKING, Any

import openai
from openai import OpenAI
from typing_extensions import override

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

from askui.models.exceptions import QueryNoResponseError
from askui.models.models import GetModel, GetSettings
from askui.models.shared.prompts import GetSystemPrompt
from askui.models.types.response_schemas import ResponseSchema, to_response_schema
from askui.prompts.get_prompts import SYSTEM_PROMPT_GET
from askui.utils.excel_utils import OfficeDocumentSource
from askui.utils.pdf_utils import PdfSource
from askui.utils.source_utils import Source

logger = logging.getLogger(__name__)


def _clean_schema_refs(schema: dict[str, Any] | list[Any]) -> None:
    """Remove ``title`` fields next to ``$ref`` fields (unsupported by OpenAI)."""
    if isinstance(schema, dict):
        if "$ref" in schema and "title" in schema:
            del schema["title"]
        for value in schema.values():
            if isinstance(value, (dict, list)):
                _clean_schema_refs(value)
    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, (dict, list)):
                _clean_schema_refs(item)


class OpenAIGetModel(GetModel):
    """GetModel implementation for any OpenAI-compatible API.

    Args:
        model_id (str): The model name to use.
        client (`OpenAI`): A pre-configured OpenAI client.

    Example:
        ```python
        from openai import OpenAI
        from askui.models.openai.get_model import OpenAIGetModel

        client = OpenAI(api_key="sk-...")
        model = OpenAIGetModel(model_id="gpt-4o", client=client)
        ```
    """

    def __init__(
        self,
        model_id: str,
        client: OpenAI,
    ) -> None:
        self._model_id = model_id
        self._client = client

    def _predict(
        self,
        source_part: dict[str, Any],
        instruction: str,
        prompt: GetSystemPrompt,
        response_schema: type[ResponseSchema] | None,
    ) -> str | None | ResponseSchema:
        _response_schema = (
            to_response_schema(response_schema) if response_schema else None
        )

        response_format: openai.NotGiven | dict[str, Any] = openai.NOT_GIVEN
        if _response_schema is not None:
            schema = _response_schema.model_json_schema()
            _clean_schema_refs(schema)

            defs = schema.pop("$defs", None)
            schema_response_wrapper: dict[str, Any] = {
                "type": "object",
                "properties": {"response": schema},
                "additionalProperties": False,
                "required": ["response"],
            }
            if defs:
                schema_response_wrapper["$defs"] = defs
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "user_json_schema",
                    "schema": schema_response_wrapper,
                    "strict": True,
                },
            }

        chat_completion: ChatCompletion = self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self._model_id,
            response_format=response_format,
            messages=[
                {
                    "role": "user",
                    "content": [
                        source_part,
                        {"type": "text", "text": str(prompt) + instruction},
                    ],
                }
            ],
            stream=False,
            timeout=300.0,
        )

        model_response = chat_completion.choices[0].message.content

        if _response_schema is not None and model_response is not None:
            try:
                response_json = json.loads(model_response)
            except json.JSONDecodeError:
                error_msg = (
                    f"Expected JSON, but model {self._model_id} "
                    f"returned: {model_response}"
                )
                logger.exception(
                    "Expected JSON, but model returned",
                    extra={"model": self._model_id, "response": model_response},
                )
                raise ValueError(error_msg) from None

            validated_response = _response_schema.model_validate(
                response_json["response"]
            )
            return validated_response.root

        return model_response

    @override
    def get(
        self,
        query: str,
        source: Source,
        response_schema: type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        if isinstance(source, OfficeDocumentSource):
            err_msg = (
                "Office Document processing is not supported"
                " for OpenAI-compatible models"
            )
            raise NotImplementedError(err_msg)

        system_prompt = get_settings.system_prompt or SYSTEM_PROMPT_GET

        if isinstance(source, PdfSource):
            source_part: dict[str, Any] = {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": source.to_data_url(),
                },
            }
        else:
            source_part = {
                "type": "image_url",
                "image_url": {"url": source.to_data_url()},
            }

        response = self._predict(
            source_part=source_part,
            instruction=query,
            prompt=system_prompt,
            response_schema=response_schema,
        )
        if response is None:
            error_msg = f'No response from model "{self._model_id}" to query: "{query}"'
            raise QueryNoResponseError(error_msg, query)
        return response
