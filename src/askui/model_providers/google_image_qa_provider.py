"""GoogleImageQAProvider — image Q&A via Google Gemini API."""

from functools import cached_property
from typing import Type

import google.genai as genai
import google.genai.types as genai_types
from google.genai import Client as GenaiClient
from typing_extensions import override

from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.models.google.get_model import GoogleGetModel
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source

_DEFAULT_MODEL_ID = "gemini-3.5-flash"


class GoogleImageQAProvider(ImageQAProvider):
    """Image Q&A provider that routes requests directly to the Google Gemini API.

    Supports multimodal Q&A and structured output extraction from images and
    PDFs. The API key is read from the `GOOGLE_API_KEY` environment variable
    lazily — validation happens on the first API call, not at construction time.

    Args:
        api_key (str | None, optional): Google Gemini API key. Reads
            `GOOGLE_API_KEY` from the environment if not provided.
        base_url (str | None, optional): Base URL for the Google GenAI API.
            Useful for proxies or custom endpoints.
        auth_token (str | None, optional): Authorization token for custom
            authentication. Added as an `Authorization` header.
        model_id (str, optional): Gemini model to use. Defaults to
            `\"gemini-3.5-flash\"`.
        client (GenaiClient | None, optional): Pre-configured Google GenAI client.
            If provided, other connection parameters are ignored.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import GoogleImageQAProvider

        agent = ComputerAgent(settings=AgentSettings(
            image_qa_provider=GoogleImageQAProvider(
                api_key=\"AIza...\",
                model_id=\"gemini-3.5-flash\",
            )
        ))
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        auth_token: str | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
        client: GenaiClient | None = None,
    ) -> None:
        self._model_id = model_id
        if client is not None:
            self.client = client
        else:
            http_options = None
            if base_url is not None or auth_token is not None:
                headers = {"Authorization": auth_token} if auth_token else None
                http_options = genai_types.HttpOptions(
                    base_url=base_url,
                    headers=headers,
                )
            self.client = genai.Client(api_key=api_key, http_options=http_options)

    @cached_property
    def _get_model(self) -> GoogleGetModel:
        return GoogleGetModel(model_id=self._model_id, client=self.client)

    @override
    def query(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        result: ResponseSchema | str = self._get_model.get(
            query=query,
            source=source,
            response_schema=response_schema,
            get_settings=get_settings,
        )
        return result
