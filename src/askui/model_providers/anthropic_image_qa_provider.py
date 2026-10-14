"""AnthropicImageQAProvider — image Q&A via direct Anthropic API."""

from functools import cached_property
from typing import Type

from anthropic import Anthropic
from typing_extensions import override

from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.models.anthropic.get_model import AnthropicGetModel
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source

_DEFAULT_MODEL_ID = "claude-haiku-5-5"


class AnthropicImageQAProvider(ImageQAProvider):
    """Image Q&A provider that routes requests directly to the Anthropic API.

    Supports structured output extraction from images using Claude models.
    The API key is read from the `ANTHROPIC_API_KEY` environment variable
    lazily — validation happens on the first API call, not at construction time.

    Args:
        api_key (str | None, optional): Anthropic API key. Reads
            `ANTHROPIC_API_KEY` from the environment if not provided.
        base_url (str | None, optional): Base URL for the Anthropic API.
            Useful for proxies or custom endpoints.
        auth_token (str | None, optional): Authorization token for custom
            authentication. Added as an `Authorization` header.
        model_id (str, optional): Claude model to use. Defaults to
            `\"claude-haiku-5-5\"`.
        client (Anthropic | None, optional): Pre-configured Anthropic client.
            If provided, other connection parameters are ignored.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AnthropicImageQAProvider

        agent = ComputerAgent(settings=AgentSettings(
            image_qa_provider=AnthropicImageQAProvider(
                api_key=\"sk-ant-...\",
                model_id=\"claude-opus-4-5-20251101\",
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
        client: Anthropic | None = None,
    ) -> None:
        self._model_id = model_id
        if client is not None:
            self.client = client
        else:
            self.client = Anthropic(
                api_key=api_key,
                base_url=base_url,
                auth_token=auth_token,
            )

    @cached_property
    def _get_model(self) -> AnthropicGetModel:
        messages_api = AnthropicMessagesApi(client=self.client)
        return AnthropicGetModel(model_id=self._model_id, messages_api=messages_api)

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
