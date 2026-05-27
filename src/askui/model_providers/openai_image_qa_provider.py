"""OpenAIImageQAProvider — image Q&A via any OpenAI-compatible API."""

from functools import cached_property

from openai import OpenAI
from typing_extensions import override

from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.models.openai.get_model import OpenAIGetModel
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source


class OpenAIImageQAProvider(ImageQAProvider):
    """Image Q&A provider for any OpenAI-compatible API.

    Works with OpenAI, Ollama, vLLM, LM Studio, Together AI, and any
    other service that exposes an OpenAI-compatible ``/v1/chat/completions``
    endpoint.

    Args:
        model_id (str): Model name to use.
        api_key (str | None, optional): API key. Reads ``OPENAI_API_KEY``
            from the environment if not provided.
        base_url (str | None, optional): Base URL for the API. Defaults
            to the OpenAI API (``https://api.openai.com/v1``).
        client (`OpenAI` | None, optional): Pre-configured OpenAI client.
            If provided, ``api_key`` and ``base_url`` are ignored.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import OpenAIImageQAProvider

        agent = ComputerAgent(settings=AgentSettings(
            image_qa_provider=OpenAIImageQAProvider(
                model_id="gpt-4o",
                api_key="sk-...",
            )
        ))
        ```
    """

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model_id = model_id
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @cached_property
    def _get_model(self) -> OpenAIGetModel:
        """Lazily initialise the `OpenAIGetModel` on first use."""
        return OpenAIGetModel(model_id=self._model_id, client=self._client)

    @override
    def query(
        self,
        query: str,
        source: Source,
        response_schema: type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        result: ResponseSchema | str = self._get_model.get(
            query=query,
            source=source,
            response_schema=response_schema,
            get_settings=get_settings,
        )
        return result
