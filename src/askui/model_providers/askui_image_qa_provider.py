"""AskUIImageQAProvider — image Q&A via AskUI's hosted Gemini proxy."""

from functools import cached_property
from typing import Type

from pydantic import SecretStr
from typing_extensions import override

from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.models.askui.get_model import AskUiGeminiGetModel
from askui.models.askui.inference_api_settings import AskUiInferenceApiSettings
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source

_DEFAULT_MODEL_ID = "gemini-3.5-flash"


class AskUIImageQAProvider(ImageQAProvider):
    """Image Q&A provider that routes requests through AskUI's hosted Gemini proxy.

    Supports multimodal Q&A and structured output extraction from images and
    PDFs. Credentials are read from the ``ASKUI_WORKSPACE_ID`` and
    ``ASKUI_TOKEN`` environment variables lazily — validation happens on the
    first API call, not at construction time.

    Args:
        workspace_id (str | None, optional): AskUI workspace ID. Reads
            `ASKUI_WORKSPACE_ID` from the environment if not provided.
        token (str | None, optional): AskUI API token. Reads `ASKUI_TOKEN`
            from the environment if not provided.
        model_id (str, optional): Gemini model to use. Defaults to
            ``"gemini-3.5-flash"``.
        get_model (AskUiGeminiGetModel | None, optional): Pre-configured get model.
            If provided, `workspace_id` and `token` are ignored.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AskUIImageQAProvider

        agent = ComputerAgent(settings=AgentSettings(
            image_qa_provider=AskUIImageQAProvider(
                model_id="gemini-3.5-flash",
            )
        ))
        ```
    """

    def __init__(
        self,
        workspace_id: str | None = None,
        token: str | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
        get_model: AskUiGeminiGetModel | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._token = token
        self._model_id = model_id
        self._injected_get_model = get_model

    @cached_property
    def _get_model(self) -> AskUiGeminiGetModel:
        """Lazily initialise the AskUiGeminiGetModel on first use."""
        if self._injected_get_model is not None:
            return self._injected_get_model

        settings_kwargs: dict[str, str | SecretStr] = {}
        if self._workspace_id is not None:
            settings_kwargs["workspace_id"] = self._workspace_id
        if self._token is not None:
            settings_kwargs["token"] = SecretStr(self._token)

        inference_api_settings = AskUiInferenceApiSettings(**settings_kwargs)  # type: ignore[arg-type]
        return AskUiGeminiGetModel(
            model_id=self._model_id,
            inference_api_settings=inference_api_settings,
        )

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
