"""AskUIVlmProvider — VLM access via AskUI's hosted model proxies."""

import os
from enum import Enum
from functools import cached_property
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.askui.inference_api_settings import AskUiInferenceApiSettings
from askui.models.openai.messages_api import OpenAIMessagesApi
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.coordinate_space import (
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
    VlmCoordinateSpace,
)
from askui.models.shared.image_scaler import ImageScaler, PatchOptimizedImageScaler
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection

_DEFAULT_MODEL_ID = "claude-sonnet-4-6"
_DEFAULT_MAX_IMAGE_EDGE = 1024
# Claude emits native pixel coordinates; Gemini emits coordinates in a
# 1000x1000 normalised grid.
_ANTHROPIC_COORDINATE_SPACE = PixelCoordinateSpace()
_GOOGLE_COORDINATE_SPACE = ScaledCoordinateSpace(width=1000, height=1000)


class _Backend(Enum):
    """The AskUI proxy backend a model is served through."""

    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"


def _infer_backend(model_id: str) -> _Backend:
    """Infer the AskUI proxy backend that serves ``model_id``.

    Claude models route to the Anthropic-compatible proxy; Gemini models (with
    or without a ``google/`` vendor prefix) route to the OpenAI-compatible proxy.

    Raises:
        ValueError: If no backend can be inferred from ``model_id``.
    """
    normalized = model_id.lower()
    if "claude" in normalized:
        return _Backend.ANTHROPIC
    if "gemini" in normalized:
        return _Backend.GOOGLE
    error_msg = (
        f"Cannot infer a backend for model id {model_id!r}. Expected the model "
        f"id to reference a Claude or Gemini model."
    )
    raise ValueError(error_msg)


class AskUIVlmProvider(VlmProvider):
    """VLM provider that routes requests through AskUI's hosted model proxies.

    The proxy used is selected from `model_id`:

    - Anthropic (Claude) models are served via the Anthropic-compatible proxy
      (``/proxy/anthropic``) using the `AnthropicMessagesApi`.
    - OpenAI-compatible models (e.g. Gemini) are served via the OpenAI-compatible
      proxy (``/proxy/openai/v1/chat/completions``) using the `OpenAIMessagesApi`.

    The backend is inferred from the model-id prefix (see `_infer_backend`); a
    `ValueError` is raised when it cannot be determined.

    Credentials are read from environment variables (`ASKUI_WORKSPACE_ID`,
    `ASKUI_TOKEN`) lazily — validation happens on the first API call, not at
    construction time.

    Args:
        askui_settings (`AskUiInferenceApiSettings` | None, optional):
            Connection settings (workspace ID, token, base URL).  Reads
            from environment variables if not provided.
        model_id (str | None, optional): Model to use. Defaults to
            ``"claude-sonnet-4-6"``.  Pass a Gemini model id (e.g.
            ``"gemini-3.5-pro"``) to route through the OpenAI-compatible proxy.
        client (`Anthropic` | `OpenAI` | None, optional): Pre-configured client.
            Pass an `Anthropic` client for Claude models or an `OpenAI` client
            for Gemini models. It is used only when it matches the proxy the
            configured ``model_id`` routes to; otherwise a client is built from
            ``askui_settings``.
        image_scaler (`ImageScaler` | None, optional): Custom image preprocessing
            callable. If ``None``, uses Anthropic-optimized patch-based scaling
            controlled by ``image_edge_max``.
        image_edge_max (int | None, optional): Maximum edge length (in pixels)
            for screenshots sent to the model.  Only used when ``image_scaler``
            is not provided.  Reads ``ASKUI_VLM_MAX_IMAGE_EDGE`` from the
            environment if not provided.  Defaults to 1024.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AskUIVlmProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=AskUIVlmProvider(
                model_id="claude-opus-4-6-20260401",
            )
        ))
        ```
    """

    def __init__(
        self,
        askui_settings: AskUiInferenceApiSettings | None = None,
        model_id: str | None = None,
        client: Anthropic | OpenAI | None = None,
        image_scaler: ImageScaler | None = None,
        image_edge_max: int | None = None,
    ) -> None:
        self._askui_settings = askui_settings or AskUiInferenceApiSettings()
        self._model_id_value = (
            model_id or os.environ.get("VLM_PROVIDER_MODEL_ID") or _DEFAULT_MODEL_ID
        )
        self._client = client
        resolved_edge_max = (
            image_edge_max
            or int(os.environ.get("ASKUI_VLM_MAX_IMAGE_EDGE", "0"))
            or _DEFAULT_MAX_IMAGE_EDGE
        )
        self._image_scaler = image_scaler or PatchOptimizedImageScaler(
            max_edge=resolved_edge_max
        )

    @property
    @override
    def model_id(self) -> str:
        return self._model_id_value

    @property
    @override
    def image_scaler(self) -> ImageScaler:
        return self._image_scaler

    @property
    @override
    def coordinate_space(self) -> VlmCoordinateSpace:
        """The coordinate grid the configured model emits coordinates in.

        Gemini (OpenAI proxy) emits coordinates in a 1000x1000 normalised grid;
        Claude emits native pixel coordinates.
        """
        if self._backend is _Backend.GOOGLE:
            return _GOOGLE_COORDINATE_SPACE
        return _ANTHROPIC_COORDINATE_SPACE

    @cached_property
    def _backend(self) -> _Backend:
        return _infer_backend(self._model_id_value)

    @cached_property
    def _messages_api(self) -> MessagesApi:
        """Lazily initialise the `MessagesApi` matching the configured model."""
        if self._backend is _Backend.OPENAI or self._backend is _Backend.GOOGLE:
            return self._build_openai_messages_api()
        return self._build_anthropic_messages_api()

    def _build_anthropic_messages_api(self) -> AnthropicMessagesApi:
        if isinstance(self._client, Anthropic):
            return AnthropicMessagesApi(client=self._client)

        # TODO askui_settings.verify_ssl are not considered! #noqa
        # if self._askui_settings.verify_ssl:
        # ...
        # http_client = ...
        client = Anthropic(
            api_key="DummyValueRequiredByAnthropicClient",
            base_url=f"{self._askui_settings.base_url}/proxy/anthropic",
            default_headers={
                "Authorization": self._askui_settings.authorization_header
            },
        )
        return AnthropicMessagesApi(client=client)

    def _build_openai_messages_api(self) -> OpenAIMessagesApi:
        if isinstance(self._client, OpenAI):
            return OpenAIMessagesApi(client=self._client)

        client = OpenAI(
            api_key="DummyValueRequiredByOpenAIClient",
            base_url=f"{self._askui_settings.base_url}/proxy/openai/v1",
            default_headers={
                "Authorization": self._askui_settings.authorization_header
            },
        )
        return OpenAIMessagesApi(client=client)

    @override
    def augment_system_prompt(self, system: SystemPrompt) -> SystemPrompt:
        """Append coordinate info to the system prompt for OpenAI-proxy models.

        Claude emits pixel coordinates natively, so the prompt is returned
        unchanged. Models routed through the OpenAI proxy (e.g. Gemini) are told
        which coordinate grid to emit so their output can be mapped back via
        `coordinate_space`.
        """
        if self._backend is not _Backend.GOOGLE:
            return system
        coord_info = self.coordinate_space.build_prompt_section()
        return SystemPrompt(prompt=f"{str(system)}\n\n{coord_info}")

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        max_tokens: int | None = None,
        system: SystemPrompt | None = None,
        thinking: ThinkingConfigParam | None = None,
        tool_choice: ToolChoiceParam | None = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        if system is not None:
            system = self.augment_system_prompt(system)
        result: MessageParam = self._messages_api.create_message(
            messages=messages,
            model_id=self._model_id_value,
            tools=tools,
            max_tokens=max_tokens,
            system=system,
            thinking=thinking,
            tool_choice=tool_choice,
            temperature=temperature,
            provider_options=provider_options,
        )
        return result
