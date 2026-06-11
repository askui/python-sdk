"""AskUIVlmProvider — VLM access via AskUI's hosted Anthropic proxy."""

import os
from functools import cached_property
from typing import Any

from anthropic import Anthropic
from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.askui.inference_api_settings import AskUiInferenceApiSettings
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.image_scaler import ImageScaler
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.llm_image_utils import compute_patch_optimized_image

_DEFAULT_MODEL_ID = "claude-sonnet-4-6"
_DEFAULT_MAX_IMAGE_EDGE = 1568


class AskUIVlmProvider(VlmProvider):
    """VLM provider that routes requests through AskUI's hosted Anthropic proxy.

    Supports Claude 4.x generation models. Credentials are read from environment
    variables (`ASKUI_WORKSPACE_ID`, `ASKUI_TOKEN`) lazily — validation happens
    on the first API call, not at construction time.

    Args:
        askui_settings (`AskUiInferenceApiSettings` | None, optional):
            Connection settings (workspace ID, token, base URL).  Reads
            from environment variables if not provided.
        model_id (str | None, optional): Claude model to use. Defaults to
            ``"claude-sonnet-4-6"``.
        client (`Anthropic` | None, optional): Pre-configured Anthropic client.
            If provided, ``askui_settings`` is only used for the base URL.
        image_scaler (`ImageScaler` | None, optional): Custom image preprocessing
            callable. If ``None``, uses Anthropic-optimized patch-based scaling.
        max_image_edge (int | None, optional): Maximum edge length (in pixels)
            for screenshots sent to the model.  Reads ``ASKUI_VLM_MAX_IMAGE_EDGE``
            from the environment if not provided.  Defaults to 1568.

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
        client: Anthropic | None = None,
        image_scaler: ImageScaler | None = None,
        max_image_edge: int | None = None,
    ) -> None:
        self._askui_settings = askui_settings or AskUiInferenceApiSettings()
        self._model_id_value = (
            model_id or os.environ.get("VLM_PROVIDER_MODEL_ID") or _DEFAULT_MODEL_ID
        )
        self._injected_client = client
        self._image_scaler_override = image_scaler
        self._max_edge = (
            max_image_edge
            or int(os.environ.get("ASKUI_VLM_MAX_IMAGE_EDGE", "0"))
            or _DEFAULT_MAX_IMAGE_EDGE
        )

    @property
    @override
    def model_id(self) -> str:
        return self._model_id_value

    @property
    @override
    def image_scaler(self) -> ImageScaler:
        if self._image_scaler_override is not None:
            return self._image_scaler_override
        max_edge = self._max_edge
        return lambda image: compute_patch_optimized_image(image, max_edge=max_edge)

    @cached_property
    def _messages_api(self) -> AnthropicMessagesApi:
        """Lazily initialise the AnthropicMessagesApi on first use."""
        if self._injected_client is not None:
            return AnthropicMessagesApi(client=self._injected_client)

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
