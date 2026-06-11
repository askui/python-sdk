"""OpenAIVlmProvider — VLM access via any OpenAI-compatible API."""

import os
from functools import cached_property
from typing import Any

from openai import OpenAI
from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.openai.messages_api import OpenAIMessagesApi
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.coordinate_space import (
    PixelCoordinateSpace,
    VlmCoordinateSpace,
)
from askui.models.shared.image_scaler import ImageScaler
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.llm_image_utils import compute_patch_optimized_image
from askui.utils.model_pricing import ModelPricing

_DEFAULT_MODEL_ID = "gpt-5.4"
_DEFAULT_COORDINATE_SPACE = PixelCoordinateSpace()
_DEFAULT_MAX_IMAGE_EDGE = 1024


class OpenAIVlmProvider(VlmProvider):
    """VLM provider for any OpenAI-compatible API.

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
        coordinate_space (VlmCoordinateSpace, optional): The coordinate grid
            the model emits coordinates in.  Defaults to the screenshot
            resolution (native pixel coordinates).
        max_image_edge (int | None, optional): Maximum edge length (in pixels)
            for screenshots sent to the model.  Reads ``ASKUI_VLM_MAX_IMAGE_EDGE``
            from the environment if not provided.  Defaults to 2048.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import OpenAIVlmProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=OpenAIVlmProvider(
                model_id="gpt-4o",
                api_key="sk-...",
            )
        ))
        ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        coordinate_space: VlmCoordinateSpace = _DEFAULT_COORDINATE_SPACE,
        image_scaler: ImageScaler | None = None,
        max_image_edge: int | None = None,
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
        cache_write_cost_per_million_tokens: float | None = None,
        cache_read_cost_per_million_tokens: float | None = None,
    ) -> None:
        self._model_id_value = (
            model_id or os.environ.get("VLM_PROVIDER_MODEL_ID") or _DEFAULT_MODEL_ID
        )
        self._coordinate_space = coordinate_space
        self._image_scaler_override = image_scaler
        self._max_edge = (
            max_image_edge
            or int(os.environ.get("ASKUI_VLM_MAX_IMAGE_EDGE", "0"))
            or _DEFAULT_MAX_IMAGE_EDGE
        )
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        self._pricing = ModelPricing.for_model(
            self._model_id_value,
            input_cost_per_million_tokens=input_cost_per_million_tokens,
            output_cost_per_million_tokens=output_cost_per_million_tokens,
            cache_write_cost_per_million_tokens=cache_write_cost_per_million_tokens,
            cache_read_cost_per_million_tokens=cache_read_cost_per_million_tokens,
        )

    @property
    @override
    def model_id(self) -> str:
        return self._model_id_value

    @property
    @override
    def coordinate_space(self) -> VlmCoordinateSpace:
        return self._coordinate_space

    @property
    @override
    def pricing(self) -> ModelPricing | None:
        return self._pricing

    @property
    @override
    def image_scaler(self) -> ImageScaler:
        if self._image_scaler_override is not None:
            return self._image_scaler_override
        max_edge = self._max_edge
        return lambda image: compute_patch_optimized_image(
            image, max_edge=max_edge, max_tokens=1536, patch_size=32
        )

    @cached_property
    def _messages_api(self) -> OpenAIMessagesApi:
        """Lazily initialise the `OpenAIMessagesApi` on first use."""
        return OpenAIMessagesApi(client=self._client)

    @override
    def augment_system_prompt(self, system: SystemPrompt) -> SystemPrompt:
        """Append coordinate and resolution info to the system prompt."""
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
        return self._messages_api.create_message(
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
