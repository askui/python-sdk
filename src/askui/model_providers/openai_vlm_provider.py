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
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.model_pricing import ModelPricing

_DEFAULT_MODEL_ID = "gpt-5.4"


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
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
        cache_write_cost_per_million_tokens: float | None = None,
        cache_read_cost_per_million_tokens: float | None = None,
    ) -> None:
        self._model_id_value = (
            model_id or os.environ.get("VLM_PROVIDER_MODEL_ID") or _DEFAULT_MODEL_ID
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
    def pricing(self) -> ModelPricing | None:
        return self._pricing

    @cached_property
    def _messages_api(self) -> OpenAIMessagesApi:
        """Lazily initialise the `OpenAIMessagesApi` on first use."""
        return OpenAIMessagesApi(client=self._client)

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
