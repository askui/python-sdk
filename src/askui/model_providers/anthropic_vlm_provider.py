"""AnthropicVlmProvider — VLM access via direct Anthropic API."""

import os
from functools import cached_property
from typing import Any

from anthropic import Anthropic
from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.request_size import ANTHROPIC_MAX_REQUEST_BYTES
from askui.models.shared.tools import ToolCollection
from askui.utils.model_pricing import ModelPricing

_DEFAULT_MODEL_ID = "claude-sonnet-4-6"


class AnthropicVlmProvider(VlmProvider):
    """VLM provider that routes requests directly to the Anthropic API.

    Supports Claude 4.x generation models. The API key is read from the
    `ANTHROPIC_API_KEY` environment variable lazily — validation happens on
    the first API call, not at construction time.

    Args:
        api_key (str | None, optional): Anthropic API key. Reads
            `ANTHROPIC_API_KEY` from the environment if not provided.
        base_url (str | None, optional): Base URL for the Anthropic API.
            Useful for proxies or custom endpoints.
        auth_token (str | None, optional): Authorization token for custom
            authentication. Added as an `Authorization` header.
        model_id (str, optional): Claude model to use. Defaults to
            `\"claude-sonnet-4-6\"`.
        client (Anthropic | None, optional): Pre-configured Anthropic client.
            If provided, other connection parameters are ignored.
        input_cost_per_million_tokens (float | None, optional): Override
            cost in USD per 1M input tokens. All override pricing params must be set to
            override the built-in defaults.
        output_cost_per_million_tokens (float | None, optional): Override
            cost in USD per 1M output tokens.
        cache_write_cost_per_million_tokens (float | None, optional): Override
            cost in USD per 1M cache write input tokens.
        cache_read_cost_per_million_tokens (float | None, optional): Override
            cost in USD per 1M cache read input tokens.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AnthropicVlmProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=AnthropicVlmProvider(
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
        model_id: str | None = None,
        client: Anthropic | None = None,
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
        cache_write_cost_per_million_tokens: float | None = None,
        cache_read_cost_per_million_tokens: float | None = None,
    ) -> None:
        self._model_id_value = (
            model_id or os.environ.get("VLM_PROVIDER_MODEL_ID") or _DEFAULT_MODEL_ID
        )
        if client is not None:
            self.client = client
        else:
            self.client = Anthropic(
                api_key=api_key,
                base_url=base_url,
                auth_token=auth_token,
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

    @property
    @override
    def max_request_bytes(self) -> int | None:
        return ANTHROPIC_MAX_REQUEST_BYTES

    @cached_property
    def _messages_api(self) -> AnthropicMessagesApi:
        """Lazily initialise the AnthropicMessagesApi on first use."""
        return AnthropicMessagesApi(client=self.client)

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
