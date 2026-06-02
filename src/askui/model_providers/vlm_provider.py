"""VlmProvider interface for Vision Language Models with tool-calling capability."""

from abc import ABC, abstractmethod
from typing import Any

from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.model_pricing import ModelPricing


class VlmProvider(ABC):
    """Interface for Vision Language Model providers.

    A `VlmProvider` encapsulates both the endpoint/credentials and the model ID
    for a VLM that supports multimodal input and tool-calling. It is used for
    `agent.act()` and any tool that requires LLM reasoning.

    The provider owns the model selection — the `model_id` is configured on the
    provider instance, not passed per-call.

    To bring your own VLM, implement this interface.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AskUIVlmProvider

        provider = AskUIVlmProvider(
            workspace_id="...",
            token="...",
            model_id="claude-sonnet-4-5-20251101",
        )
        agent = ComputerAgent(settings=AgentSettings(vlm_provider=provider))
        ```
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier used by this provider."""

    @property
    def pricing(self) -> ModelPricing | None:
        """Pricing information for this provider's model.

        Returns ``None`` if no pricing information is available.
        Override in subclasses to provide model-specific pricing.
        """
        return None

    @property
    def max_request_bytes(self) -> int | None:
        """Hard cap on the serialized request body size, in bytes.
        Returns ``None`` when the provider has no known/specific limit, in
        which case callers fall back to a conservative default. Override in
        subclasses backed by an endpoint with a known cap.
        """
        return None

    @abstractmethod
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
        """Create a message using this provider's VLM.

        The model used is determined by `self.model_id`.

        Args:
            messages (list[MessageParam]): The message history.
            tools (ToolCollection | None): Tools available to the model.
            max_tokens (int | None): Maximum tokens to generate.
            system (SystemPrompt | None): The system prompt.
            thinking (ThinkingConfigParam | None): Provider-specific thinking config.
            tool_choice (ToolChoiceParam | None): Provider-specific tool choice config.
            temperature (float | None): Sampling temperature (0–1).
            provider_options (dict[str, Any] | None): Provider-specific options.
                Each provider can define its own keys. Common options include:
                - "betas": List of beta features to enable (e.g., for Anthropic)

        Returns:
            MessageParam: The model's response message.
        """
