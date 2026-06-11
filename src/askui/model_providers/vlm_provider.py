"""VlmProvider interface for Vision Language Models with tool-calling capability."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

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
from askui.utils.llm_image_utils import compute_contained_size, resize_image
from askui.utils.model_pricing import ModelPricing

_DEFAULT_COORDINATE_SPACE = PixelCoordinateSpace()


def _default_image_scaler(image: Image.Image) -> Image.Image:
    target = compute_contained_size(image.width, image.height)
    return resize_image(image, target)


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
    def coordinate_space(self) -> VlmCoordinateSpace:
        """The coordinate space this model emits coordinates in.

        Returns a `VlmCoordinateSpace` describing the grid the model uses.
        The default is `PixelCoordinateSpace` (native pixel coordinates).
        Override in subclasses when the model uses a different grid
        (e.g. ``ScaledCoordinateSpace(1000, 1000)`` for Qwen).
        """
        return _DEFAULT_COORDINATE_SPACE

    @property
    def pricing(self) -> ModelPricing | None:
        """Pricing information for this provider's model.

        Returns ``None`` if no pricing information is available.
        Override in subclasses to provide model-specific pricing.
        """
        return None

    @property
    def image_scaler(self) -> ImageScaler:
        """Callable that preprocesses a screenshot before sending to the model.

        Override in subclasses for provider-specific sizing.
        """
        return _default_image_scaler

    def augment_system_prompt(self, system: SystemPrompt) -> SystemPrompt:
        """Hook for providers to augment the system prompt before sending.

        Called by ``create_message()`` implementations.  The base
        implementation returns the prompt unchanged.  Override in
        subclasses that need to inject provider-specific information
        (e.g. coordinate bounds for non-Anthropic models).

        The original ``SystemPrompt`` object is **not** mutated —
        implementations should create a new ``SystemPrompt`` wrapping
        the augmented text.
        """
        return system

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
