"""OllamaVlmProvider — VLM access via a local Ollama instance."""

from openai import OpenAI
from typing_extensions import override

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider
from askui.models.shared.coordinate_space import (
    PixelCoordinateSpace,
    ScaledCoordinateSpace,
    VlmCoordinateSpace,
)
from askui.models.shared.image_scaler import ImageScaler

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL_ID = "qwen3.5"

_QWEN_COORDINATE_SPACE = ScaledCoordinateSpace(width=1000, height=1000)
_HOLO_COORDINATE_SPACE = ScaledCoordinateSpace(width=1000, height=1000)
_KIMI_COORDINATE_SPACE = ScaledCoordinateSpace(width=1000, height=1000)


class OllamaVlmProvider(OpenAIVlmProvider):
    """VLM provider that routes requests to a local Ollama instance.

    Thin convenience wrapper around `OpenAIVlmProvider` with Ollama
    defaults (``base_url``, ``api_key``, ``model_id``).

    Qwen and Holo models are automatically detected and their coordinate
    space is set to ``ScaledCoordinateSpace(width=1000, height=1000)``.
    Kimi models use ``NormalizedCoordinateSpace()``.
    Pass ``coordinate_space`` explicitly to override auto-detection.

    Args:
        model_id (str, optional): Ollama model to use. Defaults to
            ``"qwen3.5"``.
        base_url (str, optional): Base URL for the Ollama OpenAI-compatible
            API. Defaults to ``"http://localhost:11434/v1"``.
        client (`OpenAI` | None, optional): Pre-configured OpenAI client.
            If provided, ``base_url`` is ignored.
        coordinate_space (VlmCoordinateSpace | None, optional): The coordinate
            grid the model emits coordinates in.  ``None`` (the default)
            enables auto-detection based on ``model_id``.
        image_scaler (`ImageScaler` | None, optional): Custom image preprocessing
            callable. If ``None``, inherits from `OpenAIVlmProvider`.
        max_image_edge (int | None, optional): Maximum edge length (in pixels)
            for screenshots sent to the model.  Reads ``ASKUI_VLM_MAX_IMAGE_EDGE``
            from the environment if not provided.  Inherits the default from
            `OpenAIVlmProvider` (2048).

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import OllamaVlmProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=OllamaVlmProvider(
                model_id="qwen3.5",
            )
        ))
        ```
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        base_url: str = _DEFAULT_BASE_URL,
        client: OpenAI | None = None,
        coordinate_space: VlmCoordinateSpace | None = None,
        image_scaler: ImageScaler | None = None,
        max_image_edge: int | None = None,
    ) -> None:
        self._coordinate_space_override = coordinate_space
        super().__init__(
            model_id=model_id,
            api_key="ollama",  # Ollama requires no auth; OpenAI SDK needs a value
            base_url=base_url,
            client=client,
            coordinate_space=coordinate_space or PixelCoordinateSpace(),
            image_scaler=image_scaler,
            max_image_edge=max_image_edge,
        )

    @property
    @override
    def coordinate_space(self) -> VlmCoordinateSpace:
        if self._coordinate_space_override is not None:
            return self._coordinate_space_override
        model_lower = self._model_id_value.lower()
        if "qwen" in model_lower:
            return _QWEN_COORDINATE_SPACE
        if "holo" in model_lower:
            return _HOLO_COORDINATE_SPACE
        if "kimi" in model_lower:
            return _KIMI_COORDINATE_SPACE
        return self._coordinate_space
