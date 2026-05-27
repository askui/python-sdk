"""Model provider interfaces and built-in implementations.

Provider interfaces:
- `VlmProvider` — multimodal input + tool-calling (for `act` and LLM-backed tools)
- `ImageQAProvider` — multimodal Q&A and structured output (for `get`)
- `DetectionProvider` — UI element coordinates from screenshot + locator (for `locate`)

Built-in providers:
- `AskUIVlmProvider` — VLM via AskUI's hosted Anthropic proxy
- `AskUIImageQAProvider` — image Q&A via AskUI's hosted Gemini proxy
- `AskUIDetectionProvider` — element detection via AskUI's inference API
- `AnthropicVlmProvider` — VLM via direct Anthropic API
- `AnthropicImageQAProvider` — image Q&A via direct Anthropic API
- `GoogleImageQAProvider` — image Q&A via Google Gemini API (direct, no proxy)
- `OpenAIVlmProvider` — VLM via any OpenAI-compatible API
- `OpenAIImageQAProvider` — image Q&A via any OpenAI-compatible API
- `OllamaVlmProvider` — VLM via local Ollama instance (OpenAI-compatible)
- `OllamaImageQAProvider` — image Q&A via local Ollama instance (OpenAI-compatible)
- `OpenAICompatibleVlmProvider` — VLM via OpenAI-compatible API with fixed URL
"""

from askui.model_providers.anthropic_image_qa_provider import AnthropicImageQAProvider
from askui.model_providers.anthropic_vlm_provider import AnthropicVlmProvider
from askui.model_providers.askui_detection_provider import AskUIDetectionProvider
from askui.model_providers.askui_image_qa_provider import AskUIImageQAProvider
from askui.model_providers.askui_vlm_provider import AskUIVlmProvider
from askui.model_providers.detection_provider import DetectionProvider
from askui.model_providers.google_image_qa_provider import GoogleImageQAProvider
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.model_providers.ollama_image_qa_provider import OllamaImageQAProvider
from askui.model_providers.ollama_vlm_provider import OllamaVlmProvider
from askui.model_providers.openai_compatible_vlm_provider import (
    OpenAICompatibleVlmProvider,
)
from askui.model_providers.openai_image_qa_provider import OpenAIImageQAProvider
from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider
from askui.model_providers.vlm_provider import VlmProvider
from askui.utils.model_pricing import ModelPricing

__all__ = [
    "AnthropicImageQAProvider",
    "AnthropicVlmProvider",
    "AskUIDetectionProvider",
    "AskUIImageQAProvider",
    "AskUIVlmProvider",
    "DetectionProvider",
    "GoogleImageQAProvider",
    "ImageQAProvider",
    "ModelPricing",
    "OllamaImageQAProvider",
    "OllamaVlmProvider",
    "OpenAIImageQAProvider",
    "OpenAIVlmProvider",
    "OpenAICompatibleVlmProvider",
    "VlmProvider",
]
