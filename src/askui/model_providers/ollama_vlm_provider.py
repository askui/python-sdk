"""OllamaVlmProvider — VLM access via a local Ollama instance."""

from openai import OpenAI

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL_ID = "qwen3.5"


class OllamaVlmProvider(OpenAIVlmProvider):
    """VLM provider that routes requests to a local Ollama instance.

    Thin convenience wrapper around `OpenAIVlmProvider` with Ollama
    defaults (``base_url``, ``api_key``, ``model_id``).

    Args:
        model_id (str, optional): Ollama model to use. Defaults to
            ``"qwen3.5"``.
        base_url (str, optional): Base URL for the Ollama OpenAI-compatible
            API. Defaults to ``"http://localhost:11434/v1"``.
        client (`OpenAI` | None, optional): Pre-configured OpenAI client.
            If provided, ``base_url`` is ignored.

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
    ) -> None:
        super().__init__(
            model_id=model_id,
            api_key="ollama",  # Ollama requires no auth; OpenAI SDK needs a value
            base_url=base_url,
            client=client,
        )
