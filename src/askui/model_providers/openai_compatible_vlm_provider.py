"""OpenAICompatibleVlmProvider — VLM access via a fixed endpoint URL."""

import httpx
from openai import OpenAI

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider


class OpenAICompatibleVlmProvider(OpenAIVlmProvider):
    """VLM provider for OpenAI-compatible APIs that require an exact endpoint URL.

    The OpenAI SDK always appends ``/chat/completions`` to ``base_url``,
    which breaks endpoints that already include the full path (e.g. RunPod,
    custom proxies, serverless deployments). This provider works around
    the issue by installing an httpx event hook that rewrites every
    outgoing request URL to the exact ``endpoint_url``.

    Args:
        endpoint_url (str): Full endpoint URL including the path
            (e.g. ``"https://my-host/v1/chat/completions"``).
        model_id (str): Model name expected by the deployment.
        api_key (str | None, optional): API key for the endpoint.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import OpenAICompatibleVlmProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=OpenAICompatibleVlmProvider(
                endpoint_url="https://my-host/v1/chat/completions",
                model_id="my-model",
                api_key="...",
            )
        ))
        ```
    """

    def __init__(
        self,
        endpoint_url: str,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        def _rewrite_url(request: httpx.Request) -> None:
            request.url = httpx.URL(endpoint_url)

        http_client = httpx.Client(event_hooks={"request": [_rewrite_url]})

        client = OpenAI(
            api_key=api_key,
            base_url=endpoint_url,
            http_client=http_client,
        )

        super().__init__(
            model_id=model_id,
            client=client,
        )
