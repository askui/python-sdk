"""OpenAICompatibleVlmProvider — VLM access via a fixed endpoint URL."""

import httpx
from openai import OpenAI

from askui.model_providers.openai_vlm_provider import OpenAIVlmProvider
from askui.models.shared.coordinate_space import (
    PixelCoordinateSpace,
    VlmCoordinateSpace,
)
from askui.models.shared.image_scaler import ImageScaler

_DEFAULT_COORDINATE_SPACE = PixelCoordinateSpace()


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
        coordinate_space (`VlmCoordinateSpace` | None, optional): The coordinate
            grid the model emits coordinates in.  If ``None``, inherits the
            default from `OpenAIVlmProvider` (pixel coordinates).
        image_scaler (`ImageScaler` | None, optional): Custom image preprocessing
            callable. If ``None``, inherits from `OpenAIVlmProvider`.
        image_edge_max (int | None, optional): Maximum edge length (in pixels)
            for screenshots sent to the model.  Only used when ``image_scaler``
            is not provided.  Reads ``ASKUI_VLM_MAX_IMAGE_EDGE`` from the
            environment if not provided.  Inherits the default from
            `OpenAIVlmProvider` (1024).

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
        coordinate_space: VlmCoordinateSpace = _DEFAULT_COORDINATE_SPACE,
        image_scaler: ImageScaler | None = None,
        image_edge_max: int | None = None,
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
            coordinate_space=coordinate_space,
            image_scaler=image_scaler,
            image_edge_max=image_edge_max,
        )
