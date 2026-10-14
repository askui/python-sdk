"""AgentSettings — provider-based configuration for Agent."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Type

from typing_extensions import override

if TYPE_CHECKING:
    from askui.locators.locators import Locator

from askui.model_providers.askui_detection_provider import AskUIDetectionProvider
from askui.model_providers.askui_image_qa_provider import AskUIImageQAProvider
from askui.model_providers.askui_vlm_provider import AskUIVlmProvider
from askui.model_providers.detection_provider import DetectionProvider
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.model_providers.vlm_provider import VlmProvider
from askui.models.models import DetectedElement, GetModel, LocateModel
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.settings import GetSettings, LocateSettings
from askui.models.shared.tools import ToolCollection
from askui.models.types.geometry import PointList
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.image_utils import ImageSource
from askui.utils.source_utils import Source


class _VlmProviderMessagesApiAdapter(MessagesApi):
    """Internal adapter that wraps a VlmProvider as a MessagesApi.

    The ``model_id`` argument accepted by ``MessagesApi.create_message`` is
    ignored — the provider already owns the model selection.
    """

    def __init__(self, provider: VlmProvider) -> None:
        self._provider = provider

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        model_id: str,  # noqa: ARG002
        tools: ToolCollection | None = None,
        max_tokens: int | None = None,
        system: SystemPrompt | None = None,
        thinking: ThinkingConfigParam | None = None,
        tool_choice: ToolChoiceParam | None = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        return self._provider.create_message(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            system=system,
            thinking=thinking,
            tool_choice=tool_choice,
            temperature=temperature,
            provider_options=provider_options,
        )


class _ImageQAProviderGetModelAdapter(GetModel):
    """Internal adapter that wraps an ImageQAProvider as a GetModel."""

    def __init__(self, provider: ImageQAProvider) -> None:
        self._provider = provider

    @override
    def get(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        return self._provider.query(
            query=query,
            source=source,
            response_schema=response_schema,
            get_settings=get_settings,
        )


class _DetectionProviderLocateModelAdapter(LocateModel):
    """Internal adapter that wraps a DetectionProvider as a LocateModel."""

    def __init__(self, provider: DetectionProvider) -> None:
        self._provider = provider

    @override
    def locate(
        self,
        locator: "str | Locator",
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> PointList:
        from askui.locators.locators import Locator, Text

        _locator: str | Locator = Text(locator) if isinstance(locator, str) else locator
        return self._provider.detect(
            locator=_locator,
            image=image,
            locate_settings=locate_settings,
        )

    @override
    def locate_all_elements(
        self,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> list[DetectedElement]:
        return self._provider.detect_all(
            image=image,
            locate_settings=locate_settings,
        )


class AgentSettings:
    """Provider-based configuration for the Agent.

    Holds one provider per AI capability. Each provider encapsulates its own
    endpoint, credentials, and model ID. Credentials are validated lazily —
    only when the first API call is made.

    The default providers read credentials from environment variables:
    - `vlm_provider` → `AskUIVlmProvider`
      (reads ``ASKUI_WORKSPACE_ID`` / ``ASKUI_TOKEN``)
    - `image_qa_provider` → `AskUIImageQAProvider`
      (reads ``ASKUI_WORKSPACE_ID`` / ``ASKUI_TOKEN``)
    - `detection_provider` → `AskUIDetectionProvider`
      (reads ``ASKUI_WORKSPACE_ID`` / ``ASKUI_TOKEN``)

    Args:
        vlm_provider (VlmProvider | None, optional): Provider for multimodal
            input + tool-calling (used by `act()`). Defaults to
            `AskUIVlmProvider`.
        image_qa_provider (ImageQAProvider | None, optional): Provider for
            multimodal Q&A and structured output (used by `get()`). Defaults to
            `AskUIImageQAProvider`.
        detection_provider (DetectionProvider | None, optional): Provider for
            UI element coordinate detection (used by `locate()`). Defaults to
            `AskUIDetectionProvider`.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import AskUIVlmProvider, AskUIImageQAProvider

        agent = ComputerAgent(settings=AgentSettings(
            vlm_provider=AskUIVlmProvider(model_id=\"claude-opus-4-6\"),
            image_qa_provider=AskUIImageQAProvider(model_id=\"gemini-3.5-flash\"),
        ))
        ```
    """

    def __init__(
        self,
        vlm_provider: VlmProvider | None = None,
        image_qa_provider: ImageQAProvider | None = None,
        detection_provider: DetectionProvider | None = None,
    ) -> None:
        self._vlm_provider = vlm_provider
        self._image_qa_provider = image_qa_provider
        self._detection_provider = detection_provider

    @cached_property
    def vlm_provider(self) -> VlmProvider:
        """Return the VlmProvider, creating the default if not provided."""
        if self._vlm_provider is not None:
            return self._vlm_provider
        return AskUIVlmProvider()

    @cached_property
    def image_qa_provider(self) -> ImageQAProvider:
        """Return the ImageQAProvider, creating the default if not provided."""
        if self._image_qa_provider is not None:
            return self._image_qa_provider
        return AskUIImageQAProvider()

    @cached_property
    def detection_provider(self) -> DetectionProvider:
        """Return the DetectionProvider, creating the default if not provided."""
        if self._detection_provider is not None:
            return self._detection_provider
        return AskUIDetectionProvider()

    def to_messages_api(self) -> MessagesApi:
        """Return a MessagesApi adapter backed by this settings' VlmProvider."""
        return _VlmProviderMessagesApiAdapter(self.vlm_provider)

    def to_get_model(self) -> GetModel:
        """Return a GetModel adapter backed by this settings' ImageQAProvider."""
        return _ImageQAProviderGetModelAdapter(self.image_qa_provider)

    def to_locate_model(self) -> LocateModel:
        """Return a LocateModel adapter backed by this settings' DetectionProvider."""
        return _DetectionProviderLocateModelAdapter(self.detection_provider)
