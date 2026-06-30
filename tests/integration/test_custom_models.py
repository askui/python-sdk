"""Integration tests for custom provider usage with AgentSettings."""

import pathlib
from typing import Any, Union

import pytest
from typing_extensions import override

from askui import (
    AgentSettings,
    ComputerAgent,
    Point,
    PointList,
    ResponseSchema,
    ResponseSchemaBase,
)
from askui.locators.locators import Locator
from askui.model_providers.detection_provider import DetectionProvider
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import (
    MessageParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.settings import GetSettings, LocateSettings
from askui.models.shared.tools import ToolCollection
from askui.tools.agent_os import ComputerAgentOS
from askui.utils.image_utils import ImageSource
from askui.utils.source_utils import Source


class SimpleVlmProvider(VlmProvider):
    """Simple VLM provider that records goals."""

    def __init__(self, model_id: str = "test-model") -> None:
        self.goals: list[list[dict[str, Any]]] = []
        self._model_id = model_id

    @property
    @override
    def model_id(self) -> str:
        return self._model_id

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
        self.goals.append([msg.model_dump(mode="json") for msg in messages])
        return MessageParam(
            role="assistant",
            content="done",
            stop_reason="end_turn",
        )


class SimpleImageQAProvider(ImageQAProvider):
    """Simple image Q&A provider that returns a fixed response."""

    def __init__(self, response: str | ResponseSchemaBase = "test response") -> None:
        self.queries: list[str] = []
        self.sources: list[Source] = []
        self.schemas: list[Any] = []
        self.response = response

    @override
    def query(
        self,
        query: str,
        source: Source,
        response_schema: type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        self.queries.append(query)
        self.sources.append(source)
        self.schemas.append(response_schema)
        if (
            response_schema is not None
            and isinstance(self.response, response_schema)
            or isinstance(self.response, str)
        ):
            return self.response
        err_msg = (
            "Response schema does not match the response type. "
            "Please use a response schema that matches the response type."
        )
        raise ValueError(err_msg)


class SimpleDetectionProvider(DetectionProvider):
    """Simple detection provider that returns fixed coordinates."""

    def __init__(self, point: Point = (100, 100)) -> None:
        self.locators: list[str | Locator] = []
        self.images: list[ImageSource] = []
        self._point = point

    @override
    def detect(
        self,
        locator: Union[str, Locator],
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> PointList:
        self.locators.append(locator)
        self.images.append(image)
        return [self._point]

    @override
    def detect_all(
        self,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> list:
        return []


class SimpleResponseSchema(ResponseSchemaBase):
    """Simple response schema for testing."""

    value: str


class TestCustomProviders:
    """Test suite for custom provider injection via AgentSettings."""

    @pytest.fixture
    def vlm_provider(self) -> SimpleVlmProvider:
        return SimpleVlmProvider()

    @pytest.fixture
    def image_qa_provider(self) -> SimpleImageQAProvider:
        return SimpleImageQAProvider()

    @pytest.fixture
    def detection_provider(self) -> SimpleDetectionProvider:
        return SimpleDetectionProvider()

    def test_inject_and_use_custom_vlm_provider(
        self,
        vlm_provider: SimpleVlmProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test injecting and using a custom VLM provider."""
        with ComputerAgent(
            settings=AgentSettings(vlm_provider=vlm_provider),
        ) as agent:
            agent.act("test goal")

        assert len(vlm_provider.goals) == 1
        assert len(vlm_provider.goals[0]) == 1
        msg = vlm_provider.goals[0][0]
        assert msg["role"] == "user"
        assert msg["stop_reason"] is None
        content = msg["content"]
        if isinstance(content, str):
            assert content == "test goal"
        else:
            assert any(
                block.get("text") == "test goal"
                for block in content
                if isinstance(block, dict)
            )

    def test_inject_and_use_custom_image_qa_provider(
        self,
        image_qa_provider: SimpleImageQAProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test injecting and using a custom image Q&A provider."""
        with ComputerAgent(
            settings=AgentSettings(image_qa_provider=image_qa_provider),
        ) as agent:
            result = agent.get("test query")

        assert result == "test response"
        assert image_qa_provider.queries == ["test query"]

    def test_inject_and_use_custom_image_qa_provider_with_pdf(
        self,
        image_qa_provider: SimpleImageQAProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
        path_fixtures_dummy_pdf: pathlib.Path,
    ) -> None:
        """Test injecting and using a custom image Q&A provider with a PDF."""
        with ComputerAgent(
            settings=AgentSettings(image_qa_provider=image_qa_provider),
        ) as agent:
            result = agent.get("test query", source=path_fixtures_dummy_pdf)

        assert result == "test response"
        assert image_qa_provider.queries == ["test query"]

    def test_inject_and_use_custom_detection_provider(
        self,
        detection_provider: SimpleDetectionProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test injecting and using a custom detection provider."""
        with ComputerAgent(
            settings=AgentSettings(detection_provider=detection_provider),
        ) as agent:
            agent.click("test element")

        assert detection_provider.locators == ["test element"]

    def test_inject_all_custom_providers(
        self,
        vlm_provider: SimpleVlmProvider,
        image_qa_provider: SimpleImageQAProvider,
        detection_provider: SimpleDetectionProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test injecting all custom providers at once."""
        with ComputerAgent(
            settings=AgentSettings(
                vlm_provider=vlm_provider,
                image_qa_provider=image_qa_provider,
                detection_provider=detection_provider,
            ),
        ) as agent:
            agent.act("test goal")
            result = agent.get("test query")
            agent.click("test element")

        assert len(vlm_provider.goals) == 1
        assert len(vlm_provider.goals[0]) == 1
        msg = vlm_provider.goals[0][0]
        assert msg["role"] == "user"
        assert msg["stop_reason"] is None
        content = msg["content"]
        if isinstance(content, str):
            assert content == "test goal"
        else:
            assert any(
                block.get("text") == "test goal"
                for block in content
                if isinstance(block, dict)
            )
        assert image_qa_provider.queries == ["test query"]
        assert result == "test response"
        assert detection_provider.locators == ["test element"]

    def test_use_response_schema_with_custom_image_qa_provider(
        self,
        image_qa_provider: SimpleImageQAProvider,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test using a response schema with a custom image Q&A provider."""
        response = SimpleResponseSchema(value="test value")
        image_qa_provider.response = response

        with ComputerAgent(
            settings=AgentSettings(image_qa_provider=image_qa_provider),
        ) as agent:
            result = agent.get("test query", response_schema=SimpleResponseSchema)

        assert isinstance(result, SimpleResponseSchema)
        assert result.value == "test value"
        assert image_qa_provider.schemas == [SimpleResponseSchema]

    def test_defaults_to_built_in_providers_when_not_provided(
        self,
        agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG002
    ) -> None:
        """Test agent uses built-in defaults when custom ones not provided."""
        with ComputerAgent() as agent:
            assert agent is not None
