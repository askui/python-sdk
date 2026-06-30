"""Shared pytest fixtures for e2e tests."""

import pathlib
from typing import Generator

import pytest
from typing_extensions import override

from askui import AgentSettings, ComputerAgent
from askui.locators.locators import Locator
from askui.locators.serializers import AskUiLocatorSerializer
from askui.model_providers.detection_provider import DetectionProvider
from askui.models.askui.ai_element_utils import AiElementCollection
from askui.models.askui.inference_api import (
    AskUiInferenceApi,
    AskUiInferenceApiSettings,
)
from askui.models.askui.locate_api import AskUiInferenceLocateApi
from askui.models.askui.locate_models import (
    AskUiAiElementLocateModel,
    AskUiComboLocateModel,
    AskUiLocateModel,
    AskUiOcrLocateModel,
    AskUiPtaLocateModel,
)
from askui.models.models import LocateModel
from askui.models.shared.settings import LocateSettings
from askui.models.types.geometry import PointList
from askui.reporting import Reporter, SimpleHtmlReporter
from askui.tools.agent_os import ComputerAgentOS
from askui.utils.image_utils import ImageSource


class _LocateModelDetectionProvider(DetectionProvider):
    """Adapter wrapping a `LocateModel` as a `DetectionProvider` for e2e tests."""

    def __init__(self, locate_model: LocateModel) -> None:
        self._locate_model = locate_model

    @override
    def detect(
        self,
        locator: str | Locator,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> PointList:
        result: PointList = self._locate_model.locate(
            locator=locator,
            image=image,
            locate_settings=locate_settings,
        )
        return result


def _make_locate_api(path_fixtures: pathlib.Path) -> AskUiInferenceLocateApi:
    locator_serializer = AskUiLocatorSerializer(
        ai_element_collection=AiElementCollection(
            additional_ai_element_locations=[path_fixtures / "images"]
        ),
        reporter=SimpleHtmlReporter(),
    )
    return AskUiInferenceLocateApi(
        locator_serializer=locator_serializer,
        inference_api=AskUiInferenceApi(settings=AskUiInferenceApiSettings()),
    )


@pytest.fixture
def simple_html_reporter() -> Reporter:
    return SimpleHtmlReporter()


@pytest.fixture
def askui_locate_model(path_fixtures: pathlib.Path) -> LocateModel:
    return AskUiLocateModel(locate_api=_make_locate_api(path_fixtures))


@pytest.fixture
def pta_locate_model(path_fixtures: pathlib.Path) -> LocateModel:
    return AskUiPtaLocateModel(locate_api=_make_locate_api(path_fixtures))


@pytest.fixture
def ocr_locate_model(path_fixtures: pathlib.Path) -> LocateModel:
    return AskUiOcrLocateModel(locate_api=_make_locate_api(path_fixtures))


@pytest.fixture
def ai_element_locate_model(path_fixtures: pathlib.Path) -> LocateModel:
    return AskUiAiElementLocateModel(locate_api=_make_locate_api(path_fixtures))


@pytest.fixture
def combo_locate_model(path_fixtures: pathlib.Path) -> LocateModel:
    return AskUiComboLocateModel(locate_api=_make_locate_api(path_fixtures))


@pytest.fixture
def agent_with_pta_model(
    pta_locate_model: LocateModel,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
) -> Generator[ComputerAgent, None, None]:
    with ComputerAgent(
        settings=AgentSettings(
            detection_provider=_LocateModelDetectionProvider(pta_locate_model)
        ),
        reporters=[simple_html_reporter],
    ) as agent:
        yield agent


@pytest.fixture
def agent_with_ocr_model(
    ocr_locate_model: LocateModel,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
) -> Generator[ComputerAgent, None, None]:
    with ComputerAgent(
        settings=AgentSettings(
            detection_provider=_LocateModelDetectionProvider(ocr_locate_model)
        ),
        reporters=[simple_html_reporter],
    ) as agent:
        yield agent


@pytest.fixture
def agent_with_ai_element_model(
    ai_element_locate_model: LocateModel,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
) -> Generator[ComputerAgent, None, None]:
    with ComputerAgent(
        settings=AgentSettings(
            detection_provider=_LocateModelDetectionProvider(ai_element_locate_model)
        ),
        reporters=[simple_html_reporter],
    ) as agent:
        yield agent


@pytest.fixture
def agent_with_combo_model(
    combo_locate_model: LocateModel,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
) -> Generator[ComputerAgent, None, None]:
    with ComputerAgent(
        settings=AgentSettings(
            detection_provider=_LocateModelDetectionProvider(combo_locate_model)
        ),
        reporters=[simple_html_reporter],
    ) as agent:
        yield agent


@pytest.fixture
def vision_agent(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
) -> Generator[ComputerAgent, None, None]:
    """Fixture providing a ComputerAgent instance."""
    with ComputerAgent(
        reporters=[simple_html_reporter],
    ) as agent:
        yield agent
