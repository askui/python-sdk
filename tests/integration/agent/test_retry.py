from typing import Optional, Tuple, Union

import pytest
from httpx import HTTPStatusError
from typing_extensions import override

from askui import AgentSettings, ComputerAgent, ConfigurableRetry
from askui.locators.locators import Locator
from askui.model_providers.detection_provider import DetectionProvider
from askui.models.exceptions import ElementNotFoundError, ModelNotFoundError
from askui.models.shared.settings import LocateSettings
from askui.models.types.geometry import PointList
from askui.tools.agent_os import ComputerAgentOS
from askui.utils.image_utils import ImageSource


class FailingDetectionProvider(DetectionProvider):
    def __init__(
        self, fail_times: int, succeed_point: Optional[Tuple[int, int]] = None
    ) -> None:
        self.fail_times = fail_times
        self.calls = 0
        if succeed_point is None:
            self.succeed_point = (10, 10)
        else:
            self.succeed_point = succeed_point

    @override
    def detect(
        self,
        locator: Union[str, Locator],
        image: ImageSource,  # noqa: ARG002
        locate_settings: LocateSettings,  # noqa: ARG002
    ) -> PointList:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ElementNotFoundError(locator, locator)
        return [self.succeed_point]

    @override
    def detect_all(
        self,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> list:
        return []


@pytest.fixture
def failing_provider() -> FailingDetectionProvider:
    return FailingDetectionProvider(fail_times=2)


@pytest.fixture
def always_failing_provider() -> FailingDetectionProvider:
    return FailingDetectionProvider(fail_times=10)


@pytest.fixture
def agent_with_retry(
    failing_provider: FailingDetectionProvider,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
) -> ComputerAgent:
    return ComputerAgent(
        settings=AgentSettings(detection_provider=failing_provider),
    )


@pytest.fixture
def agent_with_retry_on_multiple_exceptions(
    failing_provider: FailingDetectionProvider,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
) -> ComputerAgent:
    return ComputerAgent(
        settings=AgentSettings(detection_provider=failing_provider),
        retry=ConfigurableRetry(
            on_exception_types=(
                ElementNotFoundError,
                HTTPStatusError,
                ModelNotFoundError,
            ),
            strategy="Fixed",
            retry_count=3,
            base_delay=1,
        ),
    )


@pytest.fixture
def agent_always_fail(
    always_failing_provider: FailingDetectionProvider,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
) -> ComputerAgent:
    return ComputerAgent(
        settings=AgentSettings(detection_provider=always_failing_provider),
        retry=ConfigurableRetry(
            on_exception_types=(ElementNotFoundError,),
            strategy="Fixed",
            retry_count=3,
            base_delay=1,
        ),
    )


def test_locate_retries_and_succeeds(
    agent_with_retry: ComputerAgent, failing_provider: FailingDetectionProvider
) -> None:
    result = agent_with_retry.locate("something", screenshot=None)
    assert result == (10, 10)
    assert failing_provider.calls == 3  # 2 fails + 1 success


def test_locate_retries_on_multiple_exceptions_and_succeeds(
    agent_with_retry_on_multiple_exceptions: ComputerAgent,
    failing_provider: FailingDetectionProvider,
) -> None:
    result = agent_with_retry_on_multiple_exceptions.locate(
        "something", screenshot=None
    )
    assert result == (10, 10)
    assert failing_provider.calls == 3


def test_locate_retries_and_fails(
    agent_always_fail: ComputerAgent, always_failing_provider: FailingDetectionProvider
) -> None:
    with pytest.raises(ElementNotFoundError):
        agent_always_fail.locate("something", screenshot=None)
    assert always_failing_provider.calls == 3  # Only 3 attempts


def test_click_retries(
    agent_with_retry: ComputerAgent, failing_provider: FailingDetectionProvider
) -> None:
    agent_with_retry.click("something")
    assert failing_provider.calls == 3


def test_mouse_move_retries(
    agent_with_retry: ComputerAgent, failing_provider: FailingDetectionProvider
) -> None:
    agent_with_retry.mouse_move("something")
    assert failing_provider.calls == 3
