import pytest
from PIL import Image

from askui import ComputerAgent
from askui import locators as loc
from askui.container import telemetry
from askui.telemetry.processors import Segment, SegmentSettings
from askui.tools.agent_os import ComputerAgentOS


@pytest.mark.timeout(60)
def test_telemetry_with_nonexistent_domain_should_not_block(
    github_login_screenshot: Image.Image,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
) -> None:
    telemetry.set_processors(
        [
            Segment(
                SegmentSettings(
                    api_url="https://this-domain-does-not-exist-123456789.com",
                    write_key="1234567890",
                )
            )
        ]
    )
    with ComputerAgent() as agent:
        agent.locate(loc.Text(), screenshot=github_login_screenshot)
    assert True
