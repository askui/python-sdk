from pathlib import Path

from pydantic import ConfigDict, validate_call

from askui.agent_settings import AgentSettings
from askui.models.shared.settings import (
    ActSettings,
    MessageSettings,
)
from askui.models.shared.thinking import make_thinking_settings
from askui.prompts.act_prompts import create_web_agent_prompt
from askui.tools.testing.execution_tools import (
    CreateExecutionTool,
    DeleteExecutionTool,
    ListExecutionTool,
    ModifyExecutionTool,
    RetrieveExecutionTool,
)
from askui.tools.testing.feature_tools import (
    CreateFeatureTool,
    DeleteFeatureTool,
    ListFeatureTool,
    ModifyFeatureTool,
    RetrieveFeatureTool,
)
from askui.tools.testing.scenario_tools import (
    CreateScenarioTool,
    DeleteScenarioTool,
    ListScenarioTool,
    ModifyScenarioTool,
    RetrieveScenarioTool,
)
from askui.web_agent import WebVisionAgent

from .reporting import Reporter
from .retry import Retry


class WebTestingAgent(WebVisionAgent):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reporters: list[Reporter] | None = None,
        settings: AgentSettings | None = None,
        retry: Retry | None = None,
    ) -> None:
        base_dir = Path.cwd() / "chat" / "testing"
        base_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            reporters=reporters,
            settings=settings,
            retry=retry,
            act_tools=[
                CreateFeatureTool(base_dir),
                RetrieveFeatureTool(base_dir),
                ListFeatureTool(base_dir),
                ModifyFeatureTool(base_dir),
                DeleteFeatureTool(base_dir),
                CreateScenarioTool(base_dir),
                RetrieveScenarioTool(base_dir),
                ListScenarioTool(base_dir),
                ModifyScenarioTool(base_dir),
                DeleteScenarioTool(base_dir),
                CreateExecutionTool(base_dir),
                RetrieveExecutionTool(base_dir),
                ListExecutionTool(base_dir),
                ModifyExecutionTool(base_dir),
                DeleteExecutionTool(base_dir),
            ],
        )
        self.act_settings = ActSettings(
            messages=MessageSettings(
                system=create_web_agent_prompt(),
                **make_thinking_settings(self._vlm_provider.model_id),
            ),
        )
