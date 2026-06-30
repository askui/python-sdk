from pathlib import Path

from pydantic import ConfigDict, validate_call

from askui.agent_settings import AgentSettings
from askui.models.shared.secrets import Secret
from askui.models.shared.settings import (
    ActSettings,
    MessageSettings,
)
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
    """Web testing agent that extends `WebAgent` with feature, scenario and
    execution management tools for authoring and running browser-based tests.

    Args:
        reporters (list[Reporter] | None, optional): Reporters used for reporting.
            Defaults to `None`.
        settings (AgentSettings | None, optional): Agent settings. Defaults to
            `None`.
        retry (Retry | None, optional): Retry strategy. Defaults to `None`.
        secrets (list[Secret] | None, optional): Sensitive values (e.g. passwords)
            the agent may use but the LLM must never see. The model only sees the
            placeholder `<|secret|>NAME<|secret|>`; the real value is substituted at
            execution time and kept out of the LLM prompt, reporter, logs and cache.
            Also usable in deterministic `type()` and overridable per call via
            `act(..., secrets=[...])`. Note: a secret typed into a visible field may
            still appear in screenshots sent to the model; on-screen secrets cannot
            currently be hidden. Defaults to `None`.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reporters: list[Reporter] | None = None,
        settings: AgentSettings | None = None,
        retry: Retry | None = None,
        secrets: list[Secret] | None = None,
    ) -> None:
        base_dir = Path.cwd() / "chat" / "testing"
        base_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            reporters=reporters,
            settings=settings,
            retry=retry,
            secrets=secrets,
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
                thinking={"type": "enabled", "budget_tokens": 2048},
            ),
        )
