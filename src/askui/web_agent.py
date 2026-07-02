import warnings

from pydantic import ConfigDict, validate_call

from askui.agent_base import Agent
from askui.agent_settings import AgentSettings
from askui.callbacks import ConversationCallback
from askui.container import telemetry
from askui.models.shared.settings import (
    ActSettings,
    MessageSettings,
)
from askui.models.shared.thinking import make_thinking_settings
from askui.models.shared.tools import Tool
from askui.models.shared.truncation_strategies import TruncationStrategy
from askui.prompts.act_prompts import create_web_agent_prompt
from askui.tools.exception_tool import ExceptionTool
from askui.tools.playwright.agent_os import PlaywrightAgentOs
from askui.tools.playwright.agent_os_facade import PlaywrightAgentOsFacade
from askui.tools.playwright.tools import (
    PlaywrightBackTool,
    PlaywrightForwardTool,
    PlaywrightGetPageTitleTool,
    PlaywrightGetPageUrlTool,
    PlaywrightGotoTool,
    PlaywrightKeyboardPressedTool,
    PlaywrightKeyboardReleaseTool,
    PlaywrightKeyboardTapTool,
    PlaywrightMouseClickTool,
    PlaywrightMouseHoldDownTool,
    PlaywrightMouseMoveTool,
    PlaywrightMouseReleaseTool,
    PlaywrightMouseScrollTool,
    PlaywrightScreenshotTool,
    PlaywrightTypeTool,
)

from .reporting import CompositeReporter, Reporter
from .retry import Retry


class WebAgent(Agent):
    @telemetry.record_call(
        exclude={
            "reporters",
            "settings",
            "act_tools",
            "callbacks",
            "truncation_strategy",
        }
    )
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reporters: list[Reporter] | None = None,
        settings: AgentSettings | None = None,
        retry: Retry | None = None,
        act_tools: list[Tool] | None = None,
        callbacks: list[ConversationCallback] | None = None,
        truncation_strategy: TruncationStrategy | None = None,
    ) -> None:
        reporter = CompositeReporter(reporters=reporters)
        self.os = PlaywrightAgentOs(reporter)
        super().__init__(
            reporter=reporter,
            retry=retry,
            tools=self.get_default_tools() + (act_tools or []),
            agent_os=self.os,
            settings=settings,
            callbacks=callbacks,
            truncation_strategy=truncation_strategy,
        )
        self.act_agent_os_facade = PlaywrightAgentOsFacade(
            self.os,
            coordinate_space=self._vlm_provider.coordinate_space,
            image_scaler=self._vlm_provider.image_scaler,
        )
        self.act_tool_collection.add_agent_os(self.act_agent_os_facade)
        self.act_settings = ActSettings(
            messages=MessageSettings(
                system=create_web_agent_prompt(),
                **make_thinking_settings(self._vlm_provider.model_id),
            ),
        )

    @staticmethod
    def get_default_tools() -> list[Tool]:
        return [
            PlaywrightScreenshotTool(),
            PlaywrightMouseMoveTool(),
            PlaywrightMouseClickTool(),
            PlaywrightMouseScrollTool(),
            PlaywrightMouseHoldDownTool(),
            PlaywrightMouseReleaseTool(),
            PlaywrightTypeTool(),
            PlaywrightKeyboardTapTool(),
            PlaywrightKeyboardPressedTool(),
            PlaywrightKeyboardReleaseTool(),
            PlaywrightGotoTool(),
            PlaywrightBackTool(),
            PlaywrightForwardTool(),
            PlaywrightGetPageTitleTool(),
            PlaywrightGetPageUrlTool(),
            ExceptionTool(),
        ]


class WebVisionAgent(WebAgent):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        warnings.warn(
            "WebVisionAgent is deprecated, use WebAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
