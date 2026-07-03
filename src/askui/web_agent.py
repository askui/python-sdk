import warnings
from pathlib import Path

from pydantic import ConfigDict, validate_call

from askui.agent_base import Agent
from askui.agent_settings import AgentSettings
from askui.callbacks import ConversationCallback
from askui.container import telemetry
from askui.models.shared.secrets import Secret
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
    """Web automation agent backed by a Playwright browser.

    Args:
        reporters (list[Reporter] | None, optional): Reporters used for reporting.
            Defaults to `None`.
        settings (AgentSettings | None, optional): Agent settings. Defaults to
            `None`.
        retry (Retry | None, optional): Retry strategy. Defaults to `None`.
        act_tools (list[Tool] | None, optional): Additional tools made available
            during `act()`. Defaults to `None`.
        callbacks (list[ConversationCallback] | None, optional): Conversation
            callbacks. Defaults to `None`.
        truncation_strategy (TruncationStrategy | None, optional): Message history
            truncation strategy. Defaults to `None`.
        secrets (list[Secret] | None, optional): Sensitive values (e.g. passwords)
            the agent may use but the LLM must never see. The model only sees the
            placeholder `<|secret|>NAME<|secret|>`; the real value is substituted at
            execution time and kept out of the LLM prompt, reporter, logs and cache.
            Also usable in deterministic `type()` and overridable per call via
            `act(..., secrets=[...])`. Note: a secret typed into a visible field may
            still appear in screenshots sent to the model; on-screen secrets cannot
            currently be hidden. Defaults to `None`.
        download_dir (str | Path | None, optional): Directory into which files
            downloaded by the browser are automatically copied once they finish
            (auto-renamed on filename collision). When `None`, downloads are left
            in Playwright's temporary location and removed when the browser
            closes. Defaults to `None`.

    Example:
        ```python
        from askui import WebAgent

        with WebAgent(download_dir="~/Downloads/askui") as agent:
            agent.act("Open example.com and download the sample PDF")
        ```
    """

    @telemetry.record_call(
        exclude={
            "reporters",
            "settings",
            "act_tools",
            "callbacks",
            "truncation_strategy",
            "secrets",
            "download_dir",
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
        secrets: list[Secret] | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        reporter = CompositeReporter(reporters=reporters)
        self.os = PlaywrightAgentOs(reporter, download_dir=download_dir)
        super().__init__(
            reporter=reporter,
            retry=retry,
            tools=self.get_default_tools() + (act_tools or []),
            agent_os=self.os,
            settings=settings,
            callbacks=callbacks,
            truncation_strategy=truncation_strategy,
            secrets=secrets,
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

    def wait_until_downloads_complete(self) -> list[Path]:
        """Block until all downloads started so far are fully saved to disk.

        Returns:
            list[Path]: Absolute paths of all downloads saved to `download_dir`
                so far this session.

        Raises:
            DownloadError: If one or more downloads could not be saved
                completely.
        """
        return self.os.wait_until_downloads_complete()

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
