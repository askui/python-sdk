from typing import Annotated, Optional, Type, overload

from pydantic import Field

from askui.agent_base import Agent
from askui.agent_settings import AgentSettings
from askui.android_agent import AndroidAgent
from askui.computer_agent import ComputerAgent
from askui.locators.locators import Locator
from askui.models.shared.secrets import Secret
from askui.models.shared.settings import GetSettings, LocateSettings
from askui.models.shared.tools import Tool
from askui.models.types.geometry import Point
from askui.models.types.response_schemas import ResponseSchema
from askui.prompts.act_prompts import create_multidevice_agent_prompt
from askui.reporting import CompositeReporter, Reporter
from askui.retry import Retry
from askui.utils.source_utils import InputSource


class MultiDeviceAgent(Agent):
    """
    Multi device agent that combines a computer and an Android agent.
    It can be used to perform actions on both devices simultaneously.

    Args:
        display (int, optional): The display number for computer screen
            interactions. Defaults to `1`.
        reporters (list[Reporter] | None, optional): List of reporter instances.
        tools (AgentToolbox | None, optional): Not supported; use `act_tools`.
        retry (Retry | None, optional): Retry instance for failed actions.
        act_tools (list[Tool] | None, optional): Additional tools for `act()`.
        android_device_sn (str | None, optional): Android device serial number
            to select on open.
        secrets (list[Secret] | None, optional): Sensitive values the agent may use but
            the LLM must never see. Applied to `act()` and to the composed
            `computer`/`android` agents. The model only sees the placeholder
            `<|secret|>NAME<|secret|>`; the real value is substituted at execution time.

    Example:
        ```python
        from askui import MultiDeviceAgent

        with MultiDeviceAgent(android_device_sn="emulator-5554") as agent:
            agent.computer.click("Start")
            agent.android.tap("OK")
            agent.act("Fill the form on the phone and submit from the desktop")
        ```
    """

    def __init__(
        self,
        desktop_display: Annotated[int, Field(ge=1)] = 1,
        android_device_sn: str | int = 0,
        reporters: list[Reporter] | None = None,
        retry: Retry | None = None,
        act_tools: list[Tool] | None = None,
        settings: AgentSettings | None = None,
        secrets: list[Secret] | None = None,
    ) -> None:
        reporter = CompositeReporter(reporters=reporters)

        # Initialize the base agent
        super().__init__(
            reporter=reporter,
            retry=retry,
            settings=settings,
            secrets=secrets,
        )

        # Initialize the computer agent (secrets also passed so that deterministic
        # `agent.computer.*` calls resolve placeholders too).
        self._computer_agent: ComputerAgent = ComputerAgent(
            display=desktop_display,
            reporters=[reporter],
            settings=settings,
            secrets=secrets,
        )

        # Initialize the Android agent
        self._android_agent: AndroidAgent = AndroidAgent(
            device=android_device_sn,
            reporters=[reporter],
            settings=settings,
            secrets=secrets,
        )

        # Combine the tool collections of the computer and Android agents
        self.act_tool_collection = (
            self._computer_agent.act_tool_collection
            + self._android_agent.act_tool_collection
        )

        self.act_tool_collection.append_tool(*(act_tools or []))

        self.act_settings.messages.system = create_multidevice_agent_prompt()

    @property
    def computer(self) -> ComputerAgent:
        """The composed computer agent."""
        return self._computer_agent

    @property
    def android(self) -> AndroidAgent:
        """The composed Android agent."""
        return self._android_agent

    @overload
    def get(
        self,
        query: Annotated[str, Field(min_length=1)],
        response_schema: None = None,
        source: Optional[InputSource] = None,
        get_settings: GetSettings | None = None,
    ) -> str: ...
    @overload
    def get(
        self,
        query: Annotated[str, Field(min_length=1)],
        response_schema: Type[ResponseSchema],
        source: Optional[InputSource] = None,
        get_settings: GetSettings | None = None,
    ) -> ResponseSchema: ...

    def get(
        self,
        query: Annotated[str, Field(min_length=1)],
        response_schema: Type[ResponseSchema] | None = None,
        source: Optional[InputSource] = None,
        get_settings: GetSettings | None = None,
    ) -> ResponseSchema | str:
        """Not supported on `MultiDeviceAgent`.

        Use `agent.computer.get()` or `agent.android.get()` instead.

        Raises:
            NotImplementedError: Always.
        """
        error_msg = (
            "MultiDeviceAgent does not support get() directly."
            " Use agent.computer.get() or agent.android.get()"
            " instead."
        )
        raise NotImplementedError(error_msg)

    def locate(
        self,
        locator: str | Locator,
        screenshot: Optional[InputSource] = None,
        locate_settings: LocateSettings | None = None,
    ) -> Point:
        """Not supported on `MultiDeviceAgent`.

        Use `agent.computer.locate()` or `agent.android.locate()`
        instead.

        Raises:
            NotImplementedError: Always.
        """
        error_msg = (
            "MultiDeviceAgent does not support locate() directly."
            " Use agent.computer.locate() or"
            " agent.android.locate() instead."
        )
        raise NotImplementedError(error_msg)

    def close(self) -> None:
        self._computer_agent.act_agent_os_facade.disconnect()
        self._android_agent.act_agent_os_facade.disconnect()
        super().close()

    def open(self) -> None:
        self._computer_agent.open()
        self._android_agent.open()
        super().open()
