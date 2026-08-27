import logging
import time
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Type, overload

from dotenv import load_dotenv
from PIL import Image as PILImage
from pydantic import ConfigDict, Field, validate_call
from typing_extensions import Self

from askui.agent_settings import AgentSettings
from askui.callbacks import ConversationCallback, ConversationStatisticsCallback
from askui.container import telemetry
from askui.locators.locators import Locator
from askui.models.shared.agent_message_param import MessageParam, TextBlockParam
from askui.models.shared.conversation import Conversation, Speakers
from askui.models.shared.secrets import Secret, SecretVault
from askui.models.shared.settings import (
    ActSettings,
    CacheFile,
    CacheWritingSettings,
    CachingSettings,
    GetSettings,
    LocateSettings,
)
from askui.models.shared.tools import Tool, ToolCollection
from askui.models.shared.truncation_strategies import TruncationStrategy
from askui.prompts.act_prompts import CACHE_USE_PROMPT, create_default_prompt
from askui.telemetry.otel import OtelSettings, setup_opentelemetry_tracing
from askui.tools.agent_os import ComputerAgentOS
from askui.tools.android.agent_os import AndroidAgentOs
from askui.tools.caching_tools import (
    InspectCacheMetadata,
    VerifyCacheExecution,
)
from askui.tools.get_tool import GetTool
from askui.tools.locate_tool import LocateTool
from askui.utils.annotation_writer import AnnotationWriter
from askui.utils.caching.cache_manager import (
    CacheManager,
    ensure_relative_cache_filename,
)
from askui.utils.caching.reporting_utils import report_cache_event
from askui.utils.image_utils import ImageSource
from askui.utils.source_utils import InputSource, load_image_source, load_source

from .models.exceptions import ElementNotFoundError, WaitUntilError
from .models.models import DetectedElement
from .models.types.geometry import Point, PointList
from .models.types.response_schemas import ResponseSchema
from .reporting import CompositeReporter, Reporter
from .retry import ConfigurableRetry, Retry
from .speaker import CacheExecutor

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        reporter: Reporter | None = None,
        retry: Retry | None = None,
        tools: list[Tool] | None = None,
        agent_os: ComputerAgentOS | AndroidAgentOs | None = None,
        settings: AgentSettings | None = None,
        callbacks: list[ConversationCallback] | None = None,
        truncation_strategy: TruncationStrategy | None = None,
        secrets: list[Secret] | None = None,
    ) -> None:
        load_dotenv()
        self._reporter: Reporter = reporter or CompositeReporter(reporters=None)
        self._agent_os = agent_os

        self._tools = tools or []

        # Secrets the agent may use but the LLM must never see. Real values are
        # substituted into tool inputs at execution time; placeholders are all the
        # model ever sees. Literal values are redacted from the LLM history and tool
        # outputs. See `askui.models.shared.secrets`.
        self._secret_vault = SecretVault(secrets)

        # Store settings and model providers
        _settings = settings or AgentSettings()
        self._vlm_provider = _settings.vlm_provider
        self._image_qa_provider = _settings.image_qa_provider
        self._detection_provider = _settings.detection_provider

        # Create conversation with speakers and model providers
        speakers = Speakers()
        _callbacks = list(callbacks or [])
        _callbacks.append(
            ConversationStatisticsCallback(
                reporter=self._reporter,
                pricing=self._vlm_provider.pricing,
            )
        )
        self._conversation = Conversation(
            speakers=speakers,
            vlm_provider=self._vlm_provider,
            image_qa_provider=self._image_qa_provider,
            detection_provider=self._detection_provider,
            reporter=self._reporter,
            truncation_strategy=truncation_strategy,
            callbacks=_callbacks,
        )

        # Provider-based tools
        self._get_tool: GetTool = GetTool(provider=_settings.image_qa_provider)
        self._locate_tool: LocateTool = LocateTool(
            provider=_settings.detection_provider
        )

        self._retry = retry or ConfigurableRetry(
            strategy="Exponential",
            base_delay=1000,
            retry_count=3,
            on_exception_types=(ElementNotFoundError,),
        )

        self.act_tool_collection = ToolCollection(tools=tools)
        if agent_os is not None:
            self.act_tool_collection.add_agent_os(agent_os)

        # Settings stored at agent level (mutable)
        self.act_settings = ActSettings()
        self.get_settings = GetSettings()
        self.locate_settings = LocateSettings()
        self.caching_settings = CachingSettings()

    @telemetry.record_call(
        exclude={"goal", "act_settings", "tools", "tracing_settings", "secrets"}
    )
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def act(
        self,
        goal: Annotated[str | list[MessageParam], Field(min_length=1)],
        act_settings: ActSettings | None = None,
        tools: list[Tool] | ToolCollection | None = None,
        caching_settings: CachingSettings | None = None,
        tracing_settings: OtelSettings | None = None,
        secrets: list[Secret] | None = None,
    ) -> None:
        """
        Instructs the agent to achieve a specified goal through autonomous actions.

        The agent will analyze the screen, determine necessary steps, and perform
        actions to accomplish the goal. This may include clicking, typing, scrolling,
        and other interface interactions.

        Args:
            goal (str | list[MessageParam]): A description of what the agent should
                achieve.
            act_settings (ActSettings | None, optional): Settings for this act
                execution. Overrides the agent's default settings if provided.
            act_model (ActModel | None, optional): Model to use for this act
                execution.
                Overrides the agent's default model if provided.
            tools (list[Tool] | ToolCollection | None, optional): The tools for the
                agent. Defaults to default tools depending on the selected model.
            caching_settings (CachingSettings | None, optional): The caching settings
                for the act execution. Controls recording and replaying of action
                sequences (trajectories). Available strategies: None (default, no
                caching), "record" (record actions to cache file), "execute" (replay
                from cached trajectories), "auto" (execute and record). Defaults to
                no caching.
            tracing_settings (OtelSettings | None, optional): The tracing settings
                for the act execution. Controls if and how traces are exported via
                Opentelemetry.
            secrets (list[Secret] | None, optional): Secrets available for this act
                execution, in addition to any defined on the agent. The model only ever
                sees the placeholder `<|secret|>NAME<|secret|>`; the real value is
                substituted into tool inputs at execution time and is never sent to the
                model. Per-call secrets override agent-level
                secrets with the same name. Defaults to `None`. Note: a secret typed
                into a visible field may still appear in screenshots sent to the model;
                on-screen secrets cannot currently be hidden.

        Returns:
            None

        Raises:
            AutomationError: If a tool raises an unfixable error that cannot be
                auto-corrected by the agent.
            MaxTokensExceededError: If the model reaches the maximum token limit
                defined in the agent settings.
            ModelRefusalError: If the model refuses to process the request.

        Example:
            Basic usage without caching:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.act("Open the settings menu")
                agent.act("Search for 'printer' in the search box")
                agent.act("Log in with username 'admin' and password '1234'")
            ```

            Recording actions to a cache file:
            ```python
            from askui import ComputerAgent
            from askui.models.shared.settings import CachingSettings

            with ComputerAgent() as agent:
                agent.act(
                    goal=(
                        "Fill out the login form with "
                        "username 'admin' and password 'secret123'"
                    ),
                    caching_settings=CachingSettings(
                        strategy="record",
                        cache_dir=".cache",
                        filename="login_flow.json"
                    )
                )
            ```

            Executing cached actions:
            ```python
            from askui import ComputerAgent
            from askui.models.shared.settings import CachingSettings

            with ComputerAgent() as agent:
                agent.act(
                    goal="Log in to the application",
                    caching_settings=CachingSettings(
                        strategy="execute",
                        cache_dir=".cache"
                    )
                )
                # Agent will automatically find and use "login_flow.json"
            ```

            Using both execute and record modes:
            ```python
            from askui import ComputerAgent
            from askui.models.shared.settings import CachingSettings

            with ComputerAgent() as agent:
                agent.act(
                    goal="Complete the checkout process",
                    caching_settings=CachingSettings(
                        strategy="auto",
                        cache_dir=".cache",
                        filename="checkout.json"
                    )
                )
                # Agent can use existing caches and will record new actions
            ```
        """
        # Merge agent-level and per-call secrets (per-call wins on name collision).
        active_vault = self._secret_vault.merge(SecretVault(secrets))

        goal_str = (
            goal
            if isinstance(goal, str)
            else "\n".join(msg.model_dump_json() for msg in goal)
        )
        # Redact any literal secret value the user may have placed in the goal before
        # it reaches the reporter/logs.
        redacted_goal_str = active_vault.redact(goal_str)
        self._reporter.add_message("User", f'act: "{redacted_goal_str}"')
        logger.debug(
            "Agent received instruction to act towards the goal '%s'", redacted_goal_str
        )
        messages: list[MessageParam] = (
            [MessageParam(role="user", content=goal)] if isinstance(goal, str) else goal
        )
        # Initial messages bypass Conversation._add_message, so redact them here to keep
        # literal secrets out of the history sent to the LLM.
        messages = [active_vault.redact_message(message) for message in messages]
        # Make the vault available for substitution (tools) and redaction (history).
        # The Conversation propagates it to the ToolCollection.
        self._conversation.secret_vault = active_vault
        # Deep-copy so caching-related mutations (e.g. injecting the CACHE_USE
        # prompt) do not accumulate on the Agent's persistent, reused settings
        # object and leak into subsequent act() calls.
        _act_settings = (act_settings or self.act_settings).model_copy(deep=True)

        _caching_settings: CachingSettings = caching_settings or self.caching_settings

        tools, cache_manager, cache_hint = self._patch_act_with_cache(
            _caching_settings, _act_settings, tools, goal_str
        )
        if cache_hint:
            messages = self._inject_cache_hint(messages, cache_hint)
        _tools = self._build_tools(tools)

        # setup opentelemetry for tracing
        if tracing_settings:
            setup_opentelemetry_tracing(tracing_settings)

        # Set toolbox on cache_manager for non-cacheable tool detection
        if cache_manager:
            cache_manager.set_toolbox(_tools)

        # Set cache_manager on conversation for recording
        self._conversation.cache_manager = cache_manager

        # Use conversation-based architecture for execution
        self._conversation.execute_conversation(
            messages=messages,
            tools=_tools,
            settings=_act_settings,
        )

    def _build_tools(self, tools: list[Tool] | ToolCollection | None) -> ToolCollection:
        # Build a fresh per-call collection copied from the agent's base tools so
        # that per-call additions (caching tools, switch_speaker, per-call tools)
        # do not accumulate on the persistent `act_tool_collection` across calls.
        # Otherwise a run-specific `VerifyCacheExecution` (wired to that run's
        # CacheExecutor/CacheManager) would linger and could persist a later,
        # unrelated run's result to the previous run's trajectory file.
        tool_collection = self.act_tool_collection + ToolCollection()
        if isinstance(tools, list):
            tool_collection.append_tool(*tools)
        if isinstance(tools, ToolCollection):
            tool_collection += tools
        return tool_collection

    def _resolve_secrets(self, text: str) -> str:
        """Substitute `<|secret|>NAME<|secret|>` placeholders with real values.

        Used by deterministic input methods (e.g. `type`) so callers/agents can pass a
        placeholder that resolves to the real value at the OS boundary.
        """
        resolved: str = self._secret_vault.substitute(text)
        return resolved

    def _redact_secrets(self, text: str) -> str:
        """Redact literal secret values to their placeholders (for reporting/logs)."""
        redacted: str = self._secret_vault.redact(text)
        return redacted

    def _patch_act_with_cache(
        self,
        caching_settings: CachingSettings,
        settings: ActSettings,
        tools: list[Tool] | ToolCollection | None,
        goal: str,
    ) -> tuple[list[Tool] | ToolCollection, CacheManager | None, str | None]:
        """Patch act settings and tools with caching functionality.

        In ``execute``/``auto`` modes the trajectory for the current test case is
        auto-detected from ``caching_settings.filename`` (no separate discovery
        tool call is required): if a usable trajectory exists, its details are
        returned as a ``cache_hint`` to be surfaced to the agent, and the
        ``CacheExecutor`` speaker plus verification tooling are wired up. In
        ``record``/``auto`` modes a cache manager is set up to record the run
        (in ``auto`` only when no usable trajectory was found, so an existing
        cache is never overwritten by a fresh recording).

        Args:
            caching_settings: The caching settings to apply
            settings: The act settings to modify
            tools: The tools list to extend with caching tools
            goal: The goal string for cache recording

        Returns:
            A tuple of ``(modified_tools, cache_manager, cache_hint)`` where
            ``cache_hint`` is an optional instruction to inject into the first
            user message.
        """
        caching_tools: list[Tool] = []
        cache_manager: CacheManager | None = None
        cache_hint: str | None = None

        # Remove any CacheExecutor registered by a previous act() call so it does
        # not leak (and get advertised via switch_speaker) into this run.
        self._conversation.speakers.remove_speaker("CacheExecutor")

        strategy = caching_settings.strategy
        filename = self._resolve_cache_filename(caching_settings)

        # Detect an existing trajectory for execute/auto modes.
        cache_file: CacheFile | None = None
        trajectory_path: Path | None = None
        if strategy in ("execute", "auto") and filename:
            trajectory_path = self._resolve_trajectory_path(
                caching_settings.cache_dir, filename
            )
            cache_file = self._read_trajectory_if_present(trajectory_path)

        # Decide whether to replay the detected trajectory. In auto mode an
        # invalid cache is re-recorded (self-heal) rather than replayed.
        execute_trajectory = cache_file is not None and (
            cache_file.metadata.is_valid or strategy == "execute"
        )
        should_record = strategy == "record" or (
            strategy == "auto" and not execute_trajectory
        )

        if execute_trajectory or should_record:
            cache_manager = CacheManager(reporter=self._reporter)

        # Setup execute mode: wire the CacheExecutor and verification tooling and
        # tell the agent (via the hint) exactly which trajectory to replay.
        if (
            execute_trajectory
            and cache_file is not None
            and trajectory_path is not None
        ):
            cache_executor = CacheExecutor(caching_settings.execution_settings)
            self._conversation.speakers.add_speaker(cache_executor)

            # switch_speaker tool is added automatically by
            # Conversation._setup_speaker_handoff
            caching_tools.extend(
                [
                    VerifyCacheExecution(
                        cache_executor=cache_executor,
                        cache_manager=cache_manager,
                        reporter=self._reporter,
                    ),
                    InspectCacheMetadata(),
                ]
            )
            if settings.messages.system is None:
                settings.messages.system = create_default_prompt()
            settings.messages.system.cache_use = CACHE_USE_PROMPT
            cache_hint = self._build_cache_execution_hint(trajectory_path, cache_file)

            validity = (
                "valid" if cache_file.metadata.is_valid else "INVALID (will try anyway)"
            )
            report_cache_event(
                self._reporter,
                f"Cache hit: replaying '{trajectory_path.name}' "
                f"({len(cache_file.trajectory)} steps, "
                f"{len(cache_file.cache_parameters)} parameter(s), {validity}).",
                log=logger,
            )
        else:
            cache_hint = self._report_cache_miss(strategy, filename)

        # Add caching tools to the tools list
        if isinstance(tools, list):
            tools = caching_tools + tools
        elif isinstance(tools, ToolCollection):
            tools.append_tool(*caching_tools)
        else:
            tools = caching_tools

        # Setup record mode: start recording the trajectory.
        if should_record and cache_manager is not None:
            cache_writer_settings = (
                caching_settings.writing_settings or CacheWritingSettings()
            )
            cache_manager.start_recording(
                cache_dir=caching_settings.cache_dir,
                file_name=filename,
                goal=goal,
                cache_writer_settings=cache_writer_settings,
                vlm_provider=self._vlm_provider,
            )

        return tools, cache_manager, cache_hint

    def _report_cache_miss(self, strategy: str | None, filename: str) -> str | None:
        """Report that no usable trajectory was found and return the miss hint.

        Returns the "no cached trajectory" hint in auto mode (so the agent knows
        it is recording for next time) and ``None`` otherwise.
        """
        if strategy == "auto":
            if filename:
                report_cache_event(
                    self._reporter,
                    f"No usable cached trajectory for '{filename}'; running "
                    "normally and recording this run for next time.",
                    log=logger,
                )
            return self._build_no_cache_hint()
        if strategy == "execute" and filename:
            report_cache_event(
                self._reporter,
                f"No usable cached trajectory for '{filename}'; running normally "
                "(execute mode does not record).",
                log=logger,
            )
        return None

    @staticmethod
    def _resolve_cache_filename(caching_settings: CachingSettings) -> str:
        """Resolve the trajectory filename, preferring the top-level setting."""
        if caching_settings.filename:
            return caching_settings.filename
        if (
            caching_settings.writing_settings
            and caching_settings.writing_settings.filename
        ):
            return caching_settings.writing_settings.filename
        return ""

    @staticmethod
    def _resolve_trajectory_path(cache_dir: str, filename: str) -> Path:
        """Build the full trajectory path, ensuring a ``.json`` suffix.

        The filename may include subdirectories but must stay within
        ``cache_dir`` (see `ensure_relative_cache_filename`).
        """
        ensure_relative_cache_filename(filename)
        name = filename if filename.endswith(".json") else f"{filename}.json"
        return Path(cache_dir) / name

    @staticmethod
    def _read_trajectory_if_present(trajectory_path: Path) -> "CacheFile | None":
        """Read a trajectory file if it exists and is readable, else ``None``."""
        if not trajectory_path.is_file():
            return None
        try:
            return CacheManager.read_cache_file(trajectory_path)
        except Exception:
            logger.exception(
                "Found trajectory %s but failed to read it; ignoring cache",
                trajectory_path,
            )
            return None

    @staticmethod
    def _build_cache_execution_hint(
        trajectory_path: Path, cache_file: CacheFile
    ) -> str:
        """Build the first-message hint describing an available cached trajectory."""
        path_str = str(trajectory_path)
        parameters = cache_file.cache_parameters
        if parameters:
            param_lines = "\n".join(
                f"  - {name}: {description}" for name, description in parameters.items()
            )
            param_block = (
                "This trajectory requires the following parameters (provide "
                f"values for ALL of them):\n{param_lines}"
            )
            example_params = ", ".join(f"'{name}': '<value>'" for name in parameters)
            switch_example = (
                "switch_speaker(speaker_name='CacheExecutor', speaker_context={"
                f"'trajectory_file': '{path_str}', "
                f"'parameter_values': {{{example_params}}}}})"
            )
        else:
            param_block = "This trajectory requires no parameters."
            switch_example = (
                "switch_speaker(speaker_name='CacheExecutor', speaker_context={"
                f"'trajectory_file': '{path_str}'}})"
            )

        validity_note = ""
        if not cache_file.metadata.is_valid:
            validity_note = (
                "\nNOTE: This cached trajectory is currently marked INVALID "
                f"(reason: {cache_file.metadata.invalidation_reason}). It may not "
                "replay correctly; execute with caution and verify the result "
                "carefully."
            )

        return (
            "<CACHED_TRAJECTORY_AVAILABLE>\n"
            "A cached trajectory for this test case is available and should be "
            "used to fast-forward execution instead of performing the steps "
            "manually.\n"
            f"- trajectory_file: {path_str}\n"
            f"{param_block}\n"
            "Before taking any other action, switch to the CacheExecutor speaker "
            "using the switch_speaker tool, for example:\n"
            f"{switch_example}"
            f"{validity_note}\n"
            "</CACHED_TRAJECTORY_AVAILABLE>"
        )

    @staticmethod
    def _build_no_cache_hint() -> str:
        """Build the first-message hint used in auto mode when no cache exists."""
        return (
            "<NO_CACHED_TRAJECTORY>\n"
            "No cached trajectory exists for this test case yet, so there is "
            "nothing to replay. Accomplish the goal normally; your actions are "
            "being recorded so they can be replayed on future runs.\n"
            "</NO_CACHED_TRAJECTORY>"
        )

    @staticmethod
    def _inject_cache_hint(
        messages: list[MessageParam], cache_hint: str
    ) -> list[MessageParam]:
        """Append the cache hint to the first user message.

        The hint is appended to (not inserted before) the first user message to
        avoid introducing consecutive same-role messages at the start of the
        history. If no user message exists (unusual), the messages are returned
        unchanged.
        """
        index = next(
            (i for i, m in enumerate(messages) if m.role == "user"),
            None,
        )
        if index is None:
            return messages
        target = messages[index]
        if isinstance(target.content, str):
            new_content: str | list[Any] = f"{target.content}\n\n{cache_hint}"
        else:
            new_content = [
                *target.content,
                TextBlockParam(type="text", text=cache_hint),
            ]
        messages[index] = target.model_copy(update={"content": new_content})
        return messages

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

    @telemetry.record_call(
        exclude={"query", "source", "response_schema", "get_settings"}
    )
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get(
        self,
        query: Annotated[str, Field(min_length=1)],
        response_schema: Type[ResponseSchema] | None = None,
        source: Optional[InputSource] = None,
        get_settings: GetSettings | None = None,
    ) -> ResponseSchema | str:
        """
        Retrieves information from an image or PDF based on the provided `query`.

        If no `source` is provided, a screenshot of the current screen is taken.

        Args:
            query (str): The query describing what information to retrieve.
            source (InputSource | None, optional): The source to extract information
                from. Can be a path to an image, PDF, or office document file,
                a PIL Image object or a data URL. Defaults to a screenshot of the
                current screen.
            response_schema (Type[ResponseSchema] | None, optional): A Pydantic model
                class that defines the response schema. If not provided, returns a
                string.

        Returns:
            ResponseSchema | str: The extracted information, `str` if no
                `response_schema` is provided.

        Raises:
            NotImplementedError: If PDF processing is not supported for the selected
                model.
            ValueError: If the `source` is not a valid PDF or image.

        Example:
            ```python
            from askui import ComputerAgent, ResponseSchemaBase
            from PIL import Image
            import json

            class UrlResponse(ResponseSchemaBase):
                url: str

            class NestedResponse(ResponseSchemaBase):
                nested: UrlResponse

            class LinkedListNode(ResponseSchemaBase):
                value: str
                next: "LinkedListNode | None"

            with ComputerAgent() as agent:
                # Get URL as string
                url = agent.get("What is the current url shown in the url bar?")

                # Get URL as Pydantic model from image at (relative) path
                response = agent.get(
                    "What is the current url shown in the url bar?",
                    response_schema=UrlResponse,
                    source="screenshot.png",
                )
                # Dump whole model
                print(response.model_dump_json(indent=2))
                # or
                response_json_dict = response.model_dump(mode="json")
                print(json.dumps(response_json_dict, indent=2))
                # or for regular dict
                response_dict = response.model_dump()
                print(response_dict["url"])

                # Get boolean response from PIL Image
                is_login_page = agent.get(
                    "Is this a login page?",
                    response_schema=bool,
                    source=Image.open("screenshot.png"),
                )
                print(is_login_page)

                # Get integer response
                input_count = agent.get(
                    "How many input fields are visible on this page?",
                    response_schema=int,
                )
                print(input_count)

                # Get float response
                design_rating = agent.get(
                    "Rate the page design quality from 0 to 1",
                    response_schema=float,
                )
                print(design_rating)

                # Get nested response
                nested = agent.get(
                    "Extract the URL and its metadata from the page",
                    response_schema=NestedResponse,
                )
                print(nested.nested.url)

                # Get recursive response
                linked_list = agent.get(
                    "Extract the breadcrumb navigation as a linked list",
                    response_schema=LinkedListNode,
                )
                current = linked_list
                while current:
                    print(current.value)
                    current = current.next

                # Get text from PDF
                text = agent.get(
                    "Extract all text from the PDF",
                    source="document.pdf",
                )
                print(text)
            ```
        """
        _get_settings = get_settings or self.get_settings

        if source is None and self._agent_os is None:
            error_msg = "A 'source' must be provided when the agent has no agent_os."
            raise RuntimeError(error_msg)
        _source = source or ImageSource(self._agent_os.screenshot())  # type: ignore[union-attr]

        # Load the source
        _loaded_source = (
            load_source(_source)
            if isinstance(_source, (str, Path, PILImage.Image))
            else _source
        )
        user_message_content = f'get: "{query}"' + (
            f" from '{_source}'" if isinstance(_source, (str, Path)) else ""
        )
        self._reporter.add_message(
            "User",
            user_message_content,
            image=_loaded_source.root
            if isinstance(_loaded_source, ImageSource)
            else None,
        )

        response = self._get_tool.run(
            query=query,
            source=_loaded_source,
            response_schema=response_schema,
            get_settings=_get_settings,
        )

        # Log the response
        message_content = (
            str(response)
            if isinstance(response, (str, bool, int, float))
            else response.model_dump()
        )
        self._reporter.add_message("Agent", message_content)
        return response

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def _locate(
        self,
        locator: str | Locator,
        screenshot: Optional[InputSource] = None,
        retry: Optional[Retry] = None,
        locate_settings: LocateSettings | None = None,
    ) -> PointList:
        _locate_settings = locate_settings or self.locate_settings

        def locate_with_screenshot() -> PointList:
            if screenshot is None and self._agent_os is None:
                error_msg = (
                    "A 'screenshot' must be provided when the agent has no agent_os."
                )
                raise RuntimeError(error_msg)
            _screenshot = load_image_source(
                self._agent_os.screenshot() if screenshot is None else screenshot  # type: ignore[union-attr]
            )
            return self._locate_tool.run(
                locator=locator,
                image=_screenshot,
                locate_settings=_locate_settings,
            )

        retry = retry or self._retry
        points = retry.attempt(locate_with_screenshot)
        self._reporter.add_message("LocateModel", f"locate {len(points)} elements")
        logger.debug("LocateModel locate: %d elements", len(points))
        return points

    @telemetry.record_call(exclude={"locator", "screenshot", "locate_settings"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def locate(
        self,
        locator: str | Locator,
        screenshot: Optional[InputSource] = None,
        locate_settings: LocateSettings | None = None,
    ) -> Point:
        """
        Locates the first matching UI element identified by the provided locator.

        Args:
            locator (str | Locator): The identifier or description of the element to
                locate.
            screenshot (InputSource | None, optional): The screenshot to use for
                locating the element. Can be a path to an image file, a PIL Image object
                or a data URL. If `None`, takes a screenshot of the currently
                selected display.
            locate_settings (LocateSettings | None, optional): Settings for this
                locate operation. If `None`, uses the agent's default locate settings.
            locate_model (LocateModel | None, optional): Model to use for this
                locate operation. If `None`, uses the agent's default locate model.

        Returns:
            Point: The coordinates of the element as a tuple (x, y).

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                point = agent.locate("Submit button")
                print(f"Element found at coordinates: {point}")
            ```
        """
        self._reporter.add_message("User", f"locate first matching element {locator}")
        logger.debug(
            "Agent received instruction to locate first matching element %s",
            locator,
        )
        return self._locate(
            locator=locator,
            screenshot=screenshot,
            locate_settings=locate_settings,
        )[0]

    @telemetry.record_call(exclude={"locator", "screenshot", "locate_settings"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def locate_all(
        self,
        locator: str | Locator,
        screenshot: Optional[InputSource] = None,
        locate_settings: LocateSettings | None = None,
    ) -> PointList:
        """
        Locates all matching UI elements identified by the provided locator.

        Note: Some LocateModels can only locate a single element. In this case, the
        returned list will have a length of 1.

        Args:
            locator (str | Locator): The identifier or description of the element to
                locate.
            screenshot (InputSource | None, optional): The screenshot to use for
                locating the element. Can be a path to an image file, a PIL Image object
                or a data URL. If `None`, takes a screenshot of the currently
                selected display.
            locate_settings (LocateSettings | None, optional): Settings for this
                locate operation. If `None`, uses the agent's default locate settings.
            locate_model (LocateModel | None, optional): Model to use for this
                locate operation. If `None`, uses the agent's default locate model.

        Returns:
            PointList: The coordinates of the elements as a list of tuples (x, y).

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                points = agent.locate_all("Submit button")
                print(f"Found {len(points)} elements at coordinates: {points}")
            ```
        """
        self._reporter.add_message("User", f"locate all matching UI elements {locator}")
        logger.debug(
            "Agent received instruction to locate all matching UI elements %s",
            locator,
        )
        return self._locate(
            locator=locator,
            screenshot=screenshot,
            locate_settings=locate_settings,
        )

    @telemetry.record_call(exclude={"screenshot"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def locate_all_elements(
        self,
        screenshot: Optional[InputSource] = None,
    ) -> list[DetectedElement]:
        """Locate all elements in the current screen using AskUI Models.

        Args:
            screenshot (InputSource | None, optional): The screenshot to use for
                locating the elements. Can be a path to an image file, a PIL Image
                object or a data URL. If `None`, takes a screenshot of the currently
                selected display.

        Returns:
            list[DetectedElement]: A list of detected elements

        Example:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                detected_elements = agent.locate_all_elements()
                print(f"Found {len(detected_elements)} elements: {detected_elements}")
            ```
        """
        if screenshot is None and self._agent_os is None:
            error_msg = (
                "A 'screenshot' must be provided when the agent has no agent_os."
            )
            raise RuntimeError(error_msg)
        _screenshot = load_image_source(
            self._agent_os.screenshot() if screenshot is None else screenshot  # type: ignore[union-attr]
        )
        return self._locate_tool.run_all(
            image=_screenshot,
            locate_settings=self.locate_settings,
        )

    @telemetry.record_call(exclude={"screenshot", "annotation_dir"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def annotate(
        self,
        screenshot: InputSource | None = None,
        annotation_dir: str = "annotations",
    ) -> None:
        """Annotate the screenshot with the detected elements.
        Creates an interactive HTML file with the detected elements
        and saves it to the annotation directory.
        The HTML file can be opened in a browser to see the annotated image.
        The user can hover over the elements to see their names and text value
        and click on the box to copy the text value to the clipboard.

        Args:
            screenshot (ImageSource | None, optional): The screenshot to annotate.
                If `None`, takes a screenshot of the currently selected display.
            annotation_dir (str): The directory to save the annotated
                image. Defaults to "annotations".

        Example Using ComputerAgent:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.annotate()
            ```

        Example Using AndroidAgent:
            ```python
            from askui import AndroidAgent

            with AndroidAgent() as agent:
                agent.annotate()
            ```

        Example Using ComputerAgent with custom screenshot and annotation directory:
            ```python
            from askui import ComputerAgent

            with ComputerAgent() as agent:
                agent.annotate(screenshot="screenshot.png", annotation_dir="htmls")
            ```
        """
        if screenshot is None:
            if self._agent_os is None:
                error_msg = (
                    "A 'screenshot' must be provided when the agent has no agent_os."
                )
                raise RuntimeError(error_msg)
            screenshot = self._agent_os.screenshot()

        self._reporter.add_message("User", "annotate screenshot with detected elements")
        detected_elements = self.locate_all_elements(
            screenshot=screenshot,
        )
        annotated_html = AnnotationWriter(
            image=screenshot,
            elements=detected_elements,
        ).save_to_dir(annotation_dir)
        self._reporter.add_message(
            "AnnotationWriter", f"annotated HTML file saved to '{annotated_html}'"
        )

    @telemetry.record_call(exclude={"until"})
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def wait(
        self,
        until: Annotated[float, Field(gt=0.0)] | str | Locator,
        retry_count: Optional[Annotated[int, Field(gt=0)]] = None,
        delay: Optional[Annotated[float, Field(gt=0.0)]] = None,
        until_condition: Literal["appear", "disappear"] = "appear",
    ) -> None:
        """
        Pauses execution or waits until a UI element appears or disappears.

        Args:
            until (float | str | Locator): If a float, pauses execution for the
                specified number of seconds (must be greater than 0.0). If a string
                or Locator, waits until the specified UI element appears or
                disappears on screen.
            retry_count (int | None): Number of retries when waiting for a UI
                element. Defaults to 3 if None.
            delay (int | None): Sleep duration in seconds between retries when
                waiting for a UI element. Defaults to 1 second if None.
            until_condition (Literal["appear", "disappear"]): The condition to wait
                until the element satisfies. Defaults to "appear".

        Raises:
            WaitUntilError: If the UI element is not found after all retries.

        Example:
            ```python
            from askui import ComputerAgent
            from askui.locators import loc

            with ComputerAgent() as agent:
                # Wait for a specific duration
                agent.wait(5)  # Pauses execution for 5 seconds
                agent.wait(0.5)  # Pauses execution for 500 milliseconds

                # Wait for a UI element to appear
                agent.wait("Submit button", retry_count=5, delay=2)
                agent.wait("Login form")  # Uses default retries and sleep time
                agent.wait(loc.Text("Password"))  # Uses default retries and sleep time

                # Wait for a UI element to disappear
                agent.wait("Loading spinner", until_condition="disappear")
            ```
        """
        if isinstance(until, float) or isinstance(until, int):
            self._reporter.add_message("User", f"wait {until} seconds")
            time.sleep(until)
            return

        self._reporter.add_message(
            "User", f"wait for element '{until}' to {until_condition}"
        )
        retry_count = retry_count if retry_count is not None else 3
        delay = delay if delay is not None else 1

        if until_condition == "appear":
            self._wait_for_appear(until, retry_count, delay)
        else:
            self._wait_for_disappear(until, retry_count, delay)

    def _wait_for_appear(
        self,
        locator: str | Locator,
        retry_count: int,
        delay: float,
    ) -> None:
        """Wait for an element to appear on screen."""
        try:
            self._locate(
                locator,
                retry=ConfigurableRetry(
                    strategy="Fixed",
                    base_delay=int(delay * 1000),
                    retry_count=retry_count,
                    on_exception_types=(ElementNotFoundError,),
                ),
            )
            self._reporter.add_message(
                "Agent", f"element '{locator}' appeared successfully"
            )
        except ElementNotFoundError as e:
            self._reporter.add_message(
                "Agent",
                f"element '{locator}' failed to appear after {retry_count} retries",
            )
            raise WaitUntilError(
                e.locator, e.locator_serialized, retry_count, delay, "appear"
            ) from e

    def _wait_for_disappear(
        self,
        locator: str | Locator,
        retry_count: int,
        delay: float,
    ) -> None:
        """Wait for an element to disappear from screen."""
        for i in range(retry_count):
            try:
                self._locate(
                    locator,
                    retry=ConfigurableRetry(
                        strategy="Fixed",
                        base_delay=int(delay * 1000),
                        retry_count=1,
                        on_exception_types=(),
                    ),
                )
                logger.debug(
                    "Element still present, retrying... %d/%d", i + 1, retry_count
                )
                time.sleep(delay)
            except ElementNotFoundError:  # noqa: PERF203
                self._reporter.add_message(
                    "Agent", f"element '{locator}' disappeared successfully"
                )
                return

        self._reporter.add_message(
            "Agent",
            f"element '{locator}' failed to disappear after {retry_count} retries",
        )
        raise WaitUntilError(locator, str(locator), retry_count, delay, "disappear")

    @telemetry.record_call()
    def close(self) -> None:
        if self._agent_os is not None:
            self._agent_os.disconnect()
        self._reporter.generate()

    @telemetry.record_call()
    def open(self) -> None:
        if self._agent_os is not None:
            self._agent_os.connect()

    @telemetry.record_call()
    def __enter__(self) -> Self:
        self.open()
        return self

    @telemetry.record_call(exclude={"exc_value", "traceback"})
    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def get_default_tools() -> list[Tool]:
        return []
