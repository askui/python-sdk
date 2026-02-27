import logging
import re
import types
import uuid
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import wraps
from typing import Any, Literal, Protocol, Type

import jsonref
import mcp
from asyncer import syncify
from fastmcp.client.client import CallToolResult, ProgressHandler
from fastmcp.tools import Tool as FastMcpTool
from fastmcp.utilities.types import Image as FastMcpImage
from mcp import Tool as McpTool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing_extensions import Self

from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    ContentBlockParam,
    ImageBlockParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.tools import AgentOs
from askui.tools.android.agent_os import AndroidAgentOs
from askui.utils.image_utils import ImageSource

logger = logging.getLogger(__name__)

PrimitiveToolCallResult = Image.Image | None | str | BaseModel

ToolCallResult = (
    PrimitiveToolCallResult
    | list[PrimitiveToolCallResult]
    | tuple[PrimitiveToolCallResult, ...]
    | CallToolResult
)


IMAGE_MEDIA_TYPES_SUPPORTED: list[
    Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
] = ["image/jpeg", "image/png", "image/gif", "image/webp"]


def _convert_to_content(
    result: ToolCallResult,
) -> list[TextBlockParam | ImageBlockParam]:
    if result is None:
        return []

    if isinstance(result, CallToolResult):
        _result: list[TextBlockParam | ImageBlockParam] = []
        for block in result.content:
            match block.type:
                case "text":
                    _result.append(TextBlockParam(text=block.text))
                case "image":
                    media_type = block.mimeType
                    if media_type not in IMAGE_MEDIA_TYPES_SUPPORTED:
                        logger.warning(
                            "Unsupported image media type",
                            extra={"media_type": media_type},
                        )
                        continue
                    _result.append(
                        ImageBlockParam(
                            source=Base64ImageSourceParam(
                                media_type=media_type,
                                data=block.data,
                            )
                        )
                    )
                case _:
                    logger.warning(
                        "Unsupported block type",
                        extra={"block_type": block.type},
                    )
        return _result

    if isinstance(result, str):
        return [TextBlockParam(text=result)]

    if isinstance(result, list | tuple):
        return [
            item
            for sublist in [_convert_to_content(item) for item in result]
            for item in sublist
        ]

    if isinstance(result, BaseModel):
        return [TextBlockParam(text=result.model_dump_json())]

    return [
        ImageBlockParam(
            source=Base64ImageSourceParam(
                media_type="image/png",
                data=ImageSource(result).to_base64(),
            )
        )
    ]


def _default_input_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "required": []}


def _convert_to_mcp_content(
    result: Any,
) -> Any:
    if isinstance(result, tuple):
        return tuple(_convert_to_mcp_content(item) for item in result)

    if isinstance(result, list):
        return [_convert_to_mcp_content(item) for item in result]

    if isinstance(result, Image.Image):
        src = ImageSource(result)
        return FastMcpImage(data=src.to_bytes(), format="png").to_image_content()

    return result


PLAYWRIGHT_TOOL_PREFIX = "browser_"


def _is_playwright_error(
    param: ToolUseBlockParam,
    error: Exception,  # noqa: ARG001
) -> bool:
    if param.name.startswith(PLAYWRIGHT_TOOL_PREFIX):
        return True
    return False


def _create_tool_result_block_param_for_playwright_error(
    param: ToolUseBlockParam, error: Exception
) -> ToolResultBlockParam:
    lines = str(error).split("\n")
    line_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.startswith('Run "npx playwright install '):
            line_idx = idx
            break

    if line_idx is not None:
        lines[line_idx] = "Download and install the browser to continue."
    return ToolResultBlockParam(
        content="\n\n".join(lines),
        is_error=True,
        tool_use_id=param.id,
    )


class Tool(BaseModel, ABC):
    model_config = ConfigDict(
        validate_by_alias=True,
    )

    base_name: str = Field(alias="name", description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: dict[str, Any] = Field(
        default_factory=_default_input_schema,
        description="JSON schema for tool parameters",
    )
    required_tags: list[str] = Field(
        description="Tags required for the tool", default=[]
    )

    _unique_id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ToolCallResult:
        """Executes the tool with the given arguments."""
        error_msg = "Tool subclasses must implement __call__ method"
        raise NotImplementedError(error_msg)

    @property
    def name(self) -> str:
        """Returns the unique name for this tool instance."""
        name_parts = [self.base_name]
        if len(self.required_tags) > 0:
            name_parts.append(f"tags_{'_'.join(self.required_tags)}")
        name_parts.append(self._unique_id)
        name = "_".join(name_parts)
        # Ensure name matches pattern ^[a-zA-Z0-9_-]$
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Ensure name is not longer than 64 characters
        return name[:64]

    @name.setter
    def name(self, value: str) -> None:
        """Sets the base name of the tool."""
        self.base_name = value

    is_cacheable: bool = Field(
        default=False,
        description=(
            "Whether this tool's actions can be cached and replayed. "
            "False by default. Set to True for tools that produce deterministic "
            "results and where side-effects during replay are unlikely or tolerable."
        ),
    )

    def to_params(
        self,
    ) -> ToolParam:
        return ToolParam(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def to_mcp_tool(
        self, tags: set[str], name_prefix: str | None = None
    ) -> FastMcpTool:
        """Convert the AskUI tool to an MCP tool."""
        tool_call = self.__call__

        @wraps(tool_call)
        def wrapped_tool_call(*args: Any, **kwargs: Any) -> Any:
            return _convert_to_mcp_content(tool_call(*args, **kwargs))

        tool_name = self.name
        if name_prefix is not None:
            tool_name = f"{name_prefix}{tool_name}"

        return FastMcpTool.from_function(
            wrapped_tool_call,
            name=tool_name,
            description=self.description,
            tags=tags,
        )


class ToolWithAgentOS(Tool):
    """Tool base class  that has an AgentOs available."""

    def __init__(
        self,
        required_tags: list[str],
        agent_os: AgentOs | AndroidAgentOs | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs, required_tags=required_tags)
        self._agent_os: AgentOs | AndroidAgentOs | None = agent_os

    @property
    def agent_os(self) -> AgentOs | AndroidAgentOs:
        """Get the agent OS.

        Returns:
            AgentOs | AndroidAgentOs: The agent OS instance.
        """
        if self._agent_os is None:
            msg = (
                "Agent OS is not initialized. "
                "Call `agent_os = ...` or initialize the tool with an "
                "agent_os parameter."
            )
            raise RuntimeError(msg)
        return self._agent_os

    @agent_os.setter
    def agent_os(self, agent_os: AgentOs | AndroidAgentOs) -> None:
        self._agent_os = agent_os

    def is_agent_os_initialized(self) -> bool:
        """Check if the agent OS is initialized."""
        return self._agent_os is not None


class AgentException(Exception):
    """
    Exception raised by the agent.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class McpClientProtocol(Protocol):
    async def list_tools(self) -> list[mcp.types.Tool]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: timedelta | float | None = None,  # noqa: ASYNC109
        progress_handler: ProgressHandler | None = None,
        raise_on_error: bool = True,
    ) -> CallToolResult: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...


def _replace_refs(tool_name: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    try:
        return jsonref.replace_refs(  # type: ignore[no-any-return]
            input_schema,
            lazy_load=False,
            proxies=False,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to replace refs for tool",
            extra={
                "tool_name": tool_name,
                "input_schema": input_schema,
            },
        )
        return input_schema


class ToolCollection:
    """A collection of tools.

    Use for dispatching tool calls

    **Important**: Tools must have unique names. A tool with the same name as a tool
    added before will override the tool added before.


    Vision:
    - Could be used for parallelizing tool calls configurable through init arg
    - Could be used for raising on an exception
      (instead of just returning `ContentBlockParam`)
      within tool call or doing tool call or if tool is not found

    Args:
        tools (list[Tool] | None, optional): The tools to add to the collection.
            Defaults to `None`.
        mcp_client (McpClientProtocol | None, optional): The client to use for
            the tools. Defaults to `None`.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        mcp_client: McpClientProtocol | None = None,
        include: set[str] | None = None,
        agent_os_list: list[AgentOs | AndroidAgentOs] | None = None,
    ) -> None:
        self._mcp_client = mcp_client
        self._include = include
        self._agent_os_list: list[AgentOs | AndroidAgentOs] = []
        self._tools: list[Tool] = tools or []
        if agent_os_list:
            for agent_os in agent_os_list:
                self.add_agent_os(agent_os)

    def add_agent_os(self, agent_os: AgentOs | AndroidAgentOs) -> None:
        """Add an agent OS to the collection.

        Args:
            agent_os (AgentOs | AndroidAgentOs): The agent OS instance to add.
        """
        self._agent_os_list.append(agent_os)

    def retrieve_tool_beta_flags(self) -> list[str]:
        result: set[str] = set()
        for tool in self._get_mcp_tools().values():
            beta_flags = (tool.meta or {}).get("betas", [])
            if not isinstance(beta_flags, list):
                continue
            for beta_flag in beta_flags:
                if not isinstance(beta_flag, str):
                    continue
                result.add(beta_flag)
        return list(result)

    def to_params(self) -> list[ToolParam]:
        tool_map = {
            **self._get_mcp_tool_params(),
            **{
                tool_name: tool.to_params() for tool_name, tool in self.tool_map.items()
            },
        }
        filtered_tool_map = {
            tool_name: tool
            for tool_name, tool in tool_map.items()
            if self._include is None or tool_name in self._include
        }
        result = list(filtered_tool_map.values())
        if result:
            result[-1]["cache_control"] = CacheControlEphemeralParam(
                type="ephemeral",
            )
        return result

    def _get_mcp_tool_params(self) -> dict[str, ToolParam]:
        if not self._mcp_client:
            return {}
        mcp_tools = self._get_mcp_tools()
        result: dict[str, ToolParam] = {}
        for tool_name, tool in mcp_tools.items():
            if params := (tool.meta or {}).get("params"):
                # validation missing
                result[tool_name] = params
                continue
            result[tool_name] = ToolParam(
                name=tool_name,
                description=tool.description or "",
                input_schema=_replace_refs(tool_name, tool.inputSchema),
            )
        return result

    def append_tool(self, *tools: Tool) -> None:
        """Append a tool to the collection."""
        self._tools.extend(tools)

    def reset_tools(self, tools: list[Tool] | None = None) -> None:
        """Reset the tools in the collection with new tools."""
        self._tools = tools or []

    def get_agent_os_by_tags(self, tags: list[str]) -> AgentOs | AndroidAgentOs:
        """Get an agent OS by tags."""
        for agent_os in self._agent_os_list:
            if all(tag in agent_os.tags for tag in tags):
                return agent_os
        msg = f"Agent OS with tags [{', '.join(tags)}] not found"
        raise ValueError(msg)

    def _initialize_tools(self) -> None:
        """Initialize the tools."""
        for tool in self._tools:
            if isinstance(tool, ToolWithAgentOS) and not tool.is_agent_os_initialized():
                agent_os = self.get_agent_os_by_tags(tool.required_tags)
                tool.agent_os = agent_os

    @property
    def tool_map(self) -> dict[str, Tool]:
        """Get the tool map."""
        self._initialize_tools()
        return {tool.name: tool for tool in self._tools}

    def run(
        self, tool_use_block_params: list[ToolUseBlockParam]
    ) -> list[ContentBlockParam]:
        return [
            self._run_tool(tool_use_block_param)
            for tool_use_block_param in tool_use_block_params
        ]

    def _run_tool(
        self, tool_use_block_param: ToolUseBlockParam
    ) -> ToolResultBlockParam:
        tool = self.tool_map.get(tool_use_block_param.name)
        if tool:
            return self._run_regular_tool(tool_use_block_param, tool)
        mcp_tool = self._get_mcp_tools().get(tool_use_block_param.name)
        if mcp_tool:
            return self._run_mcp_tool(tool_use_block_param)
        # Fallback: try prefix matching (for cached trajectories with different UUIDs)
        tool = self.find_tool_by_prefix(tool_use_block_param.name)
        if tool:
            return self._run_regular_tool(tool_use_block_param, tool)
        msg = f"no matching tool found with name {tool_use_block_param.name}"
        logger.error(msg)
        return ToolResultBlockParam(
            content=f"Tool not found: {tool_use_block_param.name}",
            is_error=True,
            tool_use_id=tool_use_block_param.id,
        )

    def find_tool_by_prefix(self, cached_name: str) -> Tool | None:
        """Find a tool by matching name prefix (without UUID suffix).

        Tool names have format: {base_name}_tags_{tags}_{uuid} or {base_name}_{uuid}
        This method strips the UUID suffix and matches by the remaining prefix.

        This is useful for cached trajectories where tool names may have different
        UUIDs than the current session.

        Args:
            cached_name: Tool name from cached trajectory (may have different UUID)

        Returns:
            Matching Tool if found, None otherwise
        """
        # Extract prefix by removing trailing UUID (pattern: _xxxxxxxx-xxxx-...)
        # UUIDs start with 8 hex chars after an underscore
        uuid_pattern = re.compile(r"_[0-9a-f]{8}-[0-9a-f]{4}.*$", re.IGNORECASE)
        cached_prefix = uuid_pattern.sub("", cached_name)

        if not cached_prefix or cached_prefix == cached_name:
            # No UUID found or name unchanged, can't match by prefix
            return None

        # Find a tool whose name starts with the same prefix
        for tool_name, tool in self.tool_map.items():
            tool_prefix = uuid_pattern.sub("", tool_name)
            if tool_prefix == cached_prefix:
                return tool

        return None

    async def _list_mcp_tools(self, mcp_client: McpClientProtocol) -> list[McpTool]:
        async with mcp_client:
            return await mcp_client.list_tools()

    def _get_mcp_tools(self) -> dict[str, McpTool]:
        """Get cached MCP tools or fetch them if not cached."""
        try:
            if not self._mcp_client:
                return {}
            list_mcp_tools_sync = syncify(self._list_mcp_tools, raise_sync_error=False)
            tools_list = list_mcp_tools_sync(self._mcp_client)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to list MCP tools",
            )
            return {}
        else:
            return {tool.name: tool for tool in tools_list}

    def _run_regular_tool(
        self,
        tool_use_block_param: ToolUseBlockParam,
        tool: Tool,
    ) -> ToolResultBlockParam:
        try:
            tool_result: ToolCallResult = tool(**tool_use_block_param.input)  # type: ignore
            return ToolResultBlockParam(
                content=_convert_to_content(tool_result),
                tool_use_id=tool_use_block_param.id,
            )
        except AgentException:
            raise
        except Exception as e:  # noqa: BLE001
            error_message = getattr(e, "message", str(e))
            logger.warning(
                "Tool failed",
                extra={"tool_name": tool_use_block_param.name, "error": error_message},
            )
            logger.warning("%s - %s", tool_use_block_param.name, error_message)
            return ToolResultBlockParam(
                content=f"Tool raised an unexpected error: {error_message}",
                is_error=True,
                tool_use_id=tool_use_block_param.id,
            )

    async def _call_mcp_tool(
        self,
        mcp_client: McpClientProtocol,
        tool_use_block_param: ToolUseBlockParam,
    ) -> ToolCallResult:
        async with mcp_client:
            return await mcp_client.call_tool(
                tool_use_block_param.name,
                tool_use_block_param.input,  # type: ignore[arg-type]
            )

    def _run_mcp_tool(
        self,
        tool_use_block_param: ToolUseBlockParam,
    ) -> ToolResultBlockParam:
        """Run an MCP tool using the client."""
        if not self._mcp_client:
            return ToolResultBlockParam(
                content="MCP client not available",
                is_error=True,
                tool_use_id=tool_use_block_param.id,
            )
        try:
            call_mcp_tool_sync = syncify(self._call_mcp_tool, raise_sync_error=False)
            result = call_mcp_tool_sync(self._mcp_client, tool_use_block_param)
            return ToolResultBlockParam(
                content=_convert_to_content(result),
                tool_use_id=tool_use_block_param.id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MCP tool failed",
                exc_info=True,
                extra={"tool_name": tool_use_block_param.name, "error": str(e)},
            )
            if _is_playwright_error(tool_use_block_param, e):
                return _create_tool_result_block_param_for_playwright_error(
                    tool_use_block_param,
                    e,
                )
            return ToolResultBlockParam(
                content=str(e),
                is_error=True,
                tool_use_id=tool_use_block_param.id,
            )

    def __add__(self, other: "ToolCollection") -> "ToolCollection":
        return ToolCollection(
            tools=self._tools + other._tools,
            mcp_client=other._mcp_client or self._mcp_client,
            agent_os_list=self._agent_os_list + other._agent_os_list,
        )
