"""OpenAIResponsesApi — MessagesApi for the OpenAI Responses API.

The chat completions endpoint rejects function tools with any
``reasoning_effort`` except ``"none"`` on current reasoning models. The
Responses API is OpenAI's supported path for reasoning + function calling,
so this backend exists to run agentic tool loops WITH reasoning enabled.

Statelessness: requests are sent with ``store=False`` and
``include=["reasoning.encrypted_content"]``. Each returned reasoning item is
preserved verbatim (as JSON in a `BetaRedactedThinkingBlock`) and re-emitted
on the next turn — reasoning models require their reasoning items to
accompany the function calls they produced, otherwise the API rejects the
request.
"""

import json
import logging
from typing import Any

from openai import OpenAI
from openai.types.responses import Response
from typing_extensions import override

from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    BetaRedactedThinkingBlock,
    BetaThinkingBlock,
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    StopReason,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UsageParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.pdf_utils import DEFAULT_PDF_FILENAME

logger = logging.getLogger(__name__)

#: Marker key inside ``ToolUseBlockParam.extra_content`` holding the raw
#: Responses ``function_call`` item, so it can be re-sent verbatim.
_RESPONSES_ITEM_KEY = "openai_responses_item"


def _image_block_to_part(block: ImageBlockParam) -> dict[str, Any]:
    if isinstance(block.source, Base64ImageSourceParam):
        url = f"data:{block.source.media_type};base64,{block.source.data}"
    else:
        url = block.source.url
    return {"type": "input_image", "image_url": url}


def _document_block_to_part(block: DocumentBlockParam) -> dict[str, Any]:
    data_url = f"data:{block.source.media_type};base64,{block.source.data}"
    return {
        "type": "input_file",
        "filename": block.title or DEFAULT_PDF_FILENAME,
        "file_data": data_url,
    }


def _user_block_to_part(block: ContentBlockParam) -> dict[str, Any] | None:
    if isinstance(block, TextBlockParam):
        return {"type": "input_text", "text": block.text}
    if isinstance(block, ImageBlockParam):
        return _image_block_to_part(block)
    if isinstance(block, DocumentBlockParam):
        return _document_block_to_part(block)
    return None


def _serialize_tool_result_content(
    content: str | list[TextBlockParam | ImageBlockParam | DocumentBlockParam],
) -> tuple[str, list[dict[str, Any]]]:
    """Split a tool result into its string output and media parts.

    ``function_call_output`` only accepts string output; images/documents are
    returned separately to be appended as a follow-up ``user`` message.
    """
    if isinstance(content, str):
        return content, []
    text_parts: list[str] = []
    media_parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlockParam):
            text_parts.append(block.text)
        elif isinstance(block, ImageBlockParam):
            media_parts.append(_image_block_to_part(block))
        elif isinstance(block, DocumentBlockParam):
            media_parts.append(_document_block_to_part(block))
    return "\n".join(text_parts), media_parts


def _convert_assistant_blocks(
    blocks: list[ContentBlockParam],
    result: list[dict[str, Any]],
) -> None:
    """Convert assistant content blocks to Responses input items.

    Reasoning items round-trip verbatim from `BetaRedactedThinkingBlock`
    (order-preserved — a reasoning item must precede the function call it
    belongs to). `BetaThinkingBlock` carries only the human-readable summary
    and is skipped on resend.
    """
    text_parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlockParam):
            text_parts.append(block.text)
        elif isinstance(block, BetaRedactedThinkingBlock):
            try:
                result.append(json.loads(block.data))
            except json.JSONDecodeError:
                logger.warning("Dropping malformed reasoning item on resend")
        elif isinstance(block, ToolUseBlockParam):
            raw = (block.extra_content or {}).get(_RESPONSES_ITEM_KEY)
            if isinstance(raw, dict):
                result.append(raw)
            else:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    }
                )
    if text_parts:
        result.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "\n".join(text_parts)}
                ],
            }
        )


def _convert_user_blocks(
    blocks: list[ContentBlockParam],
    result: list[dict[str, Any]],
) -> None:
    tool_result_media: list[dict[str, Any]] = []
    content_parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, ToolResultBlockParam):
            output, media = _serialize_tool_result_content(block.content)
            tool_result_media.extend(media)
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": output,
                }
            )
        else:
            part = _user_block_to_part(block)
            if part is not None:
                content_parts.append(part)
    if content_parts:
        result.append({"role": "user", "content": content_parts})
    if tool_result_media:
        result.append({"role": "user", "content": tool_result_media})


def _to_responses_input(messages: list[MessageParam]) -> list[dict[str, Any]]:
    """Convert internal ``MessageParam`` history to Responses ``input`` items."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message.content, str):
            result.append({"role": message.role, "content": message.content})
            continue
        if message.role == "assistant":
            _convert_assistant_blocks(message.content, result)
        else:
            _convert_user_blocks(message.content, result)
    return result


_TOOL_CHOICE_TYPE_MAP: dict[str, str] = {
    "auto": "auto",
    "any": "required",
    "none": "none",
}


def _to_responses_tool_choice(
    tool_choice: ToolChoiceParam,
) -> str | dict[str, Any] | None:
    """Map internal (Anthropic-style) ``tool_choice`` to the Responses format.

    Unlike chat completions, a forced function is flat:
    ``{"type": "function", "name": N}`` (no nested ``function`` object).
    """
    choice_type = tool_choice.get("type")
    if choice_type == "tool":
        return {"type": "function", "name": tool_choice["name"]}
    mapped = _TOOL_CHOICE_TYPE_MAP.get(choice_type) if choice_type else None
    if mapped is None:
        logger.warning(
            "Unsupported tool_choice for the Responses API, omitting",
            extra={"tool_choice": tool_choice},
        )
    return mapped


def _to_responses_tools(tools: ToolCollection) -> list[dict[str, Any]]:
    """Convert a `ToolCollection` to Responses function-tool format (flat)."""
    result: list[dict[str, Any]] = []
    for tool_param in tools.to_params():
        schema = dict(tool_param.get("input_schema", {}))
        schema.pop("cache_control", None)
        tool: dict[str, Any] = {
            "type": "function",
            "name": tool_param["name"],
            "parameters": schema,
            "strict": False,
        }
        if "description" in tool_param:
            tool["description"] = tool_param["description"]
        result.append(tool)
    return result


def _to_reasoning_param(thinking: ThinkingConfigParam) -> dict[str, Any] | None:
    """Map an internal ``thinking`` config to the Responses ``reasoning`` param.

    An explicit ``"effort"`` key passes through (with auto summaries);
    ``{"type": "disabled"}`` becomes effort ``"none"`` (no summary — the API
    rejects summaries with reasoning off); ``"adaptive"``/``"enabled"`` keep
    the model's default effort and request auto summaries.
    """
    effort = thinking.get("effort")
    if isinstance(effort, str):
        if effort == "none":
            return {"effort": "none"}
        return {"effort": effort, "summary": "auto"}
    if thinking.get("type") == "disabled":
        return {"effort": "none"}
    return {"summary": "auto"}


def _reasoning_summary_text(item: Any) -> str:
    parts = getattr(item, "summary", None) or []
    return "\n\n".join(p.text for p in parts if getattr(p, "text", None))


def _from_response(response: Response) -> MessageParam:
    """Convert a Responses API `Response` to an internal `MessageParam`."""
    content_blocks: list[ContentBlockParam] = []
    saw_tool_use = False
    saw_refusal = False

    for item in response.output:
        if item.type == "reasoning":
            summary = _reasoning_summary_text(item)
            if summary:
                content_blocks.append(
                    BetaThinkingBlock(
                        signature="", thinking=summary, type="thinking"
                    )
                )
            content_blocks.append(
                BetaRedactedThinkingBlock(
                    data=item.model_dump_json(exclude_none=True),
                    type="redacted_thinking",
                )
            )
        elif item.type == "message":
            for part in item.content:
                if part.type == "output_text" and part.text:
                    content_blocks.append(TextBlockParam(text=part.text))
                elif part.type == "refusal":
                    saw_refusal = True
                    if part.refusal:
                        content_blocks.append(TextBlockParam(text=part.refusal))
        elif item.type == "function_call":
            saw_tool_use = True
            try:
                arguments = json.loads(item.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed JSON in function call arguments, passing raw string",
                    extra={"call_id": item.call_id, "function": item.name},
                )
                arguments = {"raw_arguments": item.arguments}
            content_blocks.append(
                ToolUseBlockParam(
                    id=item.call_id,
                    name=item.name,
                    input=arguments,
                    extra_content={
                        _RESPONSES_ITEM_KEY: item.model_dump(exclude_none=True)
                    },
                )
            )

    stop_reason: StopReason = "end_turn"
    if saw_tool_use:
        stop_reason = "tool_use"
    elif response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        stop_reason = "max_tokens" if reason == "max_output_tokens" else "end_turn"
    elif saw_refusal:
        stop_reason = "refusal"

    usage: UsageParam | None = None
    if response.usage is not None:
        cached_tokens: int | None = None
        if response.usage.input_tokens_details is not None:
            cached_tokens = response.usage.input_tokens_details.cached_tokens
        # Responses usage counts cached tokens INSIDE input_tokens; `UsageParam`
        # consumers expect Anthropic-style disjoint fields — subtract so cached
        # tokens are never counted or billed twice.
        input_tokens = response.usage.input_tokens
        if isinstance(cached_tokens, int) and cached_tokens > 0:
            input_tokens = max(0, input_tokens - cached_tokens)
        usage = UsageParam(
            input_tokens=input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=cached_tokens,
        )

    if len(content_blocks) == 1 and isinstance(content_blocks[0], TextBlockParam):
        return MessageParam(
            role="assistant",
            content=content_blocks[0].text,
            stop_reason=stop_reason,
            usage=usage,
        )
    return MessageParam(
        role="assistant",
        content=content_blocks,
        stop_reason=stop_reason,
        usage=usage,
    )


def _optional_request_kwargs(
    tools: ToolCollection | None,
    max_tokens: int | None,
    thinking: ThinkingConfigParam | None,
    tool_choice: ToolChoiceParam | None,
    temperature: float | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens

    if temperature is not None:
        kwargs["temperature"] = temperature

    if tools is not None:
        responses_tools = _to_responses_tools(tools)
        if responses_tools:
            kwargs["tools"] = responses_tools

    if tool_choice is not None and "tools" in kwargs:
        responses_tool_choice = _to_responses_tool_choice(tool_choice)
        if responses_tool_choice is not None:
            kwargs["tool_choice"] = responses_tool_choice

    if thinking is not None:
        reasoning = _to_reasoning_param(thinking)
        if reasoning is not None:
            kwargs["reasoning"] = reasoning

    return kwargs


class OpenAIResponsesApi(MessagesApi):
    """MessagesApi implementation for the OpenAI Responses API.

    Use instead of `OpenAIMessagesApi` when reasoning and function tools must
    be active in the same request — chat completions rejects that combination
    on current reasoning models. OpenAI API only (OpenAI-compatible gateways
    generally do not implement ``/v1/responses``).
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        model_id: str,
        tools: ToolCollection | None = None,
        max_tokens: int | None = None,
        system: SystemPrompt | None = None,
        thinking: ThinkingConfigParam | None = None,
        tool_choice: ToolChoiceParam | None = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        """Create a message via the OpenAI Responses API.

        Args:
            messages: The conversation history.
            model_id: The model name (e.g. ``"gpt-5.6-sol"``).
            tools: Tools available to the model for function-calling.
            max_tokens: Mapped to ``max_output_tokens`` (shared between
                reasoning and the answer).
            system: System prompt, sent as ``instructions``.
            thinking: Mapped to the ``reasoning`` param: an explicit
                ``"effort"`` key passes through, ``{"type": "disabled"}``
                becomes effort ``"none"``, ``"adaptive"``/``"enabled"`` keep
                the model default with auto summaries.
            tool_choice: Mapped to the Responses ``tool_choice`` format
                (``{"type": "tool", "name": N}`` forces the named function).
            temperature: Sampling temperature.
            provider_options: Additional keyword arguments forwarded directly
                to ``responses.create``.

        Returns:
            The model's response as a `MessageParam`.
        """
        kwargs: dict[str, Any] = {
            "model": model_id,
            "input": _to_responses_input(messages),
            "stream": False,
            # Stateless: nothing persisted server-side; encrypted reasoning
            # comes back inline so multi-turn tool loops can echo it.
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "timeout": 300.0,
        }
        if system is not None:
            kwargs["instructions"] = str(system)
        kwargs.update(
            _optional_request_kwargs(
                tools=tools,
                max_tokens=max_tokens,
                thinking=thinking,
                tool_choice=tool_choice,
                temperature=temperature,
            )
        )
        if provider_options is not None:
            kwargs.update(provider_options)

        response = self._client.responses.create(**kwargs)
        return _from_response(response)
