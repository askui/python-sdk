"""OpenAIMessagesApi — MessagesApi for any OpenAI-compatible API."""

import json
import logging
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
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

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def _map_finish_reason(finish_reason: str | None) -> StopReason | None:
    """Map an OpenAI ``finish_reason`` to the internal `StopReason`."""
    if finish_reason is None:
        return None
    return _FINISH_REASON_MAP.get(finish_reason, "end_turn")


def _image_block_to_openai(block: ImageBlockParam) -> dict[str, Any]:
    """Convert an `ImageBlockParam` to an OpenAI ``image_url`` content part."""
    if isinstance(block.source, Base64ImageSourceParam):
        url = f"data:{block.source.media_type};base64,{block.source.data}"
    else:
        url = block.source.url
    return {"type": "image_url", "image_url": {"url": url}}


def _document_block_to_openai(block: DocumentBlockParam) -> dict[str, Any]:
    """Convert a `DocumentBlockParam` (PDF) to an OpenAI ``file`` content part."""
    data_url = f"data:{block.source.media_type};base64,{block.source.data}"
    return {
        "type": "file",
        "file": {
            "filename": "document.pdf",
            "file_data": data_url,
        },
    }


def _serialize_tool_result_content(
    content: str | list[TextBlockParam | ImageBlockParam | DocumentBlockParam],
) -> tuple[str, list[dict[str, Any]]]:
    """Serialize ``ToolResultBlockParam.content`` for OpenAI's ``tool`` role.

    Returns the text portion as a string and any images/documents as OpenAI
    content parts (to be appended as a separate ``user`` message since the
    OpenAI ``tool`` role only accepts string content).
    """
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    media_parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlockParam):
            text_parts.append(block.text)
        elif isinstance(block, ImageBlockParam):
            media_parts.append(_image_block_to_openai(block))
        elif isinstance(block, DocumentBlockParam):
            media_parts.append(_document_block_to_openai(block))

    return "\n".join(text_parts), media_parts


def _content_block_to_openai(block: ContentBlockParam) -> dict[str, Any] | None:
    """Convert a user-message content block to an OpenAI content part.

    Returns ``None`` for block types that should be skipped (e.g. thinking).
    """
    if isinstance(block, TextBlockParam):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlockParam):
        return _image_block_to_openai(block)
    if isinstance(block, DocumentBlockParam):
        return _document_block_to_openai(block)
    if isinstance(block, (BetaThinkingBlock, BetaRedactedThinkingBlock)):
        return None
    return None


def _to_openai_messages(
    messages: list[MessageParam],
    system: SystemPrompt | None = None,
) -> list[dict[str, Any]]:
    """Convert internal ``MessageParam`` list to OpenAI chat messages.

    Handles:
    - System prompt prepended as a ``system`` role message
    - User messages with text/image content parts
    - Assistant messages with text content and ``tool_calls``
    - Tool result messages converted to ``tool`` role
    - Images inside tool results appended as a follow-up ``user`` message
    - Thinking blocks silently skipped
    """
    result: list[dict[str, Any]] = []

    if system is not None:
        result.append({"role": "system", "content": str(system)})

    for message in messages:
        if isinstance(message.content, str):
            result.append({"role": message.role, "content": message.content})
            continue

        if message.role == "assistant":
            _convert_assistant_message(message.content, result)
        else:
            _convert_user_message(message.content, result)

    return result


def _convert_assistant_message(
    blocks: list[ContentBlockParam],
    result: list[dict[str, Any]],
) -> None:
    """Convert an assistant message's content blocks to OpenAI format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        if isinstance(block, TextBlockParam):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlockParam):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        # Skip thinking blocks silently

    openai_msg: dict[str, Any] = {"role": "assistant"}
    content_text = "\n".join(text_parts) if text_parts else None
    openai_msg["content"] = content_text
    if tool_calls:
        openai_msg["tool_calls"] = tool_calls
    result.append(openai_msg)


def _convert_user_message(
    blocks: list[ContentBlockParam],
    result: list[dict[str, Any]],
) -> None:
    """Convert a user message's content blocks to OpenAI format.

    ``ToolResultBlockParam`` blocks become ``tool`` role messages.
    Images and documents inside tool results are collected and appended as a
    separate ``user`` message so the model can still see them.
    """
    tool_result_media: list[dict[str, Any]] = []
    content_parts: list[dict[str, Any]] = []

    for block in blocks:
        if isinstance(block, ToolResultBlockParam):
            text_content, media = _serialize_tool_result_content(block.content)
            tool_result_media.extend(media)
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": text_content,
                }
            )
        else:
            part = _content_block_to_openai(block)
            if part is not None:
                content_parts.append(part)

    if content_parts:
        result.append({"role": "user", "content": content_parts})

    # Append images/documents from tool results as a separate user message
    if tool_result_media:
        result.append({"role": "user", "content": tool_result_media})


def _to_openai_tools(tools: ToolCollection) -> list[dict[str, Any]]:
    """Convert a `ToolCollection` to OpenAI function-calling tool format.

    Strips ``cache_control`` (Anthropic-specific) from tool parameters.
    """
    result: list[dict[str, Any]] = []
    for tool_param in tools.to_params():
        schema = dict(tool_param.get("input_schema", {}))
        schema.pop("cache_control", None)
        func: dict[str, Any] = {
            "name": tool_param["name"],
            "parameters": schema,
        }
        if "description" in tool_param:
            func["description"] = tool_param["description"]
        result.append({"type": "function", "function": func})
    return result


def _parse_tool_calls(
    message: ChatCompletionMessage,
    content_blocks: list[ContentBlockParam],
) -> None:
    """Extract tool calls from the OpenAI response and append as `ToolUseBlockParam`."""
    if not message.tool_calls:
        return
    for tool_call in message.tool_calls:
        if not isinstance(tool_call, ChatCompletionMessageToolCall):
            continue
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.warning(
                "Malformed JSON in tool call arguments, passing raw string",
                extra={
                    "tool_call_id": tool_call.id,
                    "function": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            )
            arguments = {"raw_arguments": tool_call.function.arguments}
        content_blocks.append(
            ToolUseBlockParam(
                id=tool_call.id,
                name=tool_call.function.name,
                input=arguments,
            )
        )


def _from_openai_response(response: ChatCompletion) -> MessageParam:
    """Convert an OpenAI ``ChatCompletion`` to an internal `MessageParam`."""
    choice = response.choices[0]
    message = choice.message
    stop_reason = _map_finish_reason(choice.finish_reason)

    content_blocks: list[ContentBlockParam] = []

    if message.content:
        content_blocks.append(TextBlockParam(text=message.content))

    _parse_tool_calls(message, content_blocks)

    usage: UsageParam | None = None
    if response.usage:
        cached_tokens: int | None = None
        if response.usage.prompt_tokens_details is not None:
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens
        usage = UsageParam(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cache_read_input_tokens=cached_tokens,
        )

    # Simple string content when there's only a single text block
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


class OpenAIMessagesApi(MessagesApi):
    """MessagesApi implementation for any OpenAI-compatible chat API."""

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
        thinking: ThinkingConfigParam | None = None,  # noqa: ARG002
        tool_choice: ToolChoiceParam | None = None,  # noqa: ARG002
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        """Create a message via an OpenAI-compatible chat completions endpoint.

        Args:
            messages: The conversation history.
            model_id: The model name (e.g. ``"gpt-4o"``, ``"qwen2.5vl"``).
            tools: Tools available to the model for function-calling.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            thinking: Ignored (not supported by the OpenAI chat API).
            tool_choice: Ignored.
            temperature: Sampling temperature.
            provider_options: Additional keyword arguments forwarded directly
                to the OpenAI ``chat.completions.create`` call (e.g.
                ``{"top_p": 0.9, "response_format": {...}}``).

        Returns:
            The model's response as a `MessageParam`.
        """
        openai_messages = _to_openai_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": openai_messages,
            "stream": False,
            "timeout": 300.0,
        }

        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature

        if tools is not None:
            openai_tools = _to_openai_tools(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools

        if provider_options is not None:
            kwargs.update(provider_options)

        response = self._client.chat.completions.create(**kwargs)
        return _from_openai_response(response)
