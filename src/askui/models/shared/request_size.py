"""Request size estimation shared across truncation and providers.

The Anthropic Messages API rejects requests whose serialized body exceeds
~32 MB with a 400 ``BadRequestError``. Base64-encoded screenshots dominate
that payload. These helpers estimate the serialized byte size cheaply so
truncation strategies (and a provider-side safety net) can keep requests
under the limit.

The estimate reads cached string lengths (``len`` is O(1) on Python
strings, and base64 ``data`` is ASCII so its length equals its serialized
byte count), making a full pass O(number of blocks) rather than
O(payload size). Structural JSON overhead (field names, braces, quotes) is
not counted; it is sub-percent of image-heavy payloads and absorbed by the
threshold headroom callers apply on top of the hard limit.
"""

from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    BetaThinkingBlock,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

# Hard cap on serialized request size for the Anthropic Messages API.
ANTHROPIC_MAX_REQUEST_BYTES = 30 * 1024 * 1024


def estimate_block_bytes(block: ContentBlockParam) -> int:
    """Cheaply estimate the serialized byte size of one content block.

    Base64 image ``data`` is ASCII, so ``len`` equals its byte count and
    is O(1) on Python strings. Walking blocks is therefore O(number of
    blocks) rather than O(payload size), keeping the byte check cheap even
    with many multi-megabyte screenshots.
    """
    if isinstance(block, ImageBlockParam):
        if isinstance(block.source, Base64ImageSourceParam):
            return len(block.source.data)
        return len(block.source.url)
    if isinstance(block, TextBlockParam):
        return len(block.text)
    if isinstance(block, ToolResultBlockParam):
        if isinstance(block.content, str):
            return len(block.content)
        return sum(estimate_block_bytes(nested) for nested in block.content)
    if isinstance(block, ToolUseBlockParam):
        return len(str(block.input)) + len(block.name)
    if isinstance(block, BetaThinkingBlock):
        return len(block.thinking) + len(block.signature)
    # BetaRedactedThinkingBlock
    return len(block.data)


def estimate_message_bytes(message: MessageParam) -> int:
    """Estimate the serialized byte size of a single message."""
    if isinstance(message.content, str):
        return len(message.content)
    return sum(estimate_block_bytes(block) for block in message.content)


def estimate_messages_bytes(messages: list[MessageParam]) -> int:
    """Estimate the serialized byte size of a message history."""
    return sum(estimate_message_bytes(message) for message in messages)
