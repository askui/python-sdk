from typing import Any

from pydantic import BaseModel
from typing_extensions import Literal


class CitationCharLocationParam(BaseModel):
    cited_text: str
    document_index: int
    document_title: str | None = None
    end_char_index: int
    start_char_index: int
    type: Literal["char_location"] = "char_location"


class CitationPageLocationParam(BaseModel):
    cited_text: str
    document_index: int
    document_title: str | None = None
    end_page_number: int
    start_page_number: int
    type: Literal["page_location"] = "page_location"


class CitationContentBlockLocationParam(BaseModel):
    cited_text: str
    document_index: int
    document_title: str | None = None
    end_block_index: int
    start_block_index: int
    type: Literal["content_block_location"] = "content_block_location"


TextCitationParam = (
    CitationCharLocationParam
    | CitationPageLocationParam
    | CitationContentBlockLocationParam
)


class UrlImageSourceParam(BaseModel):
    type: Literal["url"] = "url"
    url: str


class Base64ImageSourceParam(BaseModel):
    data: str
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    type: Literal["base64"] = "base64"


class CacheControlEphemeralParam(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"


class ImageBlockParam(BaseModel):
    source: Base64ImageSourceParam | UrlImageSourceParam
    type: Literal["image"] = "image"
    cache_control: CacheControlEphemeralParam | None = None


class TextBlockParam(BaseModel):
    text: str
    type: Literal["text"] = "text"
    cache_control: CacheControlEphemeralParam | None = None
    citations: list[TextCitationParam] | None = None


class ToolResultBlockParam(BaseModel):
    tool_use_id: str
    type: Literal["tool_result"] = "tool_result"
    cache_control: CacheControlEphemeralParam | None = None
    content: str | list[TextBlockParam | ImageBlockParam]
    is_error: bool = False


class ToolUseBlockParam(BaseModel):
    id: str
    input: object
    name: str
    type: Literal["tool_use"] = "tool_use"
    cache_control: CacheControlEphemeralParam | None = None
    visual_representation: str | None = None  # Visual hash for cache validation


class BetaThinkingBlock(BaseModel):
    signature: str
    thinking: str
    type: Literal["thinking"]


class BetaRedactedThinkingBlock(BaseModel):
    data: str
    type: Literal["redacted_thinking"]


ContentBlockParam = (
    ImageBlockParam
    | TextBlockParam
    | ToolResultBlockParam
    | ToolUseBlockParam
    | BetaThinkingBlock
    | BetaRedactedThinkingBlock
)

StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]

# Generic type aliases for provider-agnostic API
# Different providers (Anthropic, OpenAI, etc.) have different structures
# Using dict[str, Any] allows flexibility for different provider requirements
ThinkingConfigParam = dict[str, Any]
ToolChoiceParam = dict[str, Any]
ToolParam = dict[str, Any]


class UsageParam(BaseModel):
    """Token usage statistics from model API calls."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessageParam(BaseModel):
    """A message in a conversation."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlockParam]
    stop_reason: StopReason | None = None
    usage: UsageParam | None = None


__all__ = [
    "Base64ImageSourceParam",
    "CacheControlEphemeralParam",
    "CitationCharLocationParam",
    "CitationContentBlockLocationParam",
    "CitationPageLocationParam",
    "ContentBlockParam",
    "ImageBlockParam",
    "MessageParam",
    "TextBlockParam",
    "TextCitationParam",
    "ToolResultBlockParam",
    "ToolUseBlockParam",
    "UrlImageSourceParam",
    "UsageParam",
    "ThinkingConfigParam",
    "ToolChoiceParam",
]
