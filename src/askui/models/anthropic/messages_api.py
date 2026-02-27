from typing import Any, Tuple, cast

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    Omit,
    omit,
)
from anthropic.types import AnthropicBetaParam
from anthropic.types.beta import (
    BetaMessageParam,
    BetaThinkingConfigParam,
    BetaToolChoiceParam,
    BetaToolUnionParam,
)
from PIL.Image import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from typing_extensions import override

from askui.models.anthropic.factory import AnthropicApiClient
from askui.models.askui.retry_utils import (
    RETRYABLE_HTTP_STATUS_CODES,
    wait_for_retry_after_header,
)
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection
from askui.utils.image_utils import image_to_base64


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is a retryable error."""
    if isinstance(exception, APIStatusError):
        return exception.status_code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exception, (APIConnectionError, APITimeoutError, APIError))


def _sanitize_message_for_api(message: MessageParam) -> dict[str, Any]:
    """Remove non-API fields from a message before sending to Anthropic API.

    Fields like `usage`, `stop_reason`, and `visual_representation` are used
    internally but not accepted by the API.
    """
    msg_dict = message.model_dump(exclude={"stop_reason", "usage"})

    # Remove visual_representation from tool_use blocks in content
    if isinstance(msg_dict.get("content"), list):
        for block in msg_dict["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                block.pop("visual_representation", None)

    return msg_dict


def built_messages_for_get_and_locate(
    scaled_image: Image, prompt: str
) -> list[MessageParam]:
    return [
        MessageParam(
            role="user",
            content=cast(
                "list[ContentBlockParam]",
                [
                    ImageBlockParam(
                        source=Base64ImageSourceParam(
                            data=image_to_base64(scaled_image),
                            media_type="image/png",
                        ),
                    ),
                    TextBlockParam(
                        text=prompt,
                    ),
                ],
            ),
        )
    ]


def _parse_to_anthropic_types(
    tools: ToolCollection | None,
    betas: list[str] | None = None,
    system: SystemPrompt | None = None,
    thinking: ThinkingConfigParam | None = None,
    tool_choice: ToolChoiceParam | None = None,
    temperature: float | None = None,
) -> Tuple[
    list[BetaToolUnionParam] | Omit,
    list[AnthropicBetaParam] | Omit,
    str | Omit,
    BetaThinkingConfigParam | Omit,
    BetaToolChoiceParam | Omit,
    float | Omit,
]:
    """Convert provider-agnostic types to Anthropic-specific types.

    This function bridges the gap between the generic MessagesApi interface
    and Anthropic's specific type requirements. The input dicts should match
    Anthropic's expected structure (see Anthropic SDK documentation).
    """
    _tools = (
        cast("list[BetaToolUnionParam]", tools.to_params())
        if tools is not None
        else omit
    )
    _betas = cast("list[AnthropicBetaParam]", betas) or omit
    _system: str | Omit = omit if system is None else str(system)
    # Cast dicts to Anthropic's TypedDict types
    # Runtime validation happens in Anthropic SDK
    _thinking = (
        cast("BetaThinkingConfigParam", thinking) if thinking is not None else omit
    )
    _tool_choice = (
        cast("BetaToolChoiceParam", tool_choice) if tool_choice is not None else omit
    )
    _temperature = temperature or omit

    return (_tools, _betas, _system, _thinking, _tool_choice, _temperature)


class AnthropicMessagesApi(MessagesApi):
    def __init__(
        self,
        client: AnthropicApiClient,
    ) -> None:
        self._client = client

    @retry(
        stop=stop_after_attempt(4),  # 3 retries
        wait=wait_for_retry_after_header(
            wait_exponential(multiplier=30, min=30, max=120)
        ),  # retry after or as a fallback 30s, 60s, 120s
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )
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
        _messages = [
            cast("BetaMessageParam", _sanitize_message_for_api(message))
            for message in messages
        ]

        # Extract betas from provider_options
        betas: list[str] | None = None
        if provider_options is not None:
            betas = provider_options.get("betas")

        _tools, _betas, _system, _thinking, _tool_choice, _temperature = (
            _parse_to_anthropic_types(
                tools, betas, system, thinking, tool_choice, temperature
            )
        )

        response = self._client.beta.messages.create(  # type: ignore[misc]
            messages=_messages,
            max_tokens=max_tokens or 8192,
            model=model_id,
            tools=_tools,
            betas=_betas,
            system=_system,
            thinking=_thinking,
            tool_choice=_tool_choice,
            temperature=_temperature,
            timeout=300.0,
        )
        return MessageParam.model_validate(response.model_dump())
