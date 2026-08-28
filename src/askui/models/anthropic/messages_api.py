import logging
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
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaMessageParam,
    BetaOutputConfigParam,
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
    Base64PdfSourceParam,
    CacheControlEphemeralParam,
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUseBlockParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.thinking import accepts_sampling_params
from askui.models.shared.tools import ToolCollection
from askui.utils.image_utils import image_to_base64
from askui.utils.pdf_utils import PdfSource

logger = logging.getLogger(__name__)


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is a retryable error."""
    if isinstance(exception, APIStatusError):
        return exception.status_code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exception, (APIConnectionError, APITimeoutError, APIError))


def from_content_block(block: ContentBlockParam) -> BetaContentBlockParam:
    """Convert an internal content block to an Anthropic API-compatible dict.

    Uses `model_dump(exclude_none=True)` to produce plain dicts compatible with
    Anthropic's TypedDicts. ``exclude_none`` omits unset optional fields (e.g.
    ``cache_control``, ``citations``) instead of serialising them as explicit
    ``null``. The Anthropic API tolerates those nulls, but stricter
    Anthropic-compatible endpoints (e.g. OpenRouter's) reject them with
    ``cache_control: expected object, received null``. Also strips
    ``visual_representation`` and ``extra_content`` from `ToolUseBlockParam` as
    they are not accepted by the API.
    """
    if isinstance(block, ToolUseBlockParam):
        # visual_representation (perceptual hash for cache validation) and
        # extra_content (provider-specific data, e.g. Gemini thought signatures)
        # are internal fields that do not exist in the Anthropic API schema.
        # Sending them would cause the API to reject the request with an
        # unknown-field error.
        return cast(
            "BetaContentBlockParam",
            block.model_dump(
                exclude={"visual_representation", "extra_content"},
                exclude_none=True,
            ),
        )
    return cast("BetaContentBlockParam", block.model_dump(exclude_none=True))


def from_message_param(message: MessageParam) -> BetaMessageParam:
    """Convert an internal `MessageParam` to an Anthropic `BetaMessageParam`.

    Strips internal-only fields (`stop_reason`, `usage`,
    `visual_representation`) that are not accepted by the Anthropic API.
    """
    if isinstance(message.content, str):
        return BetaMessageParam(role=message.role, content=message.content)

    return BetaMessageParam(
        role=message.role,
        content=[from_content_block(block) for block in message.content],
    )


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


def built_messages_for_get_pdf(
    pdf_source: PdfSource, prompt: str
) -> list[MessageParam]:
    # Anthropic accepts a base64 PDF `document` block (no beta header); placing
    # the document before the text follows Anthropic's PDF best practices.
    return [
        MessageParam(
            role="user",
            content=cast(
                "list[ContentBlockParam]",
                [
                    DocumentBlockParam(
                        source=Base64PdfSourceParam(
                            data=pdf_source.to_base64(),
                        ),
                        title=pdf_source.filename,
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
    cache_control: CacheControlEphemeralParam | None = None,
    system: SystemPrompt | None = None,
    thinking: ThinkingConfigParam | None = None,
    output_config: dict[str, Any] | None = None,
    tool_choice: ToolChoiceParam | None = None,
) -> Tuple[
    list[BetaToolUnionParam] | Omit,
    list[AnthropicBetaParam] | Omit,
    BetaCacheControlEphemeralParam | Omit,
    str | Omit,
    BetaThinkingConfigParam | Omit,
    BetaOutputConfigParam | Omit,
    BetaToolChoiceParam | Omit,
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

    _cache_control = (
        cast("BetaCacheControlEphemeralParam", cache_control.model_dump())
        if cache_control is not None
        else omit
    )

    _system: str | Omit = omit if system is None else str(system)
    # Cast dicts to Anthropic's TypedDict types
    # Runtime validation happens in Anthropic SDK
    _thinking = (
        cast("BetaThinkingConfigParam", thinking) if thinking is not None else omit
    )
    # `output_config` carries `effort` (the string replacement for the integer
    # `budget_tokens` used by older models) on models with adaptive thinking.
    _output_config = (
        cast("BetaOutputConfigParam", output_config)
        if output_config is not None
        else omit
    )
    _tool_choice = (
        cast("BetaToolChoiceParam", tool_choice) if tool_choice is not None else omit
    )

    return (
        _tools,
        _betas,
        _cache_control,
        _system,
        _thinking,
        _output_config,
        _tool_choice,
    )


class AnthropicMessagesApi(MessagesApi):
    def __init__(
        self,
        client: AnthropicApiClient,
    ) -> None:
        self._client = client
        # Models for which we already warned about an ignored temperature, so
        # the warning fires at most once per model (not on every step).
        self._temperature_warned: set[str] = set()

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
        # convert each message to anthropic BetaMessageParam type
        _messages = [from_message_param(message) for message in messages]

        # Extract Anthropic-specific options from provider_options
        betas: list[str] | None = None
        cache_control: CacheControlEphemeralParam | None = None
        output_config: dict[str, Any] | None = None
        if provider_options is not None:
            betas = provider_options.get("betas")
            cache_control = provider_options.get("cache_control")
            output_config = provider_options.get("output_config")

        (
            _tools,
            _betas,
            _cache_control,
            _system,
            _thinking,
            _output_config,
            _tool_choice,
        ) = _parse_to_anthropic_types(
            tools,
            betas,
            cache_control,
            system,
            thinking,
            output_config,
            tool_choice,
        )

        # Forward `temperature` only to models that accept sampling parameters.
        # The adaptive-thinking Claude generation (Sonnet 4.6 onward) rejects a
        # non-default value with `400 "temperature is deprecated for this
        # model."`, and newer `anthropic` clients removed `temperature` from the
        # typed `beta.messages.create` (passing it as a keyword would crash).
        # So, when accepted, send it in the request body via `extra_body`; when
        # not, drop it and warn once per model.
        extra_body: dict[str, Any] = {}
        if temperature is not None:
            if accepts_sampling_params(model_id):
                extra_body["temperature"] = temperature
            elif model_id not in self._temperature_warned:
                self._temperature_warned.add(model_id)
                logger.warning(
                    "Ignoring temperature=%s: model %s does not accept sampling "
                    "parameters (Anthropic deprecated them for this model "
                    "generation).",
                    temperature,
                    model_id,
                )

        create_kwargs: dict[str, Any] = {"extra_body": extra_body} if extra_body else {}

        response = self._client.beta.messages.create(  # type: ignore[misc]
            messages=_messages,
            max_tokens=max_tokens or 8192,
            cache_control=_cache_control,
            model=model_id,
            tools=_tools,
            betas=_betas,
            system=_system,
            thinking=_thinking,
            output_config=_output_config,
            tool_choice=_tool_choice,
            timeout=300.0,
            **create_kwargs,
        )
        return MessageParam.model_validate(response.model_dump())
