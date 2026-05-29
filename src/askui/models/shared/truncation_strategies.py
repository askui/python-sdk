"""Truncation strategies for managing conversation message history."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.token_counter import SimpleTokenCounter
from askui.models.shared.tools import ToolCollection
from askui.prompts.truncation import SUMMARIZE_INSTRUCTION_PROMPT
from askui.reporting import Reporter

if TYPE_CHECKING:
    from askui.callbacks.conversation_callback import ConversationCallback
    from askui.models.shared.conversation import Conversation

logger = logging.getLogger(__name__)

# needs to be below limits imposed by endpoint
MAX_INPUT_TOKENS = 100_000

# we will truncate as soon as we reach this threshold
TRUNCATION_THRESHOLD = 0.7

# see https://docs.anthropic.com/en/api/messages#body-messages
MAX_MESSAGES = 100_000

IMAGE_REMOVED_PLACEHOLDER = "[Screenshot removed to reduce message history length]"
"""Text used to replace stripped base64 images."""


def _has_orphaned_tool_results(msg: MessageParam) -> bool:
    """Check if a message contains tool_result blocks.

    Such a message cannot be the first in ``recent``
    because the preceding assistant message with the
    matching tool_use would be lost.
    """
    if msg.role != "user" or isinstance(msg.content, str):
        return False
    return any(isinstance(b, ToolResultBlockParam) for b in msg.content)


def _has_pending_tool_use(messages: list[MessageParam]) -> bool:
    """Check if the last message is an assistant message with tool_use blocks.

    Truncation can fire when ``Conversation._get_next_message``
    adds the assistant response to history *before* the matching
    ``tool_result`` user message is appended by
    ``_execute_tools_if_present``.  In that window the history
    ends on an assistant ``tool_use``, and
    ``_summarize_message_history`` would append a plain-text user
    message (the summarization prompt) which violates the API
    constraint that every ``tool_use`` must be followed by its
    ``tool_result``.
    """
    if not messages:
        return False
    last = messages[-1]
    if last.role != "assistant" or isinstance(last.content, str):
        return False
    return any(isinstance(b, ToolUseBlockParam) for b in last.content)


def _summarize_message_history(
    vlm_provider: VlmProvider,
    messages: list[MessageParam],
    system: SystemPrompt | None = None,
    tools: ToolCollection | None = None,
    provider_options: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> MessageParam:
    """Ask the VLM to summarize the conversation history.

    The ``system`` and ``tools`` arguments are forwarded to
    `VlmProvider.create_message` so the prompt-cache prefix matches
    the regular conversation calls and we get cache hits on the
    history. Without them, the cache key differs and every
    summarization call is a cache miss.

    Args:
        vlm_provider: VLM provider to use for summarization.
        messages: Messages to summarize.
        system: System prompt used by the regular conversation
            calls. Required for cache hits on the prefix.
        tools: Tools used by the regular conversation calls.
            Required for cache hits on the prefix.
        provider_options: Provider-specific options (e.g. ``betas``)
            used by the regular conversation calls.
        reporter: Reporter to log errors during summarization to

    Returns:
        The raw VLM response message.
    """
    messages_to_summarize = list(messages)

    # Ensure valid role alternation
    if messages_to_summarize and messages_to_summarize[-1].role == "user":
        messages_to_summarize.append(
            MessageParam(
                role="assistant",
                content="I understand. Please go ahead.",
            )
        )

    messages_to_summarize.append(
        MessageParam(
            role="user",
            content=SUMMARIZE_INSTRUCTION_PROMPT,
        )
    )

    try:
        return vlm_provider.create_message(
            messages=messages_to_summarize,
            max_tokens=2048,
            system=system,
            tools=tools,
            provider_options=provider_options,
        )
    except Exception as e:
        # catch e.g. BadRequestError
        error_msg = f"Truncation Failed with error: {e}"
        logger.exception(error_msg)
        if reporter:
            reporter.add_message("TruncationStrategy", error_msg)
        raise


def _extract_summary_text(response: MessageParam) -> str:
    """Extract text content from a VLM summary response."""
    if isinstance(response.content, str):
        return response.content
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextBlockParam)
    )


def _clear_cache_control(msg: MessageParam) -> None:
    """Clear ``cache_control`` on all blocks of a message."""
    if isinstance(msg.content, str):
        return
    cacheable = (
        ImageBlockParam,
        TextBlockParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
    )
    for block in msg.content:
        if isinstance(block, cacheable):
            block.cache_control = None
        if isinstance(block, ToolResultBlockParam) and isinstance(block.content, list):
            for nested in block.content:
                nested.cache_control = None


def _set_cache_breakpoint(msg: MessageParam) -> None:
    """Set cache breakpoint on last block of a message."""
    if isinstance(msg.content, str) or not msg.content:
        return
    last_block = msg.content[-1]
    cacheable = (
        ImageBlockParam,
        TextBlockParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
    )
    if isinstance(last_block, cacheable):
        last_block.cache_control = CacheControlEphemeralParam()


class TruncationStrategy(ABC):
    """Abstract base class for truncation strategies.

    Manages two separate message histories:

    - ``full_messages``: append-only, preserves all original
      messages for cache recording
    - ``truncated_messages``: may have images stripped and
      history summarized for LLM calls

    Conversation-owned dependencies (``vlm_provider``, ``reporter``,
    ``callbacks``, ``conversation``) are auto-injected by the
    `Conversation` when it takes ownership of the strategy, so
    users only need to configure strategy-specific parameters when
    constructing a custom strategy:

    .. code-block:: python

        strategy = SummarizingTruncationStrategy(n_messages_to_keep=5)
        agent = ComputerAgent(truncation_strategy=strategy)

    ``vlm_provider`` can be pre-set to override the conversation's
    default VLM for summarization (e.g. to use a cheaper model).

    Args:
        max_messages: Maximum number of messages before
            forcing truncation.
        max_input_tokens: Maximum input tokens for the endpoint.
        truncation_threshold: Fraction of `max_input_tokens`
            at which to truncate.
    """

    def __init__(
        self,
        max_messages: int = MAX_MESSAGES,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        truncation_threshold: float = TRUNCATION_THRESHOLD,
    ) -> None:
        self._full_message_history: list[MessageParam] = []
        self._truncated_message_history: list[MessageParam] = []
        self._first_user_message: MessageParam | None = None
        self._max_messages = max_messages
        self._absolute_truncation_threshold = int(
            max_input_tokens * truncation_threshold
        )
        # Conversation-owned dependencies, auto-injected by Conversation.
        # Can be set manually before passing the strategy to the
        # conversation (e.g. vlm_provider override, tests in isolation).
        self.vlm_provider: VlmProvider | None = None
        self.reporter: Reporter | None = None
        self.callbacks: list[ConversationCallback] = []
        self.conversation: "Conversation | None" = None

    def _summarization_request_context(
        self,
    ) -> tuple[SystemPrompt | None, ToolCollection | None, dict[str, Any] | None]:
        """Read the request context used by the regular conversation calls.

        Returns ``(system, tools, provider_options)`` from the owning
        conversation, or all-``None`` if no conversation is attached
        (e.g. in unit tests that exercise the strategy in isolation).
        """
        if self.conversation is None:
            return None, None, None
        return (
            self.conversation.settings.messages.system,
            self.conversation.tools,
            self.conversation.settings.messages.provider_options,
        )

    @abstractmethod
    def append_message(self, message: MessageParam) -> None:
        """Append a message and apply any truncation logic."""
        ...

    @abstractmethod
    def truncate(self) -> None:
        """Force-truncate the message history."""
        ...

    def _capture_first_user_message(self, message: MessageParam) -> None:
        """Store the first user message if not already captured.

        Args:
            message: The message to check and potentially store.
        """
        if self._first_user_message is None and message.role == "user":
            self._first_user_message = message

    def _build_truncated_messages(
        self,
        summary: str,
        recent: list[MessageParam],
    ) -> list[MessageParam]:
        """Build the new truncated history with the first user message preserved.

        Prepends the original first user message (if captured),
        adds the summary, ensures valid role alternation, and
        appends the recent messages.

        Args:
            summary: The LLM-generated conversation summary.
            recent: The most-recent messages to keep verbatim.

        Returns:
            The assembled message list.
        """
        new_messages: list[MessageParam] = []

        if self._first_user_message is not None:
            new_messages.append(self._first_user_message)
            new_messages.append(
                MessageParam(
                    role="assistant",
                    content="Understood. I'll keep your original instructions in mind.",
                )
            )

        new_messages.append(MessageParam(role="user", content=summary))

        if recent and recent[0].role == "user":
            new_messages.append(
                MessageParam(
                    role="assistant",
                    content=(
                        "Understood. I'll continue based on "
                        "the conversation summary above."
                    ),
                )
            )

        new_messages.extend(recent)
        return new_messages

    def reset(self, messages: list[MessageParam] | None = None) -> None:
        """Reset message histories with optional initial messages.

        Creates independent copies so modifications to truncated
        history do not affect full history.

        Args:
            messages: Initial messages to populate both histories.
                If ``None``, both histories are cleared.
        """
        if messages is not None:
            self._full_message_history = list(messages)
            self._truncated_message_history = list(messages)
            self._first_user_message = next(
                (m for m in messages if m.role == "user"), None
            )
        else:
            self._full_message_history = []
            self._truncated_message_history = []
            self._first_user_message = None

    @property
    def truncated_messages(self) -> list[MessageParam]:
        """Get the truncated messages sent to the LLM."""
        return self._truncated_message_history

    @property
    def full_messages(self) -> list[MessageParam]:
        """Get the full, untruncated messages for cache recording."""
        return self._full_message_history


class SlidingImageWindowSummarizingTruncationStrategy(TruncationStrategy):
    """Truncation strategy that strips old images, manages
    cache breakpoints, and summarizes.

    On each appended message:

    1. Strips base64 images beyond `n_images_to_keep`
       (oldest first)
    2. Places dual cache breakpoints (at image-removal
       boundary + last user message)
    3. If token count exceeds threshold, summarizes
       the history via the VLM

    Conversation-owned dependencies (``vlm_provider``, ``reporter``,
    ``callbacks``, ``conversation``) are auto-injected by
    `Conversation`. Pre-set ``vlm_provider`` to override the
    conversation's default (e.g. to use a cheaper model for
    summarization).

    Args:
        n_images_to_keep: Number of most-recent base64 images
            to retain.
        n_messages_to_keep: Number of most-recent messages to
            preserve during summarization.
        max_messages: Maximum number of messages before
            forcing truncation.
        max_input_tokens: Maximum input tokens for the
            endpoint.
        truncation_threshold: Fraction of `max_input_tokens`
            at which to truncate.
        vlm_provider: Optional override for the summarization
            VLM. When ``None`` (default), the conversation's
            ``vlm_provider`` is used.
    """

    def __init__(
        self,
        n_images_to_keep: int = 3,
        n_messages_to_keep: int = 10,
        max_messages: int = MAX_MESSAGES,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        truncation_threshold: float = TRUNCATION_THRESHOLD,
        vlm_provider: VlmProvider | None = None,
    ) -> None:
        super().__init__(
            max_messages,
            max_input_tokens,
            truncation_threshold,
        )
        self.vlm_provider = vlm_provider
        self._n_images_to_keep = n_images_to_keep
        self._n_messages_to_keep = n_messages_to_keep
        self._token_counter = SimpleTokenCounter()
        self._image_removal_boundary_index: int | None = None
        self._debug_writer = None

        logger.warning(
            "%s is experimental and may change, misbehave or crash "
            "without warning. For production use, prefer "
            "SummarizingTruncationStrategy.",
            type(self).__name__,
        )

    @override
    def append_message(self, message: MessageParam) -> None:
        """Append a message and apply image stripping,
        cache breakpoints, and truncation.

        Args:
            message: The message to append.
        """
        self._capture_first_user_message(message)
        self._full_message_history.append(message)
        self._truncated_message_history.append(message)

        # Strip old base64 images (sets _image_removal_boundary_index)
        self._remove_images()

        # Place cache breakpoints using the boundary index
        self._move_cache_breakpoints()

        # Check if truncation is needed
        token_counts = self._token_counter.count_tokens(
            messages=self._truncated_message_history,
        )
        truncated = False
        if (
            len(self._truncated_message_history) > self._max_messages
            or token_counts.total > self._absolute_truncation_threshold
        ):
            self.truncate()
            truncated = True

        if self._debug_writer:
            # will only be used if compatible debug writer is injected
            self._debug_writer.write_snapshot(  # type: ignore[unreachable]
                event="truncate" if truncated else "append",
                full_messages=self._full_message_history,
                truncated_messages=self._truncated_message_history,
                token_estimate=token_counts.total,
                threshold=self._absolute_truncation_threshold,
                image_boundary_idx=self._image_removal_boundary_index,
            )

    @override
    def truncate(self) -> None:
        """Summarize old messages and keep only recent ones."""
        if len(self._truncated_message_history) <= self._n_messages_to_keep:
            msg = "Cannot truncate: too few messages in history"
            logger.warning(msg)
            return
        if self.vlm_provider is None:
            msg = "Cannot truncate: no vlm_provider available"
            logger.warning(msg)
            return
        if _has_pending_tool_use(self._truncated_message_history):
            logger.debug("Deferring truncation: last message has pending tool_use")
            return

        logger.info("Summarizing message history")
        system, tools, provider_options = self._summarization_request_context()
        response = _summarize_message_history(
            self.vlm_provider,
            self._truncated_message_history,
            system=system,
            tools=tools,
            provider_options=provider_options,
            reporter=self.reporter,
        )
        if self.reporter:
            self.reporter.add_message(
                "TruncationStrategy",
                response.model_dump(mode="json"),
            )
        if response.usage is not None:
            for callback in self.callbacks:
                callback.on_truncation_summarize(response.usage)
        summary = _extract_summary_text(response)

        # Find a safe cut point that doesn't orphan tool_results.
        # A user message with tool_result blocks requires the
        # preceding assistant message to contain the matching
        # tool_use blocks, so we must not start `recent` on one.
        cut = len(self._truncated_message_history) - self._n_messages_to_keep
        while cut > 0 and _has_orphaned_tool_results(
            self._truncated_message_history[cut]
        ):
            cut -= 1

        if cut <= 0:
            msg = "Cannot truncate: no safe cut point found"
            logger.warning(msg)
            return

        recent = self._truncated_message_history[cut:]
        self._truncated_message_history = self._build_truncated_messages(
            summary, recent
        )
        self._image_removal_boundary_index = None

    # ------------------------------------------------------------------
    # Image removal
    # ------------------------------------------------------------------

    def _remove_images(self) -> None:
        """Strip old base64 images from truncated history.

        Walks from the beginning and replaces excess base64
        `ImageBlockParam` blocks with text placeholders.  Also
        recurses into `ToolResultBlockParam.content` lists.
        URL-based images are never stripped.
        """
        total = self._count_base64_images(self._truncated_message_history)
        to_remove = total - self._n_images_to_keep
        if to_remove <= 0:
            return

        removed = 0
        for i, msg in enumerate(self._truncated_message_history):
            if removed >= to_remove:
                break
            if isinstance(msg.content, str):
                continue

            new_content, removed_in_msg = self._strip_base64_images(
                msg.content, to_remove - removed
            )
            if removed_in_msg > 0:
                self._truncated_message_history[i] = MessageParam(
                    role=msg.role,
                    content=new_content,
                    stop_reason=msg.stop_reason,
                    usage=msg.usage,
                )
                self._image_removal_boundary_index = i
                removed += removed_in_msg

    @staticmethod
    def _count_base64_images(
        messages: list[MessageParam],
    ) -> int:
        """Count total base64 image blocks across messages."""
        count = 0
        for msg in messages:
            if isinstance(msg.content, str):
                continue
            for block in msg.content:
                if isinstance(block, ImageBlockParam) and isinstance(
                    block.source, Base64ImageSourceParam
                ):
                    count += 1
                elif isinstance(block, ToolResultBlockParam) and isinstance(
                    block.content, list
                ):
                    for nested in block.content:
                        if isinstance(nested, ImageBlockParam) and isinstance(
                            nested.source,
                            Base64ImageSourceParam,
                        ):
                            count += 1
        return count

    @staticmethod
    def _strip_base64_images(
        content: list[ContentBlockParam],
        max_to_strip: int,
    ) -> tuple[list[ContentBlockParam], int]:
        """Strip up to `max_to_strip` base64 images.

        Args:
            content: The content blocks to process.
            max_to_strip: Maximum number of images to strip.

        Returns:
            Tuple of (new content list, count stripped).
        """
        stripped = 0
        new_content: list[ContentBlockParam] = []

        for block in content:
            if stripped >= max_to_strip:
                new_content.append(block)
                continue

            if isinstance(block, ImageBlockParam) and isinstance(
                block.source, Base64ImageSourceParam
            ):
                new_content.append(TextBlockParam(text=IMAGE_REMOVED_PLACEHOLDER))
                stripped += 1
            elif isinstance(block, ToolResultBlockParam) and isinstance(
                block.content, list
            ):
                new_nested: list[TextBlockParam | ImageBlockParam] = []
                for nested in block.content:
                    if (
                        stripped < max_to_strip
                        and isinstance(nested, ImageBlockParam)
                        and isinstance(
                            nested.source,
                            Base64ImageSourceParam,
                        )
                    ):
                        new_nested.append(
                            TextBlockParam(text=IMAGE_REMOVED_PLACEHOLDER)
                        )
                        stripped += 1
                    else:
                        new_nested.append(nested)
                new_content.append(
                    ToolResultBlockParam(
                        tool_use_id=block.tool_use_id,
                        content=new_nested,
                        is_error=block.is_error,
                        cache_control=block.cache_control,
                    )
                )
            else:
                new_content.append(block)

        return new_content, stripped

    # ------------------------------------------------------------------
    # Cache breakpoints
    # ------------------------------------------------------------------

    def _move_cache_breakpoints(self) -> None:
        """Place dual cache breakpoints on truncated history.

        - **Breakpoint 1** – image-removal boundary.
        - **Breakpoint 2** – last user message.
        """
        # Clear all existing cache_control
        for msg in self._truncated_message_history:
            self._clear_cache_control(msg)

        # Breakpoint 1: at image removal boundary
        if (
            self._image_removal_boundary_index is not None
            and self._image_removal_boundary_index
            < len(self._truncated_message_history)
        ):
            self._set_cache_breakpoint(
                self._truncated_message_history[self._image_removal_boundary_index]
            )

        # Breakpoint 2: last user message
        for msg in reversed(self._truncated_message_history):
            if msg.role == "user":
                self._set_cache_breakpoint(msg)
                break

    @staticmethod
    def _clear_cache_control(msg: MessageParam) -> None:
        """Clear ``cache_control`` on all blocks."""
        _clear_cache_control(msg)

    @staticmethod
    def _set_cache_breakpoint(msg: MessageParam) -> None:
        """Set cache breakpoint on last block of a message."""
        _set_cache_breakpoint(msg)


class SummarizingTruncationStrategy(TruncationStrategy):
    """Truncation strategy that summarizes when limits are hit.

    Unlike `SlidingImageWindowSummarizingTruncationStrategy`,
    this strategy does **not** strip images. It places a
    single cache breakpoint on the last user message (moving
    it forward on each append) and summarizes the conversation
    history via the VLM when the token or message count
    exceeds the configured threshold.

    Conversation-owned dependencies (``vlm_provider``, ``reporter``,
    ``callbacks``, ``conversation``) are auto-injected by
    `Conversation`. Pre-set ``vlm_provider`` to override the
    conversation's default (e.g. to use a cheaper model for
    summarization).

    Args:
        n_messages_to_keep: Number of most-recent messages to
            preserve during summarization.
        max_messages: Maximum number of messages before
            forcing truncation.
        max_input_tokens: Maximum input tokens for the
            endpoint.
        truncation_threshold: Fraction of `max_input_tokens`
            at which to truncate.
        vlm_provider: Optional override for the summarization
            VLM. When ``None`` (default), the conversation's
            ``vlm_provider`` is used.
    """

    def __init__(
        self,
        n_messages_to_keep: int = 10,
        max_messages: int = MAX_MESSAGES,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        truncation_threshold: float = TRUNCATION_THRESHOLD,
        vlm_provider: VlmProvider | None = None,
    ) -> None:
        super().__init__(
            max_messages,
            max_input_tokens,
            truncation_threshold,
        )
        self.vlm_provider = vlm_provider
        self._n_messages_to_keep = n_messages_to_keep
        self._token_counter = SimpleTokenCounter()

    @override
    def append_message(self, message: MessageParam) -> None:
        """Append a message, move cache breakpoint, summarize
        if limits are hit.

        Places a cache breakpoint on the last user message
        and clears it from any previous position so the LLM
        caches the full prefix optimally.

        Args:
            message: The message to append.
        """
        self._capture_first_user_message(message)
        self._full_message_history.append(message)
        self._truncated_message_history.append(message)

        # Move cache breakpoint to last user message
        self._move_cache_breakpoint()

        token_counts = self._token_counter.count_tokens(
            messages=self._truncated_message_history,
        )
        if (
            len(self._truncated_message_history) > self._max_messages
            or token_counts.total > self._absolute_truncation_threshold
        ):
            self.truncate()

    def _move_cache_breakpoint(self) -> None:
        """Place a cache breakpoint on the last user message.

        Clears ``cache_control`` from the previous last user
        message first so only one breakpoint exists at a time.
        """
        found_last = False
        for msg in reversed(self._truncated_message_history):
            if msg.role != "user":
                continue
            if not found_last:
                found_last = True
                _set_cache_breakpoint(msg)
            else:
                _clear_cache_control(msg)
                break

    @override
    def truncate(self) -> None:
        """Summarize old messages and keep only recent ones."""
        if len(self._truncated_message_history) <= self._n_messages_to_keep:
            msg = "Cannot truncate: too few messages in history"
            logger.warning(msg)
            return
        if self.vlm_provider is None:
            msg = "Cannot truncate: no vlm_provider available"
            logger.warning(msg)
            return
        if _has_pending_tool_use(self._truncated_message_history):
            logger.debug("Deferring truncation: last message has pending tool_use")
            return

        logger.info("Summarizing message history")
        system, tools, provider_options = self._summarization_request_context()
        response = _summarize_message_history(
            self.vlm_provider,
            self._truncated_message_history,
            system=system,
            tools=tools,
            provider_options=provider_options,
            reporter=self.reporter,
        )
        if self.reporter:
            self.reporter.add_message(
                "TruncationStrategy",
                response.model_dump(mode="json"),
            )
        if response.usage is not None:
            for callback in self.callbacks:
                callback.on_truncation_summarize(response.usage)
        summary = _extract_summary_text(response)

        # Find a safe cut point that doesn't orphan
        # tool_results from their tool_use.
        cut = len(self._truncated_message_history) - self._n_messages_to_keep
        while cut > 0 and _has_orphaned_tool_results(
            self._truncated_message_history[cut]
        ):
            cut -= 1

        if cut <= 0:
            msg = "Cannot truncate: no safe cut point found"
            logger.warning(msg)
            return

        recent = self._truncated_message_history[cut:]
        self._truncated_message_history = self._build_truncated_messages(
            summary, recent
        )
