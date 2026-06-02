"""Unit tests for truncation strategies."""

import logging
from unittest.mock import MagicMock

import pytest

from askui.callbacks.conversation_callback import ConversationCallback
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UrlImageSourceParam,
    UsageParam,
)
from askui.models.shared.request_size import (
    estimate_messages_bytes,
)
from askui.models.shared.truncation_strategies import (
    SlidingImageWindowSummarizingTruncationStrategy,
    SummarizingTruncationStrategy,
    _image_keep_count_for_byte_budget,
)

IMAGE_REMOVED_PLACEHOLDER = "[Screenshot removed to reduce message history length]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base64_image_block(data: str = "abc123") -> ImageBlockParam:
    return ImageBlockParam(
        source=Base64ImageSourceParam(data=data, media_type="image/png"),
    )


def _make_sized_image_block(n_bytes: int) -> ImageBlockParam:
    """Base64 image whose estimated byte size is ``n_bytes``."""
    return _make_base64_image_block(data="x" * n_bytes)


def _make_url_image_block() -> ImageBlockParam:
    return ImageBlockParam(
        source=UrlImageSourceParam(url="https://example.com/img.png"),
    )


def _make_tool_result_with_image(tool_use_id: str = "tool_1") -> ToolResultBlockParam:
    return ToolResultBlockParam(
        tool_use_id=tool_use_id,
        content=[
            TextBlockParam(text="result text"),
            _make_base64_image_block(),
        ],
    )


def _make_vlm_provider(usage: UsageParam | None = None) -> MagicMock:
    provider = MagicMock()
    provider.create_message.return_value = MessageParam(
        role="assistant",
        content="Summary of the conversation.",
        usage=usage,
    )
    # A bare MagicMock attribute is truthy; set None so byte-budget
    # enforcement is skipped (no images stripped to meet a budget) unless
    # a test configures an explicit limit.
    provider.max_request_bytes = None
    return provider


def _make_strategy(
    vlm_provider: MagicMock | None = None,
    n_images_to_keep: int = 3,
    n_messages_to_keep: int = 10,
    max_input_tokens: int = 100_000,
) -> SlidingImageWindowSummarizingTruncationStrategy:
    return SlidingImageWindowSummarizingTruncationStrategy(
        vlm_provider=vlm_provider or _make_vlm_provider(),
        n_images_to_keep=n_images_to_keep,
        n_messages_to_keep=n_messages_to_keep,
        max_input_tokens=max_input_tokens,
    )


def _get_cache_control(block: ContentBlockParam) -> object:
    """Safely get cache_control from a block (returns None for thinking blocks)."""
    return getattr(block, "cache_control", None)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_creates_independent_lists(self) -> None:
        strategy = _make_strategy()
        msgs = [MessageParam(role="user", content="hello")]
        strategy.reset(msgs)
        assert strategy.full_messages is not strategy.truncated_messages

    def test_reset_none_clears_both(self) -> None:
        strategy = _make_strategy()
        strategy.reset([MessageParam(role="user", content="hello")])
        strategy.reset()
        assert strategy.full_messages == []
        assert strategy.truncated_messages == []

    def test_reset_populates_both_histories(self) -> None:
        strategy = _make_strategy()
        msgs = [
            MessageParam(role="user", content="hi"),
            MessageParam(role="assistant", content="hey"),
        ]
        strategy.reset(msgs)
        assert len(strategy.full_messages) == 2
        assert len(strategy.truncated_messages) == 2


# ---------------------------------------------------------------------------
# Append message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    def test_appends_to_both_histories(self) -> None:
        strategy = _make_strategy()
        msg = MessageParam(role="user", content="hello")
        strategy.append_message(msg)
        assert len(strategy.full_messages) == 1
        assert len(strategy.truncated_messages) == 1

    def test_string_content_no_crash(self) -> None:
        strategy = _make_strategy()
        strategy.append_message(MessageParam(role="user", content="just text"))
        assert strategy.truncated_messages[0].content == "just text"


# ---------------------------------------------------------------------------
# Image removal
# ---------------------------------------------------------------------------


class TestRemoveImages:
    def test_strips_oldest_base64_images(self) -> None:
        strategy = _make_strategy(n_images_to_keep=1)
        # Append 3 messages each with a base64 image
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(
                    role=role,
                    content=[_make_base64_image_block()],
                )
            )
        # Only the last image should remain; first two should be placeholders
        truncated = strategy.truncated_messages
        # Message 0: stripped
        assert isinstance(truncated[0].content, list)
        assert isinstance(truncated[0].content[0], TextBlockParam)
        assert truncated[0].content[0].text == IMAGE_REMOVED_PLACEHOLDER
        # Message 1: stripped
        assert isinstance(truncated[1].content, list)
        assert isinstance(truncated[1].content[0], TextBlockParam)
        assert truncated[1].content[0].text == IMAGE_REMOVED_PLACEHOLDER
        # Message 2: preserved
        assert isinstance(truncated[2].content, list)
        assert isinstance(truncated[2].content[0], ImageBlockParam)

    def test_skips_url_images(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_url_image_block()],
            )
        )
        # URL image should not be stripped
        content = strategy.truncated_messages[0].content
        assert isinstance(content, list)
        assert isinstance(content[0], ImageBlockParam)

    def test_strips_images_inside_tool_results(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_tool_result_with_image("tool_1")],
            )
        )
        content = strategy.truncated_messages[0].content
        assert isinstance(content, list)
        tool_result = content[0]
        assert isinstance(tool_result, ToolResultBlockParam)
        assert isinstance(tool_result.content, list)
        # First block is text (kept), second was image (stripped)
        assert isinstance(tool_result.content[0], TextBlockParam)
        assert tool_result.content[0].text == "result text"
        assert isinstance(tool_result.content[1], TextBlockParam)
        assert tool_result.content[1].text == IMAGE_REMOVED_PLACEHOLDER

    def test_preserves_non_image_blocks(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    TextBlockParam(text="keep me"),
                    _make_base64_image_block(),
                ],
            )
        )
        content = strategy.truncated_messages[0].content
        assert isinstance(content, list)
        assert isinstance(content[0], TextBlockParam)
        assert content[0].text == "keep me"

    def test_full_messages_unaffected_by_stripping(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_base64_image_block()],
            )
        )
        # Full history should still have the original image
        full_content = strategy.full_messages[0].content
        assert isinstance(full_content, list)
        assert isinstance(full_content[0], ImageBlockParam)

    def test_no_stripping_when_under_limit(self) -> None:
        strategy = _make_strategy(n_images_to_keep=5)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_base64_image_block()],
            )
        )
        content = strategy.truncated_messages[0].content
        assert isinstance(content, list)
        assert isinstance(content[0], ImageBlockParam)


# ---------------------------------------------------------------------------
# Cache breakpoints
# ---------------------------------------------------------------------------


class TestCacheBreakpoints:
    def test_breakpoint_on_last_user_message(self) -> None:
        strategy = _make_strategy()
        strategy.append_message(
            MessageParam(role="user", content=[TextBlockParam(text="hello")])
        )
        strategy.append_message(
            MessageParam(role="assistant", content=[TextBlockParam(text="hi")])
        )
        # Last user message (index 0) should have cache_control on its last block
        user_msg = strategy.truncated_messages[0]
        assert isinstance(user_msg.content, list)
        assert _get_cache_control(user_msg.content[-1]) is not None

    def test_breakpoint_at_image_removal_boundary(self) -> None:
        strategy = _make_strategy(n_images_to_keep=1)
        # Add messages with images - first two will be stripped
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_base64_image_block()],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[_make_base64_image_block()],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_base64_image_block()],
            )
        )
        # Boundary message (last stripped = index 1) should have cache_control
        boundary_msg = strategy.truncated_messages[1]
        assert isinstance(boundary_msg.content, list)
        assert _get_cache_control(boundary_msg.content[-1]) is not None

    def test_clears_previous_breakpoints(self) -> None:
        strategy = _make_strategy()
        # First append sets breakpoint on message 0
        strategy.append_message(
            MessageParam(role="user", content=[TextBlockParam(text="first")])
        )
        assert isinstance(strategy.truncated_messages[0].content, list)
        assert (
            _get_cache_control(strategy.truncated_messages[0].content[-1]) is not None
        )
        # Second append should clear old breakpoint and set on new last user
        strategy.append_message(
            MessageParam(role="assistant", content=[TextBlockParam(text="reply")])
        )
        strategy.append_message(
            MessageParam(role="user", content=[TextBlockParam(text="second")])
        )
        # Old user message (index 0) should have cache_control cleared
        # New user message (index 2) should have it set
        old_content = strategy.truncated_messages[0].content
        new_content = strategy.truncated_messages[2].content
        assert isinstance(old_content, list)
        assert isinstance(new_content, list)
        assert _get_cache_control(old_content[-1]) is None
        assert _get_cache_control(new_content[-1]) is not None


# ---------------------------------------------------------------------------
# Truncation / summarization
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_truncate_replaces_history_with_summary(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        # Add enough messages to truncate
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        # Force truncation
        strategy.truncate()
        msgs = strategy.truncated_messages
        # First message is the preserved original first user message
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"
        # Then assistant ack, then summary
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "user"
        assert msgs[2].content == "Summary of the conversation."
        # Last 2 messages preserved
        assert msgs[-1].content == "msg 5"
        assert msgs[-2].content == "msg 4"

    def test_truncate_inserts_synthetic_assistant_for_alternation(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        # First user message preserved, then ack, then summary
        assert msgs[0].role == "user"  # original first user message
        assert msgs[0].content == "msg 0"
        assert msgs[1].role == "assistant"  # ack for first user message
        assert msgs[2].role == "user"  # summary
        assert msgs[3].role == "assistant"  # synthetic ack for alternation
        assert "Understood" in str(msgs[3].content)
        assert msgs[4].role == "user"  # msg 4

    def test_truncate_skips_when_too_few_messages(self) -> None:
        strategy = _make_strategy(n_messages_to_keep=10)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # Should not truncate - still 4 messages
        assert len(strategy.truncated_messages) == 4

    def test_truncate_resets_image_boundary(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0, n_messages_to_keep=2)
        strategy.append_message(
            MessageParam(
                role="user",
                content=[_make_base64_image_block()],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[TextBlockParam(text="ok")],
            )
        )
        strategy.append_message(
            MessageParam(role="user", content=[TextBlockParam(text="more")])
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[TextBlockParam(text="sure")],
            )
        )
        # _image_removal_boundary_index should be set after image stripping
        assert strategy._image_removal_boundary_index is not None  # noqa: SLF001
        strategy.truncate()
        assert strategy._image_removal_boundary_index is None  # noqa: SLF001

    def test_full_messages_preserved_after_truncation(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # Full messages should still have all 10
        assert len(strategy.full_messages) == 10
        # Truncated messages should be shorter
        assert len(strategy.truncated_messages) < 10

    def test_truncate_preserves_tool_use_tool_result_pairs(self) -> None:
        vlm = _make_vlm_provider()
        # n_messages_to_keep=3: naive cut would start on the
        # user tool_result, orphaning it from its tool_use.
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=3)
        # Build a realistic tool-calling conversation:
        # user(goal) -> asst(tool_use) -> user(tool_result)
        #            -> asst(tool_use) -> user(tool_result)
        #            -> asst(text)
        strategy.append_message(MessageParam(role="user", content="do something"))
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_1", input={}, name="tool_a", type="tool_use"
                    ),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_1", content="result 1"),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_2", input={}, name="tool_b", type="tool_use"
                    ),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_2", content="result 2"),
                ],
            )
        )
        strategy.append_message(MessageParam(role="assistant", content="all done"))

        strategy.truncate()
        msgs = strategy.truncated_messages

        # Every user message with tool_results must be preceded
        # by an assistant message (not the summary or synthetic).
        for i, m in enumerate(msgs):
            if m.role != "user" or isinstance(m.content, str):
                continue
            has_tr = any(isinstance(b, ToolResultBlockParam) for b in m.content)
            if has_tr:
                prev = msgs[i - 1]
                assert prev.role == "assistant"
                assert (
                    not isinstance(prev.content, str)
                    or "Understood" not in prev.content
                ), f"tool_result at index {i} follows synthetic assistant"

    def test_auto_truncation_on_token_limit(self) -> None:
        vlm = _make_vlm_provider()
        # Very low token threshold to trigger auto-truncation
        strategy = _make_strategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
            max_input_tokens=100,
        )
        # Add messages with enough text to exceed 100 * 0.7 = 70 token threshold
        strategy.append_message(MessageParam(role="user", content="x" * 300))
        strategy.append_message(MessageParam(role="assistant", content="y" * 300))
        strategy.append_message(MessageParam(role="user", content="z" * 300))
        # Should have been auto-truncated
        vlm.create_message.assert_called_once()

    def test_truncate_deferred_when_last_message_has_tool_use(self) -> None:
        """Truncation must not fire when the last message is an assistant
        tool_use whose tool_result hasn't been appended yet."""
        vlm = _make_vlm_provider()
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        # Append an assistant message with tool_use (simulates the window
        # between _get_next_message and _execute_tools_if_present)
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_1", input={}, name="tool_a", type="tool_use"
                    ),
                ],
            )
        )
        # Truncation should be deferred — VLM must NOT be called
        strategy.truncate()
        vlm.create_message.assert_not_called()
        # After appending the matching tool_result, truncation should proceed
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_1", content="result"),
                ],
            )
        )
        strategy.truncate()
        vlm.create_message.assert_called_once()


# ---------------------------------------------------------------------------
# First user message preservation
# ---------------------------------------------------------------------------


class TestFirstUserMessagePreservation:
    """Both strategies must always keep the original first user message."""

    def test_sliding_preserves_first_user_message(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"

    def test_summarizing_preserves_first_user_message(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"

    def test_first_user_message_survives_multiple_truncations(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # Add more messages and truncate again
        for i in range(6, 12):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        # Original first user message must still be at position 0
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"

    def test_first_user_message_captured_from_reset(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        initial_msgs = [
            MessageParam(role="user", content="initial task"),
            MessageParam(role="assistant", content="ok"),
        ]
        strategy.reset(initial_msgs)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        assert msgs[0].role == "user"
        assert msgs[0].content == "initial task"

    def test_first_user_message_cleared_on_none_reset(self) -> None:
        strategy = _make_summarizing_strategy()
        strategy.append_message(MessageParam(role="user", content="first"))
        strategy.reset()
        assert strategy._first_user_message is None  # noqa: SLF001

    def test_role_alternation_valid_after_truncation(self) -> None:
        """Verify user/assistant roles alternate correctly after truncation."""
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        for i in range(len(msgs) - 1):
            assert msgs[i].role != msgs[i + 1].role, (
                f"Adjacent messages at {i} and {i + 1} have the same role: "
                f"{msgs[i].role}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_messages_no_crash(self) -> None:
        strategy = _make_strategy()
        strategy.reset([])
        assert strategy.truncated_messages == []
        assert strategy.full_messages == []

    def test_single_message_with_many_images(self) -> None:
        strategy = _make_strategy(n_images_to_keep=1)
        content: list[ContentBlockParam] = [
            _make_base64_image_block() for _ in range(5)
        ]
        strategy.append_message(MessageParam(role="user", content=content))
        result = strategy.truncated_messages[0].content
        assert isinstance(result, list)
        # First 4 should be placeholders, last should be image
        placeholders = [b for b in result if isinstance(b, TextBlockParam)]
        images = [b for b in result if isinstance(b, ImageBlockParam)]
        assert len(placeholders) == 4
        assert len(images) == 1

    def test_mixed_base64_and_url_images(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        content: list[ContentBlockParam] = [
            _make_base64_image_block(),
            _make_url_image_block(),
            _make_base64_image_block(),
        ]
        strategy.append_message(MessageParam(role="user", content=content))
        result = strategy.truncated_messages[0].content
        assert isinstance(result, list)
        # base64 images stripped, URL image kept
        assert isinstance(result[0], TextBlockParam)  # was base64
        assert isinstance(result[1], ImageBlockParam)  # URL kept
        assert isinstance(result[2], TextBlockParam)  # was base64

    def test_tool_use_blocks_preserved(self) -> None:
        strategy = _make_strategy(n_images_to_keep=0)
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="t1",
                        input={"x": 1},
                        name="my_tool",
                        type="tool_use",
                    ),
                ],
            )
        )
        result = strategy.truncated_messages[0].content
        assert isinstance(result, list)
        assert isinstance(result[0], ToolUseBlockParam)


# ---------------------------------------------------------------------------
# SummarizingTruncationStrategy
# ---------------------------------------------------------------------------


def _make_summarizing_strategy(
    vlm_provider: MagicMock | None = None,
    n_messages_to_keep: int = 10,
    max_input_tokens: int = 100_000,
) -> SummarizingTruncationStrategy:
    return SummarizingTruncationStrategy(
        vlm_provider=vlm_provider or _make_vlm_provider(),
        n_messages_to_keep=n_messages_to_keep,
        max_input_tokens=max_input_tokens,
    )


class TestSummarizingAppend:
    def test_appends_to_both_histories(self) -> None:
        strategy = _make_summarizing_strategy()
        msg = MessageParam(role="user", content="hello")
        strategy.append_message(msg)
        assert len(strategy.full_messages) == 1
        assert len(strategy.truncated_messages) == 1

    def test_does_not_strip_images(self) -> None:
        strategy = _make_summarizing_strategy()
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(
                    role=role,
                    content=[_make_base64_image_block()],
                )
            )
        # All images should remain since no image stripping
        for msg in strategy.truncated_messages:
            assert isinstance(msg.content, list)
            assert isinstance(msg.content[0], ImageBlockParam)

    def test_sets_cache_breakpoint_on_last_user_message(self) -> None:
        strategy = _make_summarizing_strategy()
        strategy.append_message(
            MessageParam(
                role="user",
                content=[TextBlockParam(text="hello")],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[TextBlockParam(text="hi")],
            )
        )
        # Last (and only) user message should have cache breakpoint
        user_msg = strategy.truncated_messages[0]
        assert isinstance(user_msg.content, list)
        assert _get_cache_control(user_msg.content[-1]) is not None
        # Assistant message should not
        asst_msg = strategy.truncated_messages[1]
        assert isinstance(asst_msg.content, list)
        assert _get_cache_control(asst_msg.content[-1]) is None

    def test_moves_cache_breakpoint_forward(self) -> None:
        strategy = _make_summarizing_strategy()
        strategy.append_message(
            MessageParam(
                role="user",
                content=[TextBlockParam(text="first")],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[TextBlockParam(text="reply")],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[TextBlockParam(text="second")],
            )
        )
        # Old user message (index 0) should have cache_control cleared
        old_content = strategy.truncated_messages[0].content
        assert isinstance(old_content, list)
        assert _get_cache_control(old_content[-1]) is None
        # New user message (index 2) should have it set
        new_content = strategy.truncated_messages[2].content
        assert isinstance(new_content, list)
        assert _get_cache_control(new_content[-1]) is not None


class TestSummarizingTruncation:
    def test_truncate_replaces_history_with_summary(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        # First message is the preserved original first user message
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"
        # Then assistant ack, then summary
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "user"
        assert msgs[2].content == "Summary of the conversation."
        assert msgs[-1].content == "msg 5"
        assert msgs[-2].content == "msg 4"

    def test_truncate_inserts_synthetic_assistant(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        # First user message preserved, then ack, then summary
        assert msgs[0].role == "user"
        assert msgs[0].content == "msg 0"
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "user"  # summary
        assert msgs[3].role == "assistant"  # synthetic ack
        assert "Understood" in str(msgs[3].content)

    def test_truncate_skips_when_too_few_messages(self) -> None:
        strategy = _make_summarizing_strategy(n_messages_to_keep=10)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        assert len(strategy.truncated_messages) == 4

    def test_full_messages_preserved_after_truncation(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        assert len(strategy.full_messages) == 10
        assert len(strategy.truncated_messages) < 10

    def test_preserves_tool_use_tool_result_pairs(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=3)
        strategy.append_message(MessageParam(role="user", content="do something"))
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_1",
                        input={},
                        name="tool_a",
                        type="tool_use",
                    ),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_1", content="result 1"),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_2",
                        input={},
                        name="tool_b",
                        type="tool_use",
                    ),
                ],
            )
        )
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_2", content="result 2"),
                ],
            )
        )
        strategy.append_message(MessageParam(role="assistant", content="all done"))
        strategy.truncate()
        msgs = strategy.truncated_messages
        for i, m in enumerate(msgs):
            if m.role != "user" or isinstance(m.content, str):
                continue
            has_tr = any(isinstance(b, ToolResultBlockParam) for b in m.content)
            if has_tr:
                prev = msgs[i - 1]
                assert prev.role == "assistant"
                assert (
                    not isinstance(prev.content, str)
                    or "Understood" not in prev.content
                ), f"tool_result at index {i} follows synthetic"

    def test_auto_truncation_on_token_limit(self) -> None:
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
            max_input_tokens=100,
        )
        strategy.append_message(MessageParam(role="user", content="x" * 300))
        strategy.append_message(MessageParam(role="assistant", content="y" * 300))
        strategy.append_message(MessageParam(role="user", content="z" * 300))
        vlm.create_message.assert_called_once()

    def test_truncate_deferred_when_last_message_has_tool_use(self) -> None:
        """Truncation must not fire when the last message is an assistant
        tool_use whose tool_result hasn't been appended yet."""
        vlm = _make_vlm_provider()
        strategy = _make_summarizing_strategy(vlm_provider=vlm, n_messages_to_keep=2)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.append_message(
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="tu_1", input={}, name="tool_a", type="tool_use"
                    ),
                ],
            )
        )
        strategy.truncate()
        vlm.create_message.assert_not_called()
        strategy.append_message(
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(tool_use_id="tu_1", content="result"),
                ],
            )
        )
        strategy.truncate()
        vlm.create_message.assert_called_once()


class TestReporterIntegration:
    def test_summarizing_strategy_reports_summary_response(self) -> None:
        vlm = _make_vlm_provider()
        reporter = MagicMock()
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.reporter = reporter
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # Byte-budget debug messages (plain strings) may also be reported;
        # the summary response is the one dict payload.
        summary_calls = [
            c
            for c in reporter.add_message.call_args_list
            if isinstance(c.args[1], dict)
        ]
        assert len(summary_calls) == 1
        call_args = summary_calls[0]
        assert call_args.args[0] == "TruncationStrategy"
        # Logged content is the raw VLM response dump
        assert call_args.args[1]["role"] == "assistant"
        assert call_args.args[1]["content"] == "Summary of the conversation."

    def test_sliding_strategy_reports_summary_response(self) -> None:
        vlm = _make_vlm_provider()
        reporter = MagicMock()
        strategy = SlidingImageWindowSummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.reporter = reporter
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # Byte-budget debug messages (plain strings) may also be reported;
        # the summary response is the one dict payload.
        summary_calls = [
            c
            for c in reporter.add_message.call_args_list
            if isinstance(c.args[1], dict)
        ]
        assert len(summary_calls) == 1
        call_args = summary_calls[0]
        assert call_args.args[0] == "TruncationStrategy"
        assert call_args.args[1]["content"] == "Summary of the conversation."

    def test_strategy_does_not_report_when_no_reporter(self) -> None:
        vlm = _make_vlm_provider()
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        # Should not crash even though no reporter is set
        strategy.truncate()
        # First message is preserved original, summary is at index 2
        assert strategy.truncated_messages[0].content == "msg 0"
        assert strategy.truncated_messages[2].content == "Summary of the conversation."


class TestCallbackIntegration:
    def test_summarizing_strategy_notifies_callback_with_usage(self) -> None:
        usage = UsageParam(
            input_tokens=42,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        vlm = _make_vlm_provider(usage=usage)
        callback = MagicMock(spec=ConversationCallback)
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.callbacks = [callback]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        callback.on_truncation_summarize.assert_called_once_with(usage)

    def test_sliding_strategy_notifies_callback_with_usage(self) -> None:
        usage = UsageParam(
            input_tokens=42,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        vlm = _make_vlm_provider(usage=usage)
        callback = MagicMock(spec=ConversationCallback)
        strategy = SlidingImageWindowSummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.callbacks = [callback]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        callback.on_truncation_summarize.assert_called_once_with(usage)

    def test_strategy_does_not_notify_callback_when_no_usage(self) -> None:
        vlm = _make_vlm_provider(usage=None)
        callback = MagicMock(spec=ConversationCallback)
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.callbacks = [callback]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        callback.on_truncation_summarize.assert_not_called()

    def test_strategy_no_callbacks_no_crash(self) -> None:
        vlm = _make_vlm_provider(usage=UsageParam(input_tokens=10, output_tokens=5))
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        # First message is preserved original, summary is at index 2
        assert strategy.truncated_messages[0].content == "msg 0"
        assert strategy.truncated_messages[2].content == "Summary of the conversation."


class TestSummarizationRequestContext:
    """Verify summarization passes the same system/tools as regular calls.

    The Anthropic prompt cache key is ``tools + system + messages_prefix``.
    If the summarization request omits ``system`` or ``tools``, its cache
    key differs from the regular conversation calls and we get 0 cache
    reads on the summarization step.
    """

    def _make_conversation(
        self,
        system: str | None = "system prompt",
        provider_options: dict[str, object] | None = None,
    ) -> MagicMock:
        from askui.models.shared.tools import ToolCollection

        conversation = MagicMock()
        conversation.settings.messages.system = system
        conversation.tools = ToolCollection()
        conversation.settings.messages.provider_options = provider_options
        return conversation

    def test_summarizing_strategy_forwards_system_and_tools(self) -> None:
        vlm = _make_vlm_provider()
        conversation = self._make_conversation(
            system="my system",
            provider_options={"betas": ["foo"]},
        )
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.conversation = conversation
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        vlm.create_message.assert_called_once()
        call_kwargs = vlm.create_message.call_args.kwargs
        assert call_kwargs["system"] == "my system"
        assert call_kwargs["tools"] is conversation.tools
        assert call_kwargs["provider_options"] == {"betas": ["foo"]}

    def test_sliding_strategy_forwards_system_and_tools(self) -> None:
        vlm = _make_vlm_provider()
        conversation = self._make_conversation(
            system="my system",
            provider_options={"betas": ["bar"]},
        )
        strategy = SlidingImageWindowSummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        strategy.conversation = conversation
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        vlm.create_message.assert_called_once()
        call_kwargs = vlm.create_message.call_args.kwargs
        assert call_kwargs["system"] == "my system"
        assert call_kwargs["tools"] is conversation.tools
        assert call_kwargs["provider_options"] == {"betas": ["bar"]}

    def test_strategy_without_conversation_passes_none_for_context(self) -> None:
        vlm = _make_vlm_provider()
        strategy = SummarizingTruncationStrategy(
            vlm_provider=vlm,
            n_messages_to_keep=2,
        )
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(MessageParam(role=role, content=f"msg {i}"))
        strategy.truncate()
        vlm.create_message.assert_called_once()
        call_kwargs = vlm.create_message.call_args.kwargs
        assert call_kwargs["system"] is None
        assert call_kwargs["tools"] is None
        assert call_kwargs["provider_options"] is None


# ---------------------------------------------------------------------------
# Byte-budget enforcement
# ---------------------------------------------------------------------------


def _is_placeholder(block: ContentBlockParam) -> bool:
    return isinstance(block, TextBlockParam) and block.text == IMAGE_REMOVED_PLACEHOLDER


def _first_block(msg: MessageParam) -> ContentBlockParam:
    assert isinstance(msg.content, list)
    return msg.content[0]


class TestByteBudgetHelpers:
    """Direct tests of the keep-count math driving byte enforcement."""

    def test_keep_all_when_under_budget(self) -> None:
        msgs = [
            MessageParam(role="user", content=[_make_sized_image_block(500)]),
            MessageParam(role="user", content=[_make_sized_image_block(500)]),
        ]
        current = estimate_messages_bytes(msgs)
        # Budget exactly equal to current => keep everything.
        assert _image_keep_count_for_byte_budget(msgs, current, current) == 2

    def test_drops_oldest_until_under_budget(self) -> None:
        msgs = [
            MessageParam(role="user", content=[_make_sized_image_block(500)]),
            MessageParam(role="assistant", content=[_make_sized_image_block(500)]),
            MessageParam(role="user", content=[_make_sized_image_block(500)]),
        ]
        current = estimate_messages_bytes(msgs)  # ~1500
        # Only the newest image (~500) fits under 1000.
        assert _image_keep_count_for_byte_budget(msgs, 1000, current) == 1

    def test_drops_all_when_even_one_image_exceeds_budget(self) -> None:
        msgs = [MessageParam(role="user", content=[_make_sized_image_block(500)])]
        current = estimate_messages_bytes(msgs)
        assert _image_keep_count_for_byte_budget(msgs, 100, current) == 0

    def test_no_images_keeps_zero(self) -> None:
        msgs = [MessageParam(role="user", content="x" * 500)]
        assert _image_keep_count_for_byte_budget(msgs, 100, 500) == 0


class TestSummarizingByteBudget:
    """`SummarizingTruncationStrategy` strips oldest images on byte overflow."""

    def _make(self, max_request_bytes: int) -> SummarizingTruncationStrategy:
        return SummarizingTruncationStrategy(
            vlm_provider=_make_vlm_provider(),
            n_messages_to_keep=100,
            # Keep token-based truncation out of the way.
            max_input_tokens=10_000_000,
            max_request_bytes=max_request_bytes,
            # Treat max_request_bytes as the exact budget for these tests.
            request_size_threshold=1.0,
        )

    def test_strips_oldest_images_when_over_budget(self) -> None:
        strategy = self._make(max_request_bytes=1000)
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(500)])
            )
        msgs = strategy.truncated_messages
        # Oldest two images replaced by placeholders, newest kept.
        assert _is_placeholder(_first_block(msgs[0]))
        assert _is_placeholder(_first_block(msgs[1]))
        assert isinstance(_first_block(msgs[2]), ImageBlockParam)

    def test_final_history_within_budget(self) -> None:
        strategy = self._make(max_request_bytes=1000)
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(500)])
            )
        assert estimate_messages_bytes(strategy.truncated_messages) <= 1000

    def test_no_stripping_when_under_budget(self) -> None:
        strategy = self._make(max_request_bytes=10_000)
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(500)])
            )
        for msg in strategy.truncated_messages:
            assert isinstance(_first_block(msg), ImageBlockParam)

    def test_full_messages_keep_original_images(self) -> None:
        strategy = self._make(max_request_bytes=1000)
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(500)])
            )
        # Full (append-only) history must retain every original image.
        for msg in strategy.full_messages:
            assert isinstance(_first_block(msg), ImageBlockParam)

    def test_url_images_not_stripped(self) -> None:
        strategy = self._make(max_request_bytes=1)
        strategy.append_message(
            MessageParam(role="user", content=[_make_url_image_block()])
        )
        # URL images carry no payload bytes and must never be stripped.
        assert isinstance(_first_block(strategy.truncated_messages[0]), ImageBlockParam)

    def test_warns_when_non_image_content_exceeds_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        strategy = self._make(max_request_bytes=100)
        with caplog.at_level(logging.WARNING):
            strategy.append_message(MessageParam(role="user", content="x" * 500))
        # Cannot strip text; logs a warning but preserves the message.
        assert any("byte budget" in record.message for record in caplog.records)
        assert strategy.truncated_messages[0].content == "x" * 500


class TestRequestSizeThreshold:
    """Stripping kicks in at ``max_request_bytes * request_size_threshold``."""

    def test_strips_at_threshold_not_hard_limit(self) -> None:
        # Hard limit 1000, threshold 0.8 => effective budget 800.
        strategy = SummarizingTruncationStrategy(
            vlm_provider=_make_vlm_provider(),
            n_messages_to_keep=100,
            max_input_tokens=10_000_000,
            max_request_bytes=1000,
            request_size_threshold=0.8,
        )
        # Two 500-byte images = ~1000 bytes: under the hard limit but over
        # the 800-byte threshold, so the oldest image must be stripped.
        for i in range(2):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(500)])
            )
        msgs = strategy.truncated_messages
        assert _is_placeholder(_first_block(msgs[0]))
        assert isinstance(_first_block(msgs[1]), ImageBlockParam)
        assert estimate_messages_bytes(msgs) <= 800

    def test_default_threshold_is_80_percent(self) -> None:
        strategy = SummarizingTruncationStrategy(max_request_bytes=1000)
        assert strategy._request_size_threshold == 0.8  # noqa: SLF001
        assert strategy._byte_budget() == 800  # noqa: SLF001


class TestByteBudgetResolution:
    """The byte limit is sourced from the provider unless overridden."""

    def test_explicit_override_wins(self) -> None:
        strategy = SummarizingTruncationStrategy(
            max_request_bytes=1234,
            request_size_threshold=1.0,
        )
        assert strategy._resolve_max_request_bytes() == 1234  # noqa: SLF001

    def test_reads_limit_from_conversation_provider(self) -> None:
        provider = MagicMock()
        provider.max_request_bytes = 5_000_000
        conversation = MagicMock()
        conversation.vlm_provider = provider

        strategy = SummarizingTruncationStrategy(request_size_threshold=1.0)
        strategy.conversation = conversation
        assert strategy._resolve_max_request_bytes() == 5_000_000  # noqa: SLF001

    def test_resolves_to_none_when_provider_has_no_limit(self) -> None:
        provider = MagicMock()
        provider.max_request_bytes = None
        conversation = MagicMock()
        conversation.vlm_provider = provider

        strategy = SummarizingTruncationStrategy(request_size_threshold=1.0)
        strategy.conversation = conversation
        assert strategy._resolve_max_request_bytes() is None  # noqa: SLF001
        assert strategy._byte_budget() is None  # noqa: SLF001

    def test_no_images_stripped_when_no_limit_defined(self) -> None:
        # No explicit override and provider advertises no limit: byte-budget
        # enforcement is skipped, so even large images are kept verbatim.
        provider = _make_vlm_provider()  # max_request_bytes is None
        strategy = SummarizingTruncationStrategy(
            vlm_provider=provider,
            n_messages_to_keep=100,
            max_input_tokens=10_000_000,
        )
        for i in range(2):
            role = "user" if i % 2 == 0 else "assistant"
            strategy.append_message(
                MessageParam(role=role, content=[_make_sized_image_block(5000)])
            )
        msgs = strategy.truncated_messages
        assert isinstance(_first_block(msgs[0]), ImageBlockParam)
        assert isinstance(_first_block(msgs[1]), ImageBlockParam)
