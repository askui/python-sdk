"""Unit tests for OpenAI messages API conversion functions."""

from unittest.mock import MagicMock

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from askui.models.openai.messages_api import (
    OpenAIMessagesApi,
    _from_openai_response,
    _image_block_to_openai,
    _map_finish_reason,
    _serialize_tool_result_content,
    _to_openai_messages,
    _to_openai_tool_choice,
    _to_openai_tools,
    _to_reasoning_effort,
)
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    Base64PdfSourceParam,
    BetaRedactedThinkingBlock,
    BetaThinkingBlock,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UrlImageSourceParam,
)
from askui.models.shared.prompts import SystemPrompt


def _make_completion(
    content: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int | None = None,
) -> ChatCompletion:
    """Create a mock ChatCompletion response."""
    prompt_tokens_details = None
    if cached_tokens is not None:
        from openai.types.completion_usage import PromptTokensDetails

        prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        created=1234567890,
        model="qwen2.5vl",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=prompt_tokens_details,
        ),
    )


class TestMapFinishReason:
    def test_stop_maps_to_end_turn(self) -> None:
        assert _map_finish_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self) -> None:
        assert _map_finish_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self) -> None:
        assert _map_finish_reason("tool_calls") == "tool_use"

    def test_content_filter_maps_to_refusal(self) -> None:
        assert _map_finish_reason("content_filter") == "refusal"

    def test_none_returns_none(self) -> None:
        assert _map_finish_reason(None) is None

    def test_unknown_falls_back_to_end_turn(self) -> None:
        assert _map_finish_reason("unknown_reason") == "end_turn"


class TestImageBlockToOpenai:
    def test_base64_image(self) -> None:
        block = ImageBlockParam(
            source=Base64ImageSourceParam(data="aWltYWdl", media_type="image/png")
        )
        result = _image_block_to_openai(block)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aWltYWdl"},
        }

    def test_url_image(self) -> None:
        block = ImageBlockParam(
            source=UrlImageSourceParam(url="https://example.com/img.png")
        )
        result = _image_block_to_openai(block)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }


class TestSerializeToolResultContent:
    def test_string_content(self) -> None:
        text, images = _serialize_tool_result_content("hello")
        assert text == "hello"
        assert images == []

    def test_text_blocks(self) -> None:
        content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = [
            TextBlockParam(text="line1"),
            TextBlockParam(text="line2"),
        ]
        text, images = _serialize_tool_result_content(content)
        assert text == "line1\nline2"
        assert images == []

    def test_image_blocks_extracted(self) -> None:
        content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = [
            TextBlockParam(text="screenshot"),
            ImageBlockParam(
                source=Base64ImageSourceParam(data="abc", media_type="image/png")
            ),
        ]
        text, images = _serialize_tool_result_content(content)
        assert text == "screenshot"
        assert len(images) == 1
        assert images[0]["type"] == "image_url"

    def test_document_blocks_become_file_parts(self) -> None:
        content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = [
            TextBlockParam(text="report attached"),
            DocumentBlockParam(source=Base64PdfSourceParam(data="cGRm")),
        ]
        text, media = _serialize_tool_result_content(content)
        assert text == "report attached"
        assert len(media) == 1
        assert media[0]["type"] == "file"
        assert media[0]["file"]["file_data"] == "data:application/pdf;base64,cGRm"


class TestToOpenaiMessages:
    def test_simple_text_message(self) -> None:
        messages = [MessageParam(role="user", content="hello")]
        result = _to_openai_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_prompt_prepended(self) -> None:
        system = SystemPrompt(prompt="Be helpful.")
        messages = [MessageParam(role="user", content="hi")]
        result = _to_openai_messages(messages, system)
        assert result[0] == {"role": "system", "content": "Be helpful."}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_user_message_with_image(self) -> None:
        messages = [
            MessageParam(
                role="user",
                content=[
                    TextBlockParam(text="What is this?"),
                    ImageBlockParam(
                        source=Base64ImageSourceParam(
                            data="abc123", media_type="image/png"
                        )
                    ),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "What is this?"}
        assert parts[1]["type"] == "image_url"

    def test_tool_call_extra_content_round_tripped(self) -> None:
        messages = [
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        id="call_1",
                        name="move_mouse",
                        input={"x": 1, "y": 2},
                        extra_content={"google": {"thought_signature": "sig-abc"}},
                    ),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["extra_content"] == {"google": {"thought_signature": "sig-abc"}}

    def test_tool_call_without_extra_content_omits_key(self) -> None:
        messages = [
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(id="call_1", name="screenshot", input={}),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert "extra_content" not in result[0]["tool_calls"][0]

    def test_assistant_message_with_tool_calls(self) -> None:
        messages = [
            MessageParam(
                role="assistant",
                content=[
                    TextBlockParam(text="I'll take a screenshot."),
                    ToolUseBlockParam(
                        id="call_1",
                        name="screenshot",
                        input={},
                    ),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I'll take a screenshot."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "screenshot"
        assert tc["function"]["arguments"] == "{}"

    def test_tool_result_message(self) -> None:
        messages = [
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        tool_use_id="call_1",
                        content="Done",
                    ),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert len(result) == 1
        assert result[0] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Done",
        }

    def test_tool_result_with_images_appended_as_user_message(self) -> None:
        messages = [
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        tool_use_id="call_1",
                        content=[
                            TextBlockParam(text="Screenshot taken"),
                            ImageBlockParam(
                                source=Base64ImageSourceParam(
                                    data="img", media_type="image/png"
                                )
                            ),
                        ],
                    ),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        # Should produce: tool message + user message with image
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Screenshot taken"
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "image_url"

    def test_thinking_blocks_skipped(self) -> None:
        messages = [
            MessageParam(
                role="assistant",
                content=[
                    BetaThinkingBlock(
                        signature="sig", thinking="hmm...", type="thinking"
                    ),
                    TextBlockParam(text="The answer is 42."),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "The answer is 42."
        assert "tool_calls" not in result[0]

    def test_redacted_thinking_blocks_skipped(self) -> None:
        messages = [
            MessageParam(
                role="assistant",
                content=[
                    BetaRedactedThinkingBlock(
                        data="redacted", type="redacted_thinking"
                    ),
                    TextBlockParam(text="Result."),
                ],
            )
        ]
        result = _to_openai_messages(messages)
        assert result[0]["content"] == "Result."


class TestToOpenaiTools:
    def test_converts_tool_collection(self) -> None:
        tool_collection = MagicMock()
        tool_collection.to_params.return_value = [
            {
                "name": "click",
                "description": "Click an element",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            }
        ]
        result = _to_openai_tools(tool_collection)
        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click an element",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            },
        }

    def test_strips_cache_control(self) -> None:
        tool_collection = MagicMock()
        tool_collection.to_params.return_value = [
            {
                "name": "screenshot",
                "description": "Take screenshot",
                "input_schema": {
                    "type": "object",
                    "cache_control": {"type": "ephemeral"},
                },
            }
        ]
        result = _to_openai_tools(tool_collection)
        assert "cache_control" not in result[0]["function"]["parameters"]


class TestFromOpenaiResponse:
    def test_text_only_response(self) -> None:
        completion = _make_completion(content="Hello!")
        result = _from_openai_response(completion)
        assert result.role == "assistant"
        assert result.content == "Hello!"
        assert result.stop_reason == "end_turn"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_tool_call_response(self) -> None:
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="click",
                    arguments='{"x": 100, "y": 200}',
                ),
            )
        ]
        completion = _make_completion(tool_calls=tool_calls, finish_reason="tool_calls")
        result = _from_openai_response(completion)
        assert result.role == "assistant"
        assert result.stop_reason == "tool_use"
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, ToolUseBlockParam)
        assert block.id == "call_1"
        assert block.name == "click"
        assert block.input == {"x": 100, "y": 200}

    def test_text_and_tool_calls(self) -> None:
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="screenshot", arguments="{}"),
            )
        ]
        completion = _make_completion(
            content="Let me take a screenshot.",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        result = _from_openai_response(completion)
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.content[0], TextBlockParam)
        assert isinstance(result.content[1], ToolUseBlockParam)

    def test_tool_call_extra_content_captured(self) -> None:
        tool_calls = [
            ChatCompletionMessageToolCall.model_validate(
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "move_mouse", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "sig-abc"}},
                }
            )
        ]
        completion = _make_completion(tool_calls=tool_calls, finish_reason="tool_calls")
        result = _from_openai_response(completion)
        assert isinstance(result.content, list)
        block = result.content[0]
        assert isinstance(block, ToolUseBlockParam)
        assert block.extra_content == {"google": {"thought_signature": "sig-abc"}}

    def test_tool_call_without_extra_content_is_none(self) -> None:
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="screenshot", arguments="{}"),
            )
        ]
        completion = _make_completion(tool_calls=tool_calls, finish_reason="tool_calls")
        result = _from_openai_response(completion)
        assert isinstance(result.content, list)
        block = result.content[0]
        assert isinstance(block, ToolUseBlockParam)
        assert block.extra_content is None

    def test_usage_captured(self) -> None:
        completion = _make_completion(
            content="ok", prompt_tokens=50, completion_tokens=100
        )
        result = _from_openai_response(completion)
        assert result.usage is not None
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 100

    def test_cached_tokens_are_subtracted_from_input(self) -> None:
        """OpenAI reports cached tokens as a SUBSET of prompt_tokens;
        `UsageParam` consumers expect disjoint fields (Anthropic style).
        Without the subtraction, cached tokens are counted and billed twice
        (statistics callback, reporting)."""
        completion = _make_completion(
            content="ok",
            prompt_tokens=1000,
            completion_tokens=50,
            cached_tokens=400,
        )
        result = _from_openai_response(completion)
        assert result.usage is not None
        assert result.usage.input_tokens == 600  # 1000 - 400 cached
        assert result.usage.cache_read_input_tokens == 400
        assert result.usage.output_tokens == 50

    def test_cached_tokens_never_drive_input_negative(self) -> None:
        completion = _make_completion(
            content="ok", prompt_tokens=100, completion_tokens=5, cached_tokens=150
        )
        result = _from_openai_response(completion)
        assert result.usage is not None
        assert result.usage.input_tokens == 0


class TestToOpenaiToolChoice:
    def test_specific_tool_becomes_forced_function(self) -> None:
        result = _to_openai_tool_choice({"type": "tool", "name": "emit_result"})
        assert result == {
            "type": "function",
            "function": {"name": "emit_result"},
        }

    def test_any_becomes_required(self) -> None:
        assert _to_openai_tool_choice({"type": "any"}) == "required"

    def test_auto_and_none_pass_through(self) -> None:
        assert _to_openai_tool_choice({"type": "auto"}) == "auto"
        assert _to_openai_tool_choice({"type": "none"}) == "none"

    def test_unknown_type_is_omitted(self) -> None:
        assert _to_openai_tool_choice({"type": "something_new"}) is None


class TestToReasoningEffort:
    def test_disabled_maps_to_none(self) -> None:
        assert _to_reasoning_effort({"type": "disabled"}) == "none"

    def test_adaptive_omits_parameter(self) -> None:
        assert _to_reasoning_effort({"type": "adaptive"}) is None

    def test_enabled_omits_parameter(self) -> None:
        assert _to_reasoning_effort({"type": "enabled", "budget_tokens": 1024}) is None

    def test_explicit_effort_passed_through(self) -> None:
        assert _to_reasoning_effort({"type": "adaptive", "effort": "high"}) == "high"


class TestOpenAIMessagesApi:
    def test_create_message_delegates_to_client(self) -> None:
        mock_client = MagicMock()
        completion = _make_completion(content="response")
        mock_client.chat.completions.create.return_value = completion

        api = OpenAIMessagesApi(client=mock_client)
        result = api.create_message(
            messages=[MessageParam(role="user", content="hello")],
            model_id="qwen2.5vl",
        )

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "qwen2.5vl"
        assert call_kwargs["stream"] is False
        assert result.content == "response"

    def test_tools_passed_when_provided(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        mock_tools = MagicMock()
        mock_tools.to_params.return_value = [
            {
                "name": "click",
                "description": "Click",
                "input_schema": {"type": "object"},
            }
        ]

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            tools=mock_tools,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "function"

    def test_optional_params_omitted_when_none(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs

    def _mock_tools(self) -> MagicMock:
        mock_tools = MagicMock()
        mock_tools.to_params.return_value = [
            {
                "name": "emit_result",
                "description": "Emit",
                "input_schema": {"type": "object"},
            }
        ]
        return mock_tools

    def test_forced_tool_choice_sent_in_openai_format(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            tools=self._mock_tools(),
            tool_choice={"type": "tool", "name": "emit_result"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "emit_result"},
        }

    def test_tool_choice_omitted_without_tools(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            tool_choice={"type": "tool", "name": "emit_result"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tool_choice" not in call_kwargs

    def test_thinking_disabled_sends_none_reasoning_effort(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            thinking={"type": "disabled"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "none"

    def test_thinking_adaptive_omits_reasoning_effort(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            thinking={"type": "adaptive"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs

    def test_provider_options_override_mapped_params(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        api = OpenAIMessagesApi(client=mock_client)
        api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="test",
            tools=self._mock_tools(),
            tool_choice={"type": "tool", "name": "emit_result"},
            thinking={"type": "disabled"},
            provider_options={
                "tool_choice": "auto",
                "reasoning_effort": "high",
            },
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["reasoning_effort"] == "high"
