"""Unit tests for the OpenAI Responses API backend."""

import json
from unittest.mock import MagicMock

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from askui.models.openai.responses_api import (
    OpenAIResponsesApi,
    _from_response,
    _to_reasoning_param,
    _to_responses_input,
    _to_responses_tool_choice,
    _to_responses_tools,
)
from askui.models.shared.agent_message_param import (
    BetaRedactedThinkingBlock,
    BetaThinkingBlock,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.prompts import SystemPrompt


def _make_response(
    output: list,
    status: str = "completed",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached_tokens: int = 0,
) -> Response:
    return Response(
        id="resp_1",
        created_at=0.0,
        model="gpt-5.6-terra",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


def _text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text=text, annotations=[])
        ],
    )


def _reasoning_item() -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id="rs_1",
        type="reasoning",
        summary=[Summary(type="summary_text", text="thought about it")],
        encrypted_content="enc-blob",
    )


def _function_call() -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="fc_1",
        type="function_call",
        call_id="call_1",
        name="emit_result",
        arguments='{"a": 1}',
    )


class TestFromResponse:
    def test_text_only_returns_string_content(self) -> None:
        message = _from_response(_make_response([_text_message("hello")]))
        assert message.role == "assistant"
        assert message.content == "hello"
        assert message.stop_reason == "end_turn"

    def test_function_call_maps_to_tool_use(self) -> None:
        message = _from_response(
            _make_response([_reasoning_item(), _function_call()])
        )
        assert message.stop_reason == "tool_use"
        tool_uses = [
            b for b in message.content if isinstance(b, ToolUseBlockParam)
        ]
        assert len(tool_uses) == 1
        assert tool_uses[0].id == "call_1"
        assert tool_uses[0].input == {"a": 1}
        raw = tool_uses[0].extra_content["openai_responses_item"]
        assert raw["call_id"] == "call_1"

    def test_reasoning_item_preserved_and_summarized(self) -> None:
        message = _from_response(
            _make_response([_reasoning_item(), _function_call()])
        )
        thinking = [
            b for b in message.content if isinstance(b, BetaThinkingBlock)
        ]
        redacted = [
            b
            for b in message.content
            if isinstance(b, BetaRedactedThinkingBlock)
        ]
        assert thinking[0].thinking == "thought about it"
        assert json.loads(redacted[0].data)["encrypted_content"] == "enc-blob"

    def test_cached_tokens_subtracted_from_input(self) -> None:
        message = _from_response(
            _make_response(
                [_text_message("x")], input_tokens=100, cached_tokens=40
            )
        )
        assert message.usage.input_tokens == 60
        assert message.usage.cache_read_input_tokens == 40

    def test_malformed_arguments_pass_raw_string(self) -> None:
        call = _function_call()
        call.arguments = "not json"
        message = _from_response(_make_response([call]))
        tool_use = message.content[0]
        assert tool_use.input == {"raw_arguments": "not json"}


class TestToResponsesInput:
    def test_tool_result_becomes_function_call_output(self) -> None:
        items = _to_responses_input(
            [
                MessageParam(
                    role="user",
                    content=[
                        ToolResultBlockParam(
                            tool_use_id="call_1", content="done"
                        )
                    ],
                )
            ]
        )
        assert items == [
            {"type": "function_call_output", "call_id": "call_1", "output": "done"}
        ]

    def test_assistant_round_trip_reemits_raw_items(self) -> None:
        response = _make_response([_reasoning_item(), _function_call()])
        assistant = _from_response(response)
        items = _to_responses_input([assistant])
        assert items[0]["type"] == "reasoning"
        assert items[0]["encrypted_content"] == "enc-blob"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_1"

    def test_tool_use_without_raw_item_is_reconstructed(self) -> None:
        items = _to_responses_input(
            [
                MessageParam(
                    role="assistant",
                    content=[
                        ToolUseBlockParam(
                            id="call_2", name="do_it", input={"x": 2}
                        )
                    ],
                )
            ]
        )
        assert items == [
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "do_it",
                "arguments": '{"x": 2}',
            }
        ]

    def test_user_text_becomes_input_text(self) -> None:
        items = _to_responses_input(
            [MessageParam(role="user", content=[TextBlockParam(text="hi")])]
        )
        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]


class TestParamMapping:
    def test_forced_tool_choice_is_flat(self) -> None:
        assert _to_responses_tool_choice({"type": "tool", "name": "emit"}) == {
            "type": "function",
            "name": "emit",
        }

    def test_any_maps_to_required(self) -> None:
        assert _to_responses_tool_choice({"type": "any"}) == "required"

    def test_disabled_thinking_maps_to_effort_none(self) -> None:
        assert _to_reasoning_param({"type": "disabled"}) == {"effort": "none"}

    def test_adaptive_thinking_requests_summaries(self) -> None:
        assert _to_reasoning_param({"type": "adaptive"}) == {"summary": "auto"}

    def test_explicit_effort_passes_through(self) -> None:
        assert _to_reasoning_param({"effort": "high"}) == {
            "effort": "high",
            "summary": "auto",
        }

    def test_tools_are_flat_function_format(self) -> None:
        tools = MagicMock()
        tools.to_params.return_value = [
            {
                "name": "emit",
                "description": "d",
                "input_schema": {"type": "object", "cache_control": {}},
            }
        ]
        assert _to_responses_tools(tools) == [
            {
                "type": "function",
                "name": "emit",
                "parameters": {"type": "object"},
                "description": "d",
                "strict": False,
            }
        ]


class TestCreateMessage:
    def test_request_shape(self) -> None:
        client = MagicMock()
        client.responses.create.return_value = _make_response(
            [_text_message("ok")]
        )
        api = OpenAIResponsesApi(client=client)
        result = api.create_message(
            messages=[MessageParam(role="user", content="hi")],
            model_id="gpt-5.6-terra",
            system=SystemPrompt(prompt="be brief"),
            max_tokens=1024,
            thinking={"type": "adaptive"},
        )
        assert result.content == "ok"
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.6-terra"
        assert kwargs["instructions"] == "be brief"
        assert kwargs["store"] is False
        assert kwargs["include"] == ["reasoning.encrypted_content"]
        assert kwargs["max_output_tokens"] == 1024
        assert kwargs["reasoning"] == {"summary": "auto"}
        assert kwargs["input"] == [{"role": "user", "content": "hi"}]
