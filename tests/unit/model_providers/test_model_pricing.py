"""Unit tests for model pricing resolution and cost calculation."""

from unittest.mock import MagicMock

import pytest

from askui.callbacks.conversation_statistics_callback import (
    ConversationStatisticsCallback,
    UsageSummary,
)
from askui.models.shared.agent_message_param import UsageParam
from askui.speaker.speaker import SpeakerResult
from askui.utils.model_pricing import ModelPricing


class TestModelPricingForModel:
    @pytest.mark.parametrize(
        (
            "model_id",
            "expected_input",
            "expected_output",
            "expected_cache_write",
            "expected_cache_read",
        ),
        [
            ("claude-haiku-4-5-20251001", 1.0, 5.0, 1.25, 0.10),
            ("claude-sonnet-4-5-20250929", 3.0, 15.0, 3.75, 0.30),
            ("claude-opus-4-5-20251101", 5.0, 25.0, 6.25, 0.50),
            ("claude-sonnet-4-6", 3.0, 15.0, 3.75, 0.30),
            ("claude-opus-4-6", 5.0, 25.0, 6.25, 0.50),
            ("claude-haiku-5-5", 0.10, 0.50, 0.125, 0.01),
        ],
    )
    def test_known_model_returns_default_pricing(
        self,
        model_id: str,
        expected_input: float,
        expected_output: float,
        expected_cache_write: float,
        expected_cache_read: float,
    ) -> None:
        pricing = ModelPricing.for_model(model_id)
        assert pricing is not None
        assert pricing.input_cost_per_million_tokens == expected_input
        assert pricing.output_cost_per_million_tokens == expected_output
        assert pricing.cache_write_cost_per_million_tokens == expected_cache_write
        assert pricing.cache_read_cost_per_million_tokens == expected_cache_read

    @pytest.mark.parametrize(
        "model_id",
        ["unknown-model-v1", "", "claude-sonnet-4"],
    )
    def test_unknown_model_returns_none(self, model_id: str) -> None:
        assert ModelPricing.for_model(model_id) is None

    @pytest.mark.parametrize(
        (
            "model_id",
            "input_token_cost",
            "output_token_cost",
            "cache_write_cost",
            "cache_read_cost",
        ),
        [
            ("claude-sonnet-4-6", 99.0, 199.0, 9.0, 0.9),
            ("unknown-model", 1.0, 2.0, 0.5, 0.1),
        ],
    )
    def test_override_costs(
        self,
        model_id: str,
        input_token_cost: float,
        output_token_cost: float,
        cache_write_cost: float,
        cache_read_cost: float,
    ) -> None:
        pricing = ModelPricing.for_model(
            model_id,
            input_cost_per_million_tokens=input_token_cost,
            output_cost_per_million_tokens=output_token_cost,
            cache_write_cost_per_million_tokens=cache_write_cost,
            cache_read_cost_per_million_tokens=cache_read_cost,
        )
        assert pricing is not None
        assert pricing.input_cost_per_million_tokens == input_token_cost
        assert pricing.output_cost_per_million_tokens == output_token_cost
        assert pricing.cache_write_cost_per_million_tokens == cache_write_cost
        assert pricing.cache_read_cost_per_million_tokens == cache_read_cost


def _get_usage_summary(reporter_mock: MagicMock) -> UsageSummary:
    return reporter_mock.add_usage_summary.call_args[0][0]  # type: ignore[no-any-return]


def _assert_close(
    actual: float | None, expected: float, tolerance: float = 1e-12
) -> None:
    assert actual is not None
    assert abs(actual - expected) <= tolerance


class TestConversationStatisticsCallbackCost:
    def _make_callback(
        self, pricing: ModelPricing | None = None
    ) -> tuple[ConversationStatisticsCallback, MagicMock]:
        reporter = MagicMock()
        callback = ConversationStatisticsCallback(reporter=reporter, pricing=pricing)
        return callback, reporter

    @pytest.mark.parametrize(
        (
            "input_tokens",
            "output_tokens",
            "input_rate",
            "output_rate",
            "expected_input_cost",
            "expected_output_cost",
        ),
        [
            (1_000_000, 100_000, 3.0, 15.0, 3.0, 1.5),
            (50_000, 10_000, 15.0, 75.0, 0.75, 0.75),
            (0, 0, 3.0, 15.0, 0.0, 0.0),
        ],
    )
    def test_cost_calculation(
        self,
        input_tokens: int,
        output_tokens: int,
        input_rate: float,
        output_rate: float,
        expected_input_cost: float,
        expected_output_cost: float,
    ) -> None:
        pricing = ModelPricing(
            input_cost_per_million_tokens=input_rate,
            output_cost_per_million_tokens=output_rate,
            cache_write_cost_per_million_tokens=input_rate * 1.25,
            cache_read_cost_per_million_tokens=input_rate * 0.1,
        )
        callback, reporter = self._make_callback(pricing)
        conversation = MagicMock(conversation_id="cost-check")
        callback.on_conversation_start(conversation)
        callback.on_step_end(
            conversation=conversation,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(
                    input_tokens=input_tokens, output_tokens=output_tokens
                ),
            ),
        )
        callback.on_conversation_end(conversation)

        summary = _get_usage_summary(reporter)
        _assert_close(summary.input_token_cost, expected_input_cost)
        _assert_close(summary.output_token_cost, expected_output_cost)
        _assert_close(summary.total_cost, expected_input_cost + expected_output_cost)
        assert summary.currency == "USD"
        assert summary.input_cost_per_million_tokens == input_rate
        assert summary.output_cost_per_million_tokens == output_rate
        assert summary.cache_write_cost_per_million_tokens == input_rate * 1.25
        assert summary.cache_read_cost_per_million_tokens == input_rate * 0.1

    def test_no_cost_when_pricing_none(self) -> None:
        callback, reporter = self._make_callback(pricing=None)
        conversation = MagicMock(conversation_id="no-cost-check")
        callback.on_conversation_start(conversation)
        callback.on_step_end(
            conversation=conversation,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(input_tokens=500, output_tokens=200),
            ),
        )
        callback.on_conversation_end(conversation)

        summary = _get_usage_summary(reporter)

        assert summary.total_cost is None
        assert summary.input_token_cost is None
        assert summary.output_token_cost is None
        assert summary.currency is None

    def test_none_tokens_treated_as_zero(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=3.75,
            cache_read_cost_per_million_tokens=0.3,
        )
        callback, reporter = self._make_callback(pricing)
        conversation = MagicMock(conversation_id="none-tokens-check")
        callback.on_conversation_start(conversation)
        callback.on_step_end(
            conversation=conversation,
            step_index=0,
            result=SpeakerResult(status="continue", usage=UsageParam()),
        )
        callback.on_conversation_end(conversation)

        summary = _get_usage_summary(reporter)
        assert summary.total_cost == 0.0

    def test_tracks_per_step_per_conversation_and_total_usage(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=3.75,
            cache_read_cost_per_million_tokens=0.3,
        )
        callback, reporter = self._make_callback(pricing)
        conversation = MagicMock(conversation_id="conversation-1")

        callback.on_conversation_start(conversation)
        callback.on_step_end(
            conversation=conversation,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(input_tokens=100, output_tokens=20),
            ),
        )
        callback.on_step_end(
            conversation=conversation,
            step_index=1,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(input_tokens=50, output_tokens=10),
            ),
        )
        callback.on_conversation_end(conversation)

        summary = _get_usage_summary(reporter)
        assert summary.input_tokens == 150
        assert summary.output_tokens == 30
        _assert_close(summary.total_cost, 0.0009)
        assert summary.per_conversation_summaries is not None
        assert len(summary.per_conversation_summaries) == 1

        per_conversation_summary = summary.per_conversation_summaries[0]
        assert per_conversation_summary.conversation_id == "conversation-1"
        assert per_conversation_summary.conversation_index == 1
        assert per_conversation_summary.input_tokens == 150
        assert per_conversation_summary.output_tokens == 30
        _assert_close(per_conversation_summary.total_cost, 0.0009)
        assert len(per_conversation_summary.step_summaries) == 2
        assert per_conversation_summary.started_at is not None
        assert per_conversation_summary.ended_at is not None
        assert per_conversation_summary.ended_at >= per_conversation_summary.started_at

        first_step = per_conversation_summary.step_summaries[0]
        assert first_step.step_index == 0
        assert first_step.input_tokens == 100
        assert first_step.output_tokens == 20
        _assert_close(first_step.total_cost, 0.0006)
        assert first_step.currency == "USD"

        second_step = per_conversation_summary.step_summaries[1]
        assert second_step.step_index == 1
        assert second_step.input_tokens == 50
        assert second_step.output_tokens == 10
        _assert_close(second_step.total_cost, 0.0003)
        assert second_step.currency == "USD"

    def test_accumulates_multiple_conversations(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=3.75,
            cache_read_cost_per_million_tokens=0.3,
        )
        callback, reporter = self._make_callback(pricing)
        conversation_1 = MagicMock(conversation_id="conversation-1")
        conversation_2 = MagicMock(conversation_id="conversation-2")

        callback.on_conversation_start(conversation_1)
        callback.on_step_end(
            conversation=conversation_1,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(input_tokens=200, output_tokens=40),
            ),
        )
        callback.on_conversation_end(conversation_1)

        callback.on_conversation_start(conversation_2)
        callback.on_step_end(
            conversation=conversation_2,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(input_tokens=300, output_tokens=60),
            ),
        )
        callback.on_conversation_end(conversation_2)

        summary = _get_usage_summary(reporter)
        assert summary.input_tokens == 500
        assert summary.output_tokens == 100
        _assert_close(summary.total_cost, 0.003)
        assert summary.per_conversation_summaries is not None
        assert len(summary.per_conversation_summaries) == 2
        assert summary.per_conversation_summaries[0].conversation_id == "conversation-1"
        assert summary.per_conversation_summaries[1].conversation_id == "conversation-2"
        for per_conversation_summary in summary.per_conversation_summaries:
            assert per_conversation_summary.started_at is not None
            assert per_conversation_summary.ended_at is not None
            assert (
                per_conversation_summary.ended_at >= per_conversation_summary.started_at
            )

    def test_includes_cache_costs_from_provider_pricing(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=6.0,
            cache_read_cost_per_million_tokens=0.6,
        )
        callback, reporter = self._make_callback(pricing)
        conversation = MagicMock(conversation_id="cache-cost-check")

        callback.on_conversation_start(conversation)
        callback.on_step_end(
            conversation=conversation,
            step_index=0,
            result=SpeakerResult(
                status="continue",
                usage=UsageParam(
                    input_tokens=100_000,
                    output_tokens=50_000,
                    cache_creation_input_tokens=200_000,
                    cache_read_input_tokens=300_000,
                ),
            ),
        )
        callback.on_conversation_end(conversation)

        summary = _get_usage_summary(reporter)
        _assert_close(summary.input_token_cost, 0.3)
        _assert_close(summary.output_token_cost, 0.75)
        _assert_close(summary.cache_write_token_cost, 1.2)
        _assert_close(summary.cache_read_token_cost, 0.18)
        _assert_close(summary.total_cost, 2.43)
        assert summary.cache_write_cost_per_million_tokens == 6.0
        assert summary.cache_read_cost_per_million_tokens == 0.6
