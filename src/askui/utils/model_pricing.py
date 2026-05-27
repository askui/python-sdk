"""Pricing information for model API calls."""

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, "ModelPricing"] = {}


class ModelPricing(BaseModel):
    """Cost per 1 million tokens for a model.

    Args:
        input_cost_per_million_tokens (float): Cost per 1M input tokens in Currency.
        output_cost_per_million_tokens (float): Cost per 1M output tokens in Currency.
        cache_write_cost_per_million_tokens (float): Cost per 1M cache write
            input tokens in Currency.
        cache_read_cost_per_million_tokens (float): Cost per 1M cache read
            input tokens in Currency.
        currency (str): descriptor of the currency. Defaults to ``"USD"``.
    """

    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    cache_write_cost_per_million_tokens: float
    cache_read_cost_per_million_tokens: float
    currency: str = "USD"

    @classmethod
    def for_model(
        cls,
        model_id: str,
        input_cost_per_million_tokens: float | None = None,
        output_cost_per_million_tokens: float | None = None,
        cache_write_cost_per_million_tokens: float | None = None,
        cache_read_cost_per_million_tokens: float | None = None,
    ) -> "ModelPricing | None":
        """Resolve pricing for a model.

        If all cost parameters are provided, creates a ``ModelPricing``
        with those values. Otherwise, looks up built-in defaults by
        ``model_id``.

        Args:
            model_id (str): The model identifier.
            input_cost_per_million_tokens (float | None, optional): Override
                cost in Currency per 1M input tokens.
            output_cost_per_million_tokens (float | None, optional): Override
                cost in Currency per 1M output tokens.
            cache_write_cost_per_million_tokens (float | None, optional):
                Override cost in Currency per 1M cache write input tokens.
            cache_read_cost_per_million_tokens (float | None, optional):
                Override cost in Currency per 1M cache read input tokens.

        Returns:
            ModelPricing | None: Resolved pricing, or ``None`` if no match
                and no overrides provided.
        """
        if (
            input_cost_per_million_tokens is not None
            and output_cost_per_million_tokens is not None
            and cache_write_cost_per_million_tokens is not None
            and cache_read_cost_per_million_tokens is not None
        ):
            return cls(
                input_cost_per_million_tokens=input_cost_per_million_tokens,
                output_cost_per_million_tokens=output_cost_per_million_tokens,
                cache_write_cost_per_million_tokens=cache_write_cost_per_million_tokens,
                cache_read_cost_per_million_tokens=cache_read_cost_per_million_tokens,
            )
        msg = "Not all Pricing values are set, trying to use default pricing instead"
        logger.warning(msg)

        return _DEFAULT_PRICING.get(model_id)


# taken from: https://platform.claude.com/docs/en/about-claude/models/overview
# last accessed: March 10, 2026
_DEFAULT_PRICING.update(
    {
        "claude-haiku-4-5-20251001": ModelPricing(
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=5.0,
            cache_write_cost_per_million_tokens=1.25,
            cache_read_cost_per_million_tokens=0.10,
        ),
        "claude-sonnet-4-5-20250929": ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=3.75,
            cache_read_cost_per_million_tokens=0.30,
        ),
        "claude-opus-4-5-20251101": ModelPricing(
            input_cost_per_million_tokens=5.0,
            output_cost_per_million_tokens=25.0,
            cache_write_cost_per_million_tokens=6.25,
            cache_read_cost_per_million_tokens=0.50,
        ),
        "claude-sonnet-4-6": ModelPricing(
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=3.75,
            cache_read_cost_per_million_tokens=0.30,
        ),
        "claude-opus-4-6": ModelPricing(
            input_cost_per_million_tokens=5.0,
            output_cost_per_million_tokens=25.0,
            cache_write_cost_per_million_tokens=6.25,
            cache_read_cost_per_million_tokens=0.50,
        ),
        "gpt-5.4": ModelPricing(
            input_cost_per_million_tokens=2.5,
            output_cost_per_million_tokens=15.0,
            cache_write_cost_per_million_tokens=2.5,
            cache_read_cost_per_million_tokens=0.25,
        ),
        "gpt-5.4-mini": ModelPricing(
            input_cost_per_million_tokens=0.75,
            output_cost_per_million_tokens=4.50,
            cache_write_cost_per_million_tokens=0.75,
            cache_read_cost_per_million_tokens=0.0075,
        ),
        "gpt-5.4-nano": ModelPricing(
            input_cost_per_million_tokens=0.20,
            output_cost_per_million_tokens=1.25,
            cache_write_cost_per_million_tokens=0.20,
            cache_read_cost_per_million_tokens=0.02,
        ),
    }
)
