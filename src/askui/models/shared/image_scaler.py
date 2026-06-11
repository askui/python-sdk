"""Type alias for image scaling callables used by VLM providers."""

from collections.abc import Callable

from PIL import Image

ImageScaler = Callable[[Image.Image], Image.Image]
"""Callable that preprocesses a screenshot before sending to a model."""
