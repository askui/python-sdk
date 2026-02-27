"""Caching utilities for agent trajectory recording and playback.

This module provides:
- `CacheManager`: High-level cache operations (recording, validation, playback)
- `CacheParameterHandler`: Parameter identification and substitution
- `CacheValidator`: Validation strategies for cache invalidation
"""

from .cache_manager import CacheManager
from .cache_parameter_handler import CacheParameterHandler
from .cache_validator import (
    CacheValidator,
    CompositeCacheValidator,
    StaleCacheValidator,
    StepFailureCountValidator,
    TotalFailureRateValidator,
)

__all__ = [
    "CacheManager",
    "CacheParameterHandler",
    "CacheValidator",
    "CompositeCacheValidator",
    "StaleCacheValidator",
    "StepFailureCountValidator",
    "TotalFailureRateValidator",
]
