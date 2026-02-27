"""Cache validation strategies for automatic cache invalidation.

This module provides an extensible validator pattern that allows users to
define custom cache invalidation logic. The system includes built-in validators
for common scenarios like step failure counts, overall failure rates, and stale caches.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from askui.models.shared.settings import CacheFile


class CacheValidator(ABC):
    """Abstract base class for cache validation strategies.

    Users can implement custom validators by subclassing this and implementing
    the should_invalidate method.
    """

    @abstractmethod
    def should_invalidate(
        self, cache_file: CacheFile, step_index: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if cache should be invalidated.

        Args:
            cache_file: The cache file with metadata and trajectory
            step_index: Optional step index where failure occurred

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return validator name for logging/debugging."""


class CompositeCacheValidator(CacheValidator):
    """Composite validator that combines multiple validation strategies.

    Invalidates cache if ANY of the validators returns True.
    Users can add custom validators via add_validator().
    """

    def __init__(self, validators: list[CacheValidator] | None = None) -> None:
        """Initialize composite validator.

        Args:
            validators: Optional list of validators to include
        """
        self.validators: list[CacheValidator] = validators or []

    def add_validator(self, validator: CacheValidator) -> None:
        """Add a validator to the composite.

        Args:
            validator: The validator to add
        """
        self.validators.append(validator)

    def should_invalidate(
        self, cache_file: CacheFile, step_index: int | None = None
    ) -> tuple[bool, str | None]:
        """Check all validators, invalidate if any returns True.

        Args:
            cache_file: The cache file with metadata and trajectory
            step_index: Optional step index where failure occurred

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
            If multiple validators trigger, reasons are combined with "; "
        """
        reasons = []
        for validator in self.validators:
            should_inv, reason = validator.should_invalidate(cache_file, step_index)
            if should_inv and reason:
                reasons.append(f"{validator.get_name()}: {reason}")

        if reasons:
            return True, "; ".join(reasons)
        return False, None

    def get_name(self) -> str:
        """Return validator name."""
        return "CompositeValidator"


# Built-in validators


class StepFailureCountValidator(CacheValidator):
    """Invalidate if same step fails too many times.

    This validator counts how many times a specific step has failed
    and invalidates the cache if it exceeds the threshold.
    """

    def __init__(self, max_failures_per_step: int = 3) -> None:
        """Initialize validator.

        Args:
            max_failures_per_step: Maximum number of failures allowed per step
        """
        self.max_failures_per_step = max_failures_per_step

    def should_invalidate(
        self, cache_file: CacheFile, step_index: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if step has failed too many times.

        Args:
            cache_file: The cache file with metadata and trajectory
            step_index: The step index to check (required for this validator)

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
        """
        if step_index is None:
            return False, None

        # Count failures at this specific step
        failures_at_step = sum(
            1 for f in cache_file.metadata.failures if f.step_index == step_index
        )

        if failures_at_step >= self.max_failures_per_step:
            return (
                True,
                f"Step {step_index} failed {failures_at_step} times "
                f"(max: {self.max_failures_per_step})",
            )
        return False, None

    def get_name(self) -> str:
        """Return validator name."""
        return "StepFailureCount"


class TotalFailureRateValidator(CacheValidator):
    """Invalidate if overall failure rate is too high.

    This validator calculates the ratio of failures to execution attempts
    and invalidates if the rate exceeds the threshold after a minimum
    number of attempts.
    """

    def __init__(self, min_attempts: int = 10, max_failure_rate: float = 0.5) -> None:
        """Initialize validator.

        Args:
            min_attempts: Minimum execution attempts before checking rate
            max_failure_rate: Maximum acceptable failure rate (0.0 to 1.0)
        """
        self.min_attempts = min_attempts
        self.max_failure_rate = max_failure_rate

    def should_invalidate(
        self,
        cache_file: CacheFile,
        step_index: int | None = None,  # noqa: ARG002
    ) -> tuple[bool, str | None]:
        """Check if overall failure rate is too high.

        Args:
            cache_file: The cache file with metadata and trajectory
            step_index: Unused for this validator

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
        """
        attempts = cache_file.metadata.execution_attempts
        if attempts < self.min_attempts:
            return False, None

        failures = len(cache_file.metadata.failures)
        failure_rate = failures / attempts if attempts > 0 else 0.0

        if failure_rate > self.max_failure_rate:
            return (
                True,
                f"Failure rate {failure_rate:.1%} exceeds "
                f"{self.max_failure_rate:.1%} after {attempts} attempts",
            )
        return False, None

    def get_name(self) -> str:
        """Return validator name."""
        return "TotalFailureRate"


class StaleCacheValidator(CacheValidator):
    """Invalidate if cache is old and has failures.

    This validator checks if a cache hasn't been successfully executed
    in a long time AND has failures. Caches without failures are not
    considered stale regardless of age.
    """

    def __init__(self, max_age_days: int = 30) -> None:
        """Initialize validator.

        Args:
            max_age_days: Maximum age in days for cache with failures
        """
        self.max_age_days = max_age_days

    def should_invalidate(
        self,
        cache_file: CacheFile,
        step_index: int | None = None,  # noqa: ARG002
    ) -> tuple[bool, str | None]:
        """Check if cache is stale (old + has failures).

        Args:
            cache_file: The cache file with metadata and trajectory
            step_index: Unused for this validator

        Returns:
            Tuple of (should_invalidate: bool, reason: Optional[str])
        """
        if not cache_file.metadata.last_executed_at:
            return False, None

        if not cache_file.metadata.failures:
            return False, None  # No failures, age doesn't matter

        # Ensure last_executed_at is timezone-aware
        last_executed = cache_file.metadata.last_executed_at
        if last_executed.tzinfo is None:
            last_executed = last_executed.replace(tzinfo=timezone.utc)

        age = datetime.now(tz=timezone.utc) - last_executed
        if age > timedelta(days=self.max_age_days):
            return (
                True,
                f"Cache not successfully executed in {age.days} days and has failures",
            )
        return False, None

    def get_name(self) -> str:
        """Return validator name."""
        return "StaleCache"
