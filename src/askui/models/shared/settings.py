from datetime import datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from askui.models.shared.agent_message_param import (
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUseBlockParam,
    UsageParam,
)
from askui.models.shared.prompts import (
    ActSystemPrompt,
    GetSystemPrompt,
    LocateSystemPrompt,
)


class Resolution(NamedTuple):
    """Screen resolution for image scaling operations.

    Used by vision models to scale images to a standard size before processing.

    Args:
        width (int): The width in pixels.
        height (int): The height in pixels.
    """

    width: int
    height: int


DEFAULT_LOCATE_RESOLUTION = Resolution(1280, 800)
DEFAULT_GET_RESOLUTION = Resolution(1280, 800)

CACHING_STRATEGY = Literal["execute", "record", "both"]
CACHE_PARAMETER_IDENTIFICATION_STRATEGY = Literal["llm", "preset"]
CACHING_VISUAL_VERIFICATION_METHOD = Literal["phash", "ahash", "none"]


class MessageSettings(BaseModel):
    """Settings for message creation in ActModel operations.

    These settings control how messages are sent to the underlying model API
    during agent action execution.

    Args:
        max_tokens (int): Maximum number of tokens the model can generate in
            its response. Higher values allow longer responses but increase
            cost and latency. Default: 8192.
        system (ActSystemPrompt | None): Custom system prompt that defines
            the agent's behavior and capabilities. If None, uses the default
            agent system prompt.
        thinking (ThinkingConfigParam | None): Configuration for thinking feature.
            When enabled, the model spends more time reasoning before responding.
        tool_choice (ToolChoiceParam | None): How the model should select
            tools. Options include "auto", "any", or a specific tool name.
        temperature (float | None): Controls response randomness. 0.0 produces
            deterministic outputs, 1.0 produces creative/varied outputs.
            Default: None (uses model default).
        provider_options (dict[str, Any] | None): Provider-specific options.
            Each provider can define its own keys. Common options include:
            - "betas": List of beta features to enable (e.g., for Anthropic)
            Default: None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_tokens: int = 8192
    system: ActSystemPrompt | None = None
    thinking: ThinkingConfigParam | None = None
    tool_choice: ToolChoiceParam | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_options: dict[str, Any] | None = None


class ActSettings(BaseModel):
    """Settings for ActModel operations (agent actions).

    Controls the behavior of agent action execution, including how messages
    are created and sent to the underlying model.

    Args:
        messages (MessageSettings): Settings for message creation including
            max tokens, temperature, and system prompt configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: MessageSettings = Field(default_factory=MessageSettings)


class GetSettings(BaseModel):
    """Settings for GetModel operations (data extraction from images/PDFs).

    Controls how GetModel extracts information from visual sources like
    screenshots, images, and PDF documents.

    Args:
        max_tokens (int): Maximum tokens for the model response. Higher values
            allow more detailed extractions but increase cost. Default: 4096.
            Note: Not all providers currently use this setting.
        temperature (float): Controls response randomness. Lower values (e.g.,
            0.0) produce more deterministic extractions, higher values allow
            more varied responses. Default: 0.5. Range: 0.0-1.0.
            Note: Not all providers currently use this setting.
        system_prompt (GetSystemPrompt | None): Custom system prompt for the
            extraction task. If None, uses the default extraction prompt.
        timeout (float | None): Timeout in seconds for the API call. If None,
            uses the provider's default timeout.
            Note: Reserved for future use; not currently implemented.
        resolution (Resolution): Target resolution for scaling images before
            processing. Images are scaled to fit within this resolution while
            maintaining aspect ratio. This affects quality vs. token usage.
            Default: 1280x800.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_tokens: int = 4096
    temperature: float = Field(default=0.5, ge=0.0, le=1.0)
    system_prompt: GetSystemPrompt | None = None
    timeout: float | None = None
    resolution: Resolution = DEFAULT_GET_RESOLUTION


class LocateSettings(BaseModel):
    """Settings for LocateModel operations (UI element location).

    Controls how LocateModel finds UI elements in screenshots based on
    text, images, or natural language descriptions.

    Args:
        query_type (str | None): Type of query being performed. Used by some
            providers to optimize detection strategies.
            Note: Reserved for future use; not currently implemented.
        confidence_threshold (float): Minimum confidence score (0.0-1.0) for
            a detection to be considered valid. Higher values reduce false
            positives but may miss valid elements. Default: 0.8.
            Note: Not all providers currently use this setting.
        max_detections (int): Maximum number of elements to return when
            multiple matches are found. Default: 10.
            Note: Not all providers currently use this setting.
        timeout (float | None): Timeout in seconds for the API call. If None,
            uses the provider's default timeout.
            Note: Reserved for future use; not currently implemented.
        system_prompt (LocateSystemPrompt | None): Custom system prompt for
            element location. If None, uses the default locate prompt.
            Note: Not all providers currently use this setting.
        resolution (Resolution): Target resolution for scaling images before
            processing. Images are scaled to fit within this resolution while
            maintaining aspect ratio. Default: 1280x800.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_type: str | None = None
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_detections: int = 10
    timeout: float | None = None
    system_prompt: LocateSystemPrompt | None = None
    resolution: Resolution = DEFAULT_LOCATE_RESOLUTION


class CacheFailure(BaseModel):
    """Record of a single cache execution failure.

    Args:
        timestamp: When the failure occurred
        step_index: Index of the step that failed
        error_message: Description of the failure
        failure_count_at_step: Running count of failures at this step
    """

    timestamp: datetime
    step_index: int
    error_message: str
    failure_count_at_step: int


class VisualValidationMetadata(BaseModel):
    enabled: bool
    method: CACHING_VISUAL_VERIFICATION_METHOD
    region_size: int


class CacheMetadata(BaseModel):
    """Metadata for a cache file including execution history and validation state.

    Args:
        version: Cache format version
        created_at: When the cache was created
        goal: Original goal text (may be parameterized)
        last_executed_at: When the cache was last executed
        token_usage: Accumulated token usage from recording
        execution_attempts: Total number of execution attempts
        failures: List of recorded failures
        is_valid: Whether cache is still valid
        invalidation_reason: Why cache was invalidated (if applicable)
        visual_validation: Visual validation configuration
    """

    version: str = "0.2"
    created_at: datetime
    goal: str | None = None
    last_executed_at: datetime | None = None
    token_usage: UsageParam | None = None
    execution_attempts: int = 0
    failures: list[CacheFailure] = Field(default_factory=list)
    is_valid: bool = True
    invalidation_reason: str | None = None
    visual_validation: VisualValidationMetadata | None = None


class CacheFile(BaseModel):
    """Complete cache file structure with metadata and trajectory.

    Args:
        metadata: Cache metadata and execution history
        trajectory: List of tool use blocks to execute
        cache_parameters: Dict mapping parameter names to descriptions
    """

    metadata: CacheMetadata
    trajectory: list[ToolUseBlockParam]
    cache_parameters: dict[str, str] = Field(default_factory=dict)


class CacheWritingSettings(BaseModel):
    """Settings for recording cache files.

    Args:
        filename: Name for the cache file (auto-generated if empty)
        parameter_identification_strategy: How to identify parameters("llm" or "preset")
        llm_parameter_id_api_provider: API provider for LLM parameter identification
        visual_verification_method: Visual hash method ("phash", "ahash", or "none")
        visual_validation_region_size: Size of region to hash around coordinates
    """

    filename: str = ""
    parameter_identification_strategy: CACHE_PARAMETER_IDENTIFICATION_STRATEGY = "llm"
    visual_verification_method: CACHING_VISUAL_VERIFICATION_METHOD = "phash"
    visual_validation_region_size: int = 100


class CacheExecutionSettings(BaseModel):
    """Settings for executing/replaying cached trajectories.

    Args:
        delay_time_between_actions: Delay in seconds between actions
        skip_visual_validation: Override to disable visual validation
        visual_validation_threshold: Max Hamming distance for validation
    """

    delay_time_between_actions: float = 1.0  # keep >1s to give UI time to materialize
    skip_visual_validation: bool = False
    visual_validation_threshold: int = 10


class CachingSettings(BaseModel):
    """Settings for caching agent action trajectories.

    Enables recording and replaying of agent actions for testing or
    performance optimization.

    Args:
        strategy (CACHING_STRATEGY | None): Caching mode. Options:
            - None: Caching disabled (default)
            - "execute": Replay actions from cache
            - "record": Record actions to cache
            - "both": Execute from cache if available, otherwise record
        cache_dir (str): Directory path for storing cache files.
            Default: ".askui_cache".
        writing_settings: Settings for cache recording (used in "record"/"both" modes)
        execution_settings: Settings for cache playback (used in "execute"/"both" modes)
    """

    strategy: CACHING_STRATEGY | None = None
    cache_dir: str = ".askui_cache"
    writing_settings: CacheWritingSettings | None = None
    execution_settings: CacheExecutionSettings | None = None
