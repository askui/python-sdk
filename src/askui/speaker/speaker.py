"""Base speaker class and result types for conversation architecture."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from pydantic import BaseModel, Field
from typing_extensions import Literal

from askui.models.shared.agent_message_param import MessageParam, UsageParam

if TYPE_CHECKING:
    from askui.models.shared.conversation import Conversation
    from askui.utils.caching.cache_manager import CacheManager

logger = logging.getLogger(__name__)

SPEAKER_RESULT_STATUS = Literal["continue", "switch_speaker", "done", "failed"]


class SpeakerResult(BaseModel):
    """Result of a speaker handling a conversation step.

    Attributes:
        status: Execution status
            - "continue": Continue with same speaker (recurse)
            - "switch_speaker": Switch to a different speaker
            - "done": Conversation finished successfully
            - "failed": Conversation failed with error
        next_speaker: Name of next speaker to switch to
            (required if status="switch_speaker")
        messages_to_add: Messages to add to conversation history
        usage: Token usage from this step (if applicable)
    """

    status: SPEAKER_RESULT_STATUS
    next_speaker: str | None = None
    messages_to_add: list[MessageParam] = Field(default_factory=list)
    usage: UsageParam | None = None


class Speaker(ABC):
    """Abstract base class for conversation speakers.

    A speaker handles specific types of conversation steps (e.g., normal agent
    API calls, cache execution, human-in-the-loop).

    Each speaker determines whether it can handle the current conversation state
    and executes one step when activated.
    """

    @abstractmethod
    def can_handle(self, conversation: "Conversation") -> bool:
        """Check if this speaker can handle the current conversation state.

        Args:
            conversation: The conversation instance with current state

        Returns:
            True if this speaker can handle the current state
        """
        ...

    @abstractmethod
    def handle_step(
        self, conversation: "Conversation", cache_manager: "CacheManager | None"
    ) -> SpeakerResult:
        """Execute one conversation step.

        Args:
            conversation: The conversation instance with current state
            cache_manager: Optional cache manager for recording/playback

        Returns:
            SpeakerResult indicating what to do next
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the speaker's name for logging and identification.

        Returns:
            Speaker name (e.g., "AgentSpeaker", "CacheExecutor")
        """
        ...


class Speakers:
    """Collection and manager of conversation speakers.

    Holds a dictionary of speakers and tracks the default speaker.
    Provides dictionary-like access to speakers by name.
    """

    def __init__(self, speakers: dict[str, Speaker] | None = None) -> None:
        # Lazy import to avoid circular dependency
        from .agent_speaker import AgentSpeaker

        self.speakers: dict[str, Speaker] = speakers or {"AgentSpeaker": AgentSpeaker()}
        self.default_speaker: str = (
            "AgentSpeaker"
            if "AgentSpeaker" in self.speakers
            else next(iter(self.speakers.keys()))
        )

    def add_speaker(self, speaker: Speaker) -> None:
        """Add a speaker to the collection."""
        self.speakers[speaker.get_name()] = speaker

    def get_names(self) -> list[str]:
        """Get list of all speaker names."""
        return list(self.speakers.keys())

    def __add__(self, other: "Speakers") -> "Speakers":
        """Combine two Speakers collections."""
        result = Speakers(speakers={})
        result.speakers = self.speakers | other.speakers
        result.default_speaker = self.default_speaker
        return result

    def __getitem__(self, name: str) -> Speaker:
        """Get speaker by name, falling back to default if not found."""
        if name in self.speakers:
            return self.speakers[name]
        msg = (
            f"Speaker {name} is not part of Speakers. "
            f"Will use default Speaker {self.default_speaker} instead"
        )
        logger.warning(msg)
        return self.speakers[self.default_speaker]

    def __contains__(self, name: str) -> bool:
        """Check if a speaker exists in the collection."""
        return name in self.speakers

    def __iter__(self) -> "Iterator[Speaker]":
        """Iterate over all speakers in the collection."""
        return iter(self.speakers.values())

    def reset_state(self) -> None:
        """Reset state for all speakers that have a reset_state method.

        This allows stateful speakers (e.g., CacheExecutor) to be reset
        between conversation runs.
        """
        for speaker in self.speakers.values():
            if hasattr(speaker, "reset_state") and callable(speaker.reset_state):
                speaker.reset_state()
