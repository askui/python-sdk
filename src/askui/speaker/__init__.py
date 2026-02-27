"""Speaker module for conversation-based agent architecture.

This module provides the speaker pattern for managing conversation flow:
- `Speaker`: Abstract base class for conversation speakers
- `SpeakerResult`: Result of a speaker handling a conversation step
- `Speakers`: Collection and manager of speakers
- `Conversation`: Main orchestrator for conversation execution
- `AgentSpeaker`: Default speaker for LLM API calls
- `CacheExecutor`: Speaker for cached trajectory playback
"""

from .agent_speaker import AgentSpeaker
from .cache_executor import CacheExecutor
from .speaker import Speaker, SpeakerResult, Speakers

__all__ = [
    "AgentSpeaker",
    "CacheExecutor",
    "Speaker",
    "SpeakerResult",
    "Speakers",
]
