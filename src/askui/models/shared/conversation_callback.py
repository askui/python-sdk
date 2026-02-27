"""Callback system for conversation execution hooks."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askui.models.shared.conversation import Conversation
    from askui.speaker.speaker import SpeakerResult


class ConversationCallback(ABC):
    """Base class for conversation callbacks.

    Subclass this and override the methods you need. All methods have
    empty default implementations, so you only need to override the
    hooks you're interested in.

    The callback methods are called in the following order:
    1. `on_conversation_start` - After setup, before control loop
    2. `on_control_loop_start` - Before the while loop begins
    3. For each step:
       a. `on_step_start` - Before step execution
       b. `on_tool_execution_start` - Before tools are executed (if any)
       c. `on_tool_execution_end` - After tools are executed (if any)
       d. `on_step_end` - After step execution
    4. `on_control_loop_end` - After the while loop ends
    5. `on_conversation_end` - Before cleanup

    Example:
        ```python
        class LoggingCallback(ConversationCallback):
            def on_step_start(self, conversation, step_index):
                print(f"Starting step {step_index}")

            def on_step_end(self, conversation, step_index, result):
                print(f"Step {step_index} completed: {result.status}")


        with ComputerAgent(callbacks=[LoggingCallback()]) as agent:
            agent.act("Open the settings menu")
        ```
    """

    @abstractmethod
    def on_conversation_start(self, conversation: "Conversation") -> None:
        """Called when conversation begins (after setup, before control loop).

        Args:
            conversation: The conversation instance with initialized state
        """

    @abstractmethod
    def on_conversation_end(self, conversation: "Conversation") -> None:
        """Called when conversation ends (after control loop, before cleanup).

        Args:
            conversation: The conversation instance
        """

    @abstractmethod
    def on_control_loop_start(self, conversation: "Conversation") -> None:
        """Called before the control loop starts iterating.

        Args:
            conversation: The conversation instance
        """

    @abstractmethod
    def on_control_loop_end(self, conversation: "Conversation") -> None:
        """Called after the control loop finishes (success or failure).

        Args:
            conversation: The conversation instance
        """

    @abstractmethod
    def on_step_start(self, conversation: "Conversation", step_index: int) -> None:
        """Called before each step execution.

        Args:
            conversation: The conversation instance
            step_index: Zero-based index of the current step
        """

    @abstractmethod
    def on_step_end(
        self,
        conversation: "Conversation",
        step_index: int,
        result: "SpeakerResult",
    ) -> None:
        """Called after each step execution.

        Args:
            conversation: The conversation instance
            step_index: Zero-based index of the completed step
            result: The result from the speaker
        """

    @abstractmethod
    def on_tool_execution_start(
        self, conversation: "Conversation", tool_names: list[str]
    ) -> None:
        """Called before tools are executed.

        Args:
            conversation: The conversation instance
            tool_names: Names of tools about to be executed
        """

    @abstractmethod
    def on_tool_execution_end(
        self, conversation: "Conversation", tool_names: list[str]
    ) -> None:
        """Called after tools are executed.

        Args:
            conversation: The conversation instance
            tool_names: Names of tools that were executed
        """
