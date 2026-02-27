# Callbacks

Callbacks provide hooks into the agent's conversation lifecycle, similar to PyTorch Lightning's callback system. Use them for logging, monitoring, custom metrics, or extending agent behavior.

## Usage

Subclass `ConversationCallback` and override the hooks you need:

```python
from askui import ComputerAgent, ConversationCallback

class MetricsCallback(ConversationCallback):
    def on_step_start(self, conversation, step_index):
        print(f"Step {step_index} starting...")

    def on_step_end(self, conversation, step_index, result):
        print(f"Step {step_index} finished: {result.status}")

with ComputerAgent(callbacks=[MetricsCallback()]) as agent:
    agent.act("Open the settings menu")
```

## Available Hooks

| Hook | When Called | Parameters |
|------|-------------|------------|
| `on_conversation_start` | After setup, before control loop | `conversation` |
| `on_conversation_end` | After control loop, before cleanup | `conversation` |
| `on_control_loop_start` | Before the iteration loop begins | `conversation` |
| `on_control_loop_end` | After the iteration loop ends | `conversation` |
| `on_step_start` | Before each step execution | `conversation`, `step_index` |
| `on_step_end` | After each step execution | `conversation`, `step_index`, `result` |
| `on_tool_execution_start` | Before tools are executed | `conversation`, `tool_names` |
| `on_tool_execution_end` | After tools are executed | `conversation`, `tool_names` |

### Parameters

- **`conversation`**: The `Conversation` instance with access to messages, settings, and state
- **`step_index`**: Zero-based index of the current step
- **`result`**: `SpeakerResult` containing `status`, `messages_to_add`, and `usage`
- **`tool_names`**: List of tool names being executed

## Example: Timing Callback

```python
import time
from askui import ComputerAgent, ConversationCallback

class TimingCallback(ConversationCallback):
    def __init__(self):
        self.start_time = None
        self.step_times = []

    def on_conversation_start(self, conversation):
        self.start_time = time.time()

    def on_step_start(self, conversation, step_index):
        self._step_start = time.time()

    def on_step_end(self, conversation, step_index, result):
        elapsed = time.time() - self._step_start
        self.step_times.append(elapsed)
        print(f"Step {step_index}: {elapsed:.2f}s")

    def on_conversation_end(self, conversation):
        total = time.time() - self.start_time
        print(f"Total: {total:.2f}s across {len(self.step_times)} steps")

with ComputerAgent(callbacks=[TimingCallback()]) as agent:
    agent.act("Search for documents")
```

## Multiple Callbacks

Pass multiple callbacks to combine behaviors:

```python
with ComputerAgent(callbacks=[TimingCallback(), MetricsCallback()]) as agent:
    agent.act("Complete the form")
```

Callbacks are called in the order they are provided.
