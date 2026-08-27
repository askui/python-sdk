# Caching (Experimental)

The caching mechanism allows you to record and replay agent action sequences (trajectories) for faster and more robust test execution. This feature is particularly useful for regression testing, where you want to replay known-good interaction sequences to verify that your application still behaves correctly.

## Overview

The caching system works by recording all tool use actions (mouse movements, clicks, typing, etc.) performed by the agent during an `act()` execution. These recorded sequences can then be replayed in subsequent executions, allowing the agent to skip the decision-making process and execute the actions directly.

## Caching Strategies

The caching mechanism supports three strategies, configured via the `caching_settings` parameter in the `act()` method:

- **`None`** (default): No caching is used. The agent executes normally without recording or replaying actions.
- **`"record"`**: Records all agent actions to a cache file for future replay.
- **`"execute"`**: Provides tools to the agent to list and execute previously cached trajectories.
- **`"auto"`**: Combines execute and record modes - the agent can use existing cached trajectories and will also record new ones.

## Configuration

Caching is configured using the `CachingSettings` class:

```python
from askui.models.shared.settings import (
    CachingSettings,
    CacheExecutionSettings,
    CacheWritingSettings,
)

caching_settings = CachingSettings(
    strategy="record",       # One of: "execute", "record", "auto", or None
    cache_dir=".askui_cache", # Directory to store cache files
    filename="my_test.json",  # Name of the trajectory for this test case
    execution_settings=CacheExecutionSettings(
        delay_time_between_actions=1.0  # Delay in seconds between each cached action
    ),
)
```

### Parameters

- **`strategy`**: The caching strategy to use (`"execute"`, `"record"`, `"auto"`, or `None`).
- **`cache_dir`**: Directory where cache files are stored. Defaults to `".askui_cache"`.
- **`filename`**: Name of the trajectory/cache file for this test case (the `.json` suffix is optional), resolved **relative to `cache_dir`**. It is the lookup key in `"execute"`/`"auto"` modes and the target filename in `"record"`/`"auto"` modes. If empty, no trajectory is auto-detected and recordings receive an auto-generated filename.

  The filename may include **subdirectories**, which lets you mirror your test tree. The SDK does not derive it automatically from the running test file — you (or your test harness) supply it — but the same value is used for both lookup and recording, so save and load always agree. For example, for a test at `tests/mytests_1/test_something.py` you can set `filename="mytests_1/test_something"` (nested directories are created on record):

  ```python
  caching_settings = CachingSettings(
      strategy="auto",
      cache_dir=".askui_cache",
      filename="mytests_1/test_something",  # -> .askui_cache/mytests_1/test_something.json
  )
  ```

  A pytest harness can derive this from the test path, e.g. `filename=str(Path(request.node.path).relative_to(rootdir).with_suffix(""))`.
- **`writing_settings`**: Configuration for cache recording (optional). See [Writing Settings](#writing-settings) below.
- **`execution_settings`**: Configuration for cache playback (optional). See [Execution Settings](#execution-settings) below.

### Writing Settings

The `CacheWritingSettings` class allows you to configure how cache files are recorded:

```python
from askui.models.shared.settings import CacheWritingSettings

writing_settings = CacheWritingSettings(
    filename="my_test.json"  # Name for the cache file (auto-generated if empty)
)
```

#### Parameters

- **`filename`**: Name of the cache file to write. Prefer setting `filename` directly on `CachingSettings` (the top-level `filename` takes precedence and is also used for trajectory lookup in `execute`/`auto` modes). If neither is specified, a timestamped filename will be generated automatically (format: `cached_trajectory_YYYYMMDDHHMMSSffffff.json`).
- **`parameter_identification_strategy`**: How dynamic values are turned into `{{parameters}}` when recording (`"llm"`, the default, or `"preset"`). See [Dynamic Parameters](#dynamic-parameters).

### Dynamic Parameters

While recording, some entered values (e.g. today's date, a one-time code) must be supplied fresh on each replay rather than replayed literally. These are turned into `{{parameter}}` placeholders and requested from the agent on execution.

With the default `"llm"` strategy, identification is deliberately **conservative and precision-first**:
- Only user-entered **free text** is considered (values typed into fields). Coordinates, key names, action/enum values, counts and tool names are never eligible, so they are never mis-parameterized.
- The model is instructed to parameterize a value **only** when replaying the recorded literal would clearly be wrong on a later run (dates relative to "now", generated IDs/tokens/OTPs, intentionally per-run identities). When in doubt, the value is left as a literal — recording zero parameters is normal and expected.
- Identified values are validated against the recorded candidates, so hallucinated or reformatted values are dropped.

If a value you expected to be parameterized was left literal (or vice-versa), the caching logs/report show what was recorded (see [Observability](#observability)); you can also switch to `"preset"` and template values yourself with the `{{name}}` syntax.

### Execution Settings

The `CacheExecutionSettings` class allows you to configure how cached trajectories are executed:

```python
from askui.models.shared.settings import CacheExecutionSettings

execution_settings = CacheExecutionSettings(
    delay_time_between_actions=1.0  # Delay in seconds between each action (default: 1.0)
)
```

#### Parameters

- **`delay_time_between_actions`**: The time to wait (in seconds) between executing consecutive cached actions during replay. This delay helps ensure UI elements can materialize before the next action is executed. Defaults to `1.0` seconds.

You can adjust this value based on your application's responsiveness:
- For faster applications or quick interactions, you might use a smaller delay (e.g., `0.2` or `0.5` seconds)
- For slower applications or complex UI updates, you might need a longer delay (e.g., `2.0` or `3.0` seconds)

> **Important:** the delay is a *playback* option, so it lives on `execution_settings`, **not** directly on `CachingSettings`:
>
> ```python
> caching_settings = CachingSettings(
>     strategy="execute",
>     execution_settings=CacheExecutionSettings(delay_time_between_actions=3.0),
> )
> ```
>
> Passing `delay_time_between_actions` directly to `CachingSettings(...)` is a mistake and now raises a validation error (previously it was silently ignored and the default of `1.0`s was used).

## Usage Examples

### Recording a Cache

Record agent actions to a cache file for later replay:

```python
from askui import ComputerAgent
from askui.models.shared.settings import CachingSettings

with ComputerAgent() as agent:
    agent.act(
        goal="Fill out the login form with username 'admin' and password 'secret123'",
        caching_settings=CachingSettings(
            strategy="record", # you could also use "auto" here
            filename="login_test.json",
        )
    )
```

After execution, a cache file will be created at `.askui_cache/login_test.json` containing all the tool use actions performed by the agent.

### Executing from Cache (Replaying)

Set `strategy="execute"` (or `"auto"`) and give the trajectory's `filename`. The SDK automatically looks up `<cache_dir>/<filename>`:

```python
from askui import ComputerAgent
from askui.models.shared.settings import CachingSettings

with ComputerAgent() as agent:
    agent.act(
        goal="Fill out the login form",
        caching_settings=CachingSettings(
            strategy="execute", # you could also use "auto" here
            filename="login_test.json",
        )
    )
```

If a usable trajectory with that name exists, the SDK surfaces its details (path
and required parameters) to the agent automatically in the first message, and the
agent replays it via the `CacheExecutor` before doing anything else — you no
longer need to describe available cache files in your goal prompt, and there is
no separate "list trajectories" tool. After replay, the agent verifies the
results (via the `verify_cache_execution` tool) and makes corrections if needed.

Behavior when a trajectory is **not** found:

- `strategy="execute"`: the agent performs the task normally (nothing is recorded).
- `strategy="auto"`: the agent is told no cache exists and performs the task
  normally, while recording the run to `filename` for next time. An existing but
  invalidated cache is re-recorded (self-healing) rather than replayed.

### Using Custom Execution Settings

You can customize the delay between cached actions to match your application's responsiveness:

```python
from askui import ComputerAgent
from askui.models.shared.settings import CachingSettings, CacheExecutionSettings

with ComputerAgent() as agent:
    agent.act(
        goal="Fill out the login form",
        caching_settings=CachingSettings(
            strategy="execute",
            execution_settings=CacheExecutionSettings(
                delay_time_between_actions=2.0  # Wait 2 seconds between each action
            ),
        )
    )
```

This is particularly useful when:
- Your application has animations or transitions that need time to complete
- UI elements take time to become interactive after appearing
- You're testing on slower hardware or environments

### Using Auto Strategy

Enable both reading and writing simultaneously:

```python
from askui import ComputerAgent
from askui.models.shared.settings import CachingSettings

with ComputerAgent() as agent:
    agent.act(
        goal="Complete the checkout process",
        caching_settings=CachingSettings(
            strategy="auto",
            filename="checkout_test.json",
        )
    )
```

In this mode:
- If a usable trajectory named `checkout_test.json` exists, it is replayed and no new cache file is written (to avoid overwriting the existing one)
- Otherwise, the agent performs the task normally and records the run to `checkout_test.json`

## Cache File Format

Cache files are JSON files containing an array of tool use blocks. Each block represents a single tool invocation with the following structure:

```json
[
    {
        "type": "tool_use",
        "id": "toolu_01AbCdEfGhIjKlMnOpQrStUv",
        "name": "computer",
        "input": {
            "action": "mouse_move",
            "coordinate": [150, 200]
        }
    },
    {
        "type": "tool_use",
        "id": "toolu_02AbCdEfGhIjKlMnOpQrStUv",
        "name": "computer",
        "input": {
            "action": "left_click"
        }
    },
    {
        "type": "tool_use",
        "id": "toolu_03AbCdEfGhIjKlMnOpQrStUv",
        "name": "computer",
        "input": {
            "action": "type",
            "text": "admin"
        }
    }
]
```

Note: Screenshot actions are excluded from cached trajectories as they don't modify the UI state.

## How It Works

### Write Mode

In write mode, the `CacheManager`:

1. Extracts tool use blocks from the full message history when the conversation ends
2. Writes them to a JSON file
3. Automatically skips writing if a cached execution was used (to avoid recording replays)

### Read Mode

In read mode:

1. The SDK checks whether a trajectory named `filename` exists in `cache_dir`
2. If a usable trajectory is found, its details (path and required parameters) are injected into the first user message, a special system prompt (`CACHE_USE_PROMPT`) is appended, and the `CacheExecutor` speaker plus the `verify_cache_execution` tool are wired up
3. The agent hands off to the `CacheExecutor` via the `switch_speaker` tool to replay the trajectory
4. During replay, each tool use block is executed sequentially with a configurable delay between actions (default: 1.0 seconds)
5. Screenshot and non-cacheable tools are skipped/paused during replay; if a non-cacheable step is encountered the agent executes it manually and (unless it was the last step) resumes replay
6. The agent is instructed to verify results after replay (via `verify_cache_execution`) and make corrections if needed; reporting failure invalidates the cache

The delay between actions can be customized using `CacheExecutionSettings` to accommodate different application response times.

## Observability

Caching explains what it is doing and why, via both standard logs and the
attached reporter(s) (so the information also appears in e.g. the HTML report,
not only on stderr). Reporter messages use the source/role `Cache`. You will see
events such as:

- **Cache hit**: `Cache hit: replaying 'login.json' (12 steps, 1 parameter(s), valid).`
- **Cache miss**: `No usable cached trajectory for 'login.json'; running normally and recording this run for next time.`
- **Pause on a non-cacheable step**: `Paused replay at step 4: the 'get_file_tool' tool cannot be replayed from cache; the agent will perform this step.`
- **Completion**: `Finished replaying 12 cached step(s); asking the agent to verify the result.`
- **Verification outcome (with the reason)**: `Cache verification FAILED for 'login.json' - the replay did not achieve the expected result. Agent's reason: the submit button was missing. The cache will be invalidated so it is not reused.`
- **Invalidation / recording**: `Cache invalidated and will not be reused: ...` and `Recorded trajectory to 'login.json' (12 steps, 1 parameter(s): current_date).`

In particular, verification failures now include the agent's explanation and the
affected cache file, instead of an unexplained `Cache verification failed!`.

## Limitations

- **UI State Sensitivity**: Cached trajectories assume the UI is in the same state as when they were recorded. If the UI has changed, the replay may fail or produce incorrect results.
- **Verification Required**: After executing a cached trajectory, the agent should verify that the results are correct, as UI changes may cause partial failures.

## Example: Complete Test Workflow

Here's a complete example showing how to record and replay a test:

```python
from askui import ComputerAgent
from askui.models.shared.settings import (
    CachingSettings,
    CacheExecutionSettings,
)

# Step 1: Record a successful login flow
print("Recording login flow...")
with ComputerAgent() as agent:
    agent.act(
        goal="Navigate to the login page and log in with username 'testuser' and password 'testpass123'",
        caching_settings=CachingSettings(
            strategy="record",
            cache_dir="test_cache",
            filename="user_login.json",
        )
    )

# Step 2: Later, replay the login flow for regression testing
print("\nReplaying login flow for regression test...")
with ComputerAgent() as agent:
    agent.act(
        goal="Log in to the application.",
        caching_settings=CachingSettings(
            strategy="execute",
            cache_dir="test_cache",
            filename="user_login.json",
            execution_settings=CacheExecutionSettings(
                delay_time_between_actions=2.0
            ),
        )
    )
```
