# Multiple Target Computers

A single `ComputerAgent` can drive **one or more machines** through the `agent_os_target_computers` argument. Each entry is an Agent OS *target computer* identified by a stable `computer_id`. This lets one agent (and one `act()` run) coordinate work across several machines — for example, research something on one computer and write up the findings on another.

## Target types

| Target | What it does |
|--------|--------------|
| `LocalComputerTarget` | Manages an Agent OS controller subprocess on **this** machine. At most one per agent. |
| `RemoteComputerTarget` | Points at an Agent OS controller already running on **another** machine, reachable over gRPC. No process management — the controller must already be running. |

Both are importable from `askui.tools.askui`:

```python
from askui.tools.askui import LocalComputerTarget, RemoteComputerTarget
```

## The active target

At any moment exactly **one** target is *active* and receives all explicit calls (`click`, `type`, `keyboard`, ...). The **first** entry in `agent_os_target_computers` is the initial active target.

```python
from askui import ComputerAgent
from askui.tools.askui import LocalComputerTarget, RemoteComputerTarget

with ComputerAgent(
    agent_os_target_computers=[
        LocalComputerTarget(computer_id="local-box"),
        RemoteComputerTarget(
            address="192.168.1.42:26000",
            description="Remote box with a text editor open",
            computer_id="remote-box",
        ),
    ],
) as agent:
    # "local-box" is active by default (first in the list).
    agent.click("Submit button")

    # Permanently switch the active target.
    agent.tools.os.switch_agent_os_target_computer("remote-box")
    agent.type("Typed on the remote box")

    # Temporarily switch for a block, then restore the previous target on exit.
    with agent.tools.os.temporary_select("local-box"):
        agent.act("Open the settings menu")
    # "remote-box" is active again here.
```

Connections to all registered targets stay open across switches — switching only changes which connection future actions are routed to.

## Letting `act()` orchestrate across machines

The `act()` model is given three extra tools so it can move between machines on its own:

- `list_agent_os_target_computers` — discover the available targets and their `computer_id`s.
- `switch_agent_os_target_computer` — make a target active.
- `get_current_computer_target_id` — check which target is active.

Give each target a clear `description` so the model knows what each machine is for:

```python
with ComputerAgent(
    agent_os_target_computers=[
        LocalComputerTarget(computer_id="research-box"),
        RemoteComputerTarget(
            address="192.168.1.42:26000",
            description="Writer box with a text editor open",
            computer_id="writer-box",
        ),
    ],
) as agent:
    agent.act(
        "On research-box, open a browser, google 'askui', and read the top "
        "results to gather key facts about what AskUI is, what it does, and "
        "notable features. Then switch to writer-box and write a Markdown "
        "document titled 'AskUI Findings' summarizing those facts as a "
        "bulleted list in the open text editor."
    )
```

## Runtime helpers

These are available on `agent.tools.os`:

| Method | Purpose |
|--------|---------|
| `switch_agent_os_target_computer(computer_id)` | Make a target active and keep it active. Returns the now-active `ComputerTarget`. |
| `temporary_select(computer_id)` | Context manager that activates a target for a `with` block and restores the previously active one on exit (even if the block raises). |
| `get_current_computer_target_id()` | Return the `computer_id` of the active target. |
| `describe_agent_os_target_computers()` | Return a readable description of every registered target. |
| `add_agent_os_target_computer(target)` | Register an additional target at runtime (auto-connects if the agent is already connected). |
| `reset_agent_os_target_computers([...])` | Disconnect and replace the registered target list. |

## Constraints

- At least one target must be registered.
- At most one `LocalComputerTarget` per agent (any number of `RemoteComputerTarget`s is allowed).
- All `computer_id`s must be unique, and all remote `address`es must be unique.
- If `computer_id` is omitted, it defaults to the target's auto-generated `session_guid`.
- When `agent_os_target_computers` is provided, the top-level `display` argument is ignored — set `display` on the individual targets instead.

## Full example

See [`examples/multi_target_computers.py`](../examples/multi_target_computers.py) for a runnable script covering both explicit switching and model-orchestrated workflows.
