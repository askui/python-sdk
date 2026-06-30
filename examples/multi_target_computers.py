"""Example demonstrating how to drive multiple target computers with one agent.

A single `ComputerAgent` can control one or more machines through the
`agent_os_target_computers` argument. Each entry is an Agent OS *target
computer* identified by a stable `computer_id`:

- `LocalComputerTarget` - manages an Agent OS controller subprocess on this
  machine (at most one per agent).
- `RemoteComputerTarget` - points at an Agent OS controller already running on
  another machine, reachable over gRPC.

At any moment exactly one target is *active* and receives all explicit calls
(`click`, `type`, `keyboard`, ...). The first target in the list is the initial
active one. You can change the active target at runtime in three ways:

1. `agent.tools.os.switch_agent_os_target_computer(computer_id)` - switch and
   keep the new target active.
2. `with agent.tools.os.temporary_select(computer_id): ...` - switch for the
   duration of a block, then restore the previously active target on exit.
3. Let `act()` orchestrate on its own - the model has `list_agent_os_target_computers`,
   `switch_agent_os_target_computer`, and `get_current_computer_target_id` tools.

Required environment variables (see .env):
- ASKUI_WORKSPACE_ID, ASKUI_TOKEN - for the default AskUI model stack
"""

import logging

from askui import ComputerAgent
from askui.reporting import SimpleHtmlReporter
from askui.tools.askui import LocalComputerTarget, RemoteComputerTarget

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


def explicit_switching() -> None:
    """Route explicit calls to specific machines via `switch`/`temporary_select`."""
    with ComputerAgent(
        agent_os_target_computers=[
            LocalComputerTarget(computer_id="local-box"),
            RemoteComputerTarget(
                address="10.0.24.11:26000",
                description="Remote box with a text editor open",
                computer_id="remote-box",
            ),
        ],
        reporters=[SimpleHtmlReporter()],
    ) as agent:
        agent.act("Take a screenshot on each machine that you are connected to")


def model_orchestrated() -> None:
    """Let `act()` decide when to switch between machines on its own."""
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


if __name__ == "__main__":
    # Pick the scenario to run. The remote examples expect an Agent OS
    # controller reachable at the configured address; adjust it to your setup.
    explicit_switching()
    # model_orchestrated()

    logger.info("Done!")
