"""Example demonstrating the secret scope.

Secrets let the agent *use* sensitive values (e.g. type a password) while the value is
never sent to the LLM. The model only ever sees the placeholder
``<|secret|>NAME<|secret|>``; the real value is substituted into tool calls at execution
time. Literal values are also redacted from the LLM history, tool outputs and the cache.

Two ways to provide a secret are shown:
1. Hardcoded value (handy for a quick demo only).
2. Read from an environment variable (recommended for real usage).

Required environment variables (see .env):
- ASKUI_WORKSPACE_ID, ASKUI_TOKEN - for the default AskUI providers
- APP_PASSWORD - the example login password, read at runtime (see below)

Set the secret in your shell before running (do NOT hardcode real secrets in code):
    export APP_PASSWORD="my-real-password"

Note: a secret typed into a *visible* field can still appear in screenshots sent to the
model; on-screen secrets cannot currently be hidden.
"""

import logging
import os

from askui import ComputerAgent, Secret

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


def secrets_from_env() -> list[Secret]:
    """Build secrets by reading their values from environment variables.

    This is the recommended approach: keep real values out of source code and pass them
    in from the environment. We only *read* env vars here (never set them in code).
    """
    return [
        Secret(
            name="password",
            value=os.environ["APP_PASSWORD"],
            description="the application login password",
        ),
    ]


def secrets_hardcoded() -> list[Secret]:
    """Build secrets with hardcoded values.

    Convenient for a quick local demo, but never commit real secrets to source control.
    """
    return [
        Secret(
            name="password",
            value="hunter2-demo-only",
            description="the application login password",
        ),
    ]


def run_with_agent_level_secrets() -> None:
    """Define secrets on the agent so they apply to every act()/type() call."""
    with ComputerAgent(secrets=secrets_from_env()) as agent:
        # The agent emits the placeholder; the real value is typed at execution time.
        agent.act("Log in as 'admin' using the password")

        # Deterministic typing also resolves the placeholder at the OS boundary.
        agent.click("Password field")
        agent.type("<|secret|>password<|secret|>")


def run_with_per_call_secrets() -> None:
    """Provide secrets only for a single act() call (overrides agent-level on name)."""
    with ComputerAgent() as agent:
        agent.act(
            "Enter the one-time PIN into the verification field",
            secrets=[
                Secret(
                    name="pin",
                    value=os.environ.get("APP_OTP", "000000"),
                    description="6-digit one-time PIN",
                ),
            ],
        )


def run_with_hardcoded_secret() -> None:
    """Quick demo using a hardcoded secret value (not for production)."""
    with ComputerAgent(secrets=secrets_hardcoded()) as agent:
        agent.act("Log in using the password")


if __name__ == "__main__":
    # Pick the variant you want to try:
    run_with_agent_level_secrets()
    # run_with_per_call_secrets()
    # run_with_hardcoded_secret()
