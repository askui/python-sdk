# Secrets

Secrets let an agent **use** a sensitive value (e.g. type a password) without ever exposing the value to the LLM.

You register a secret by name. The model only ever sees a placeholder of the form `<|secret|>NAME<|secret|>`. The real value — held internally as a Pydantic `SecretStr` — is substituted into tool inputs only at execution time, right before the action reaches the operating system. As defense-in-depth, any literal secret value that slips into the conversation is redacted back to its placeholder before it leaves the trusted boundary (LLM prompt, history, reporter, logs, and cache).

## Quick Start

```python
from askui import ComputerAgent, Secret

with ComputerAgent(secrets=[Secret(name="password", value="hunter2")]) as agent:
    agent.act("Log in as 'admin' using the password")
```

The agent never receives `hunter2`. Instead it is told that a placeholder named `password` is available, emits `<|secret|>password<|secret|>` as the text to type, and the SDK substitutes the real value at the OS boundary.

> **Do not hardcode real secrets in source.** The example above uses a literal value for brevity only. In real usage, read the value from your environment (see [Providing Values](#providing-values)).

## How It Works

```
LLM sees:          <|secret|>password<|secret|>          (placeholder only)
                              │
                              ▼  substitution at execution time
Operating system receives:   hunter2                     (real value)
```

1. **Advertise** — Registered placeholders are listed in an `<AVAILABLE_SECRETS>` section appended to the system prompt, so the model knows which placeholders it may use and is instructed never to guess or write out the real value.
2. **Substitute** — Immediately before a tool runs, every `<|secret|>NAME<|secret|>` placeholder in the tool input is replaced with the real value. This works for the built-in `type` tool, custom tools, and MCP tools alike.
3. **Redact** — Any literal secret value found in messages added to the history, tool outputs, or error messages is replaced with its placeholder before it reaches the model, reporter, logs, or cache.

## Providing Values

A `Secret` accepts a plain `str` or a Pydantic `SecretStr` as its `value`; either way it is stored as a `SecretStr` so it stays masked in reprs, logs, and `model_dump()`. Read real values from the environment rather than embedding them in code:

```python
import os

from askui import ComputerAgent, Secret

secrets = [
    Secret(
        name="password",
        value=os.environ["APP_PASSWORD"],
        description="the application login password",
    ),
]

with ComputerAgent(secrets=secrets) as agent:
    agent.act("Log in as 'admin' using the password")
```

The optional `description` is a human-readable hint shown to the model (e.g. so it knows which placeholder is the password). The description **is** sent to the model, so it must not contain the secret itself.

## Agent-Level vs. Per-Call Secrets

Secrets passed to the constructor apply to every `act()`, `get()`, and deterministic `type()` call on the agent. You can also supply secrets for a single `act()` call; per-call secrets override agent-level ones with the same name:

```python
with ComputerAgent() as agent:
    agent.act(
        "Enter the one-time PIN into the verification field",
        secrets=[Secret(name="pin", value=os.environ["APP_OTP"])],
    )
```

## Use in Deterministic `type()`

Placeholders also resolve in the deterministic `type()` method, so a secret can be entered without going through the model at all:

```python
with ComputerAgent(secrets=secrets) as agent:
    agent.click("Password field")
    agent.type("<|secret|>password<|secret|>")
```

## Supported Agents

`secrets=` is available on all agent types: `ComputerAgent`, `AndroidAgent`, `WebAgent`, `WebTestingAgent`, and `MultiDeviceAgent`. On `MultiDeviceAgent`, secrets are propagated to both the composed computer and Android agents. Secrets are excluded from telemetry.

## Limitations

Redaction is a best-effort safety net; placeholder usage is the primary, reliable path. Be aware of the following:

- **Screenshots are not protected.** A secret typed into a *visible* field can still appear in subsequent screenshots sent to the model (and in `get()`/OCR over such a screen). On-screen secrets cannot currently be hidden.
- **Exact-substring redaction only.** Transformed forms (base64-, URL-encoded, or partial) of a secret value are not caught.
- **Very short values (< 4 characters) are not redacted**, to avoid over-redacting unrelated text. The placeholder path still works for short values.

## API

### `Secret`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Identifier used in the placeholder (`<\|secret\|>NAME<\|secret\|>`). Must be non-empty. |
| `value` | `str \| SecretStr` | The sensitive value. Stored as a `SecretStr`. Read the real value via `secret.value.get_secret_value()` only where you actually need it. |
| `description` | `str`, optional | Human-readable hint shown to the model. Must not contain the secret. Defaults to `""`. |

`Secret.placeholder` returns the placeholder string the LLM uses to reference the secret.

### `SecretVault`

`SecretVault` holds registered secrets and performs substitution and redaction. You usually do not need to construct it directly — pass `Secret` instances via `secrets=` and the agent builds the vault for you. Both `Secret` and `SecretVault` are exported from the top-level `askui` package.
