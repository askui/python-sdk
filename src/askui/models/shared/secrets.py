"""Secret scope — values the agent may use but the LLM must never see.

Users register secrets by name. The LLM only ever sees placeholders of the form
``<|secret|>NAME<|secret|>``. Real values are substituted only at tool-execution time
(deepest point before the OS call), and a redaction safety-net scrubs any literal
secret value from anything that leaves the trusted boundary (LLM prompt/history,
reporter, logs, cache files).

Two operations:
- ``substitute`` (placeholder -> value): applied to a copy of tool input immediately
  before a tool runs. Conversation history keeps the placeholder.
- ``redact`` / ``redact_message`` (value -> placeholder): defense-in-depth applied
  before content reaches the LLM, reporter, logs or cache.

Limitations (best-effort):
- **Screenshots are NOT protected.** A secret typed into a visible field can appear in
  subsequent screenshots sent to the model (and in `get()`/OCR over such a screen).
  On-screen secrets cannot currently be hidden — only text fed to the model/reporter/
  logs/cache is scrubbed.
- Redaction is exact-substring only — transformed forms (base64/url-encoded/partial)
  are not caught.
- Very short values (< 4 chars) are not redacted to avoid over-redacting unrelated text
  (placeholder usage remains the primary path).
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from askui.models.shared.agent_message_param import (
    ContentBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

logger = logging.getLogger(__name__)

_PLACEHOLDER_PREFIX = "<|secret|>"
_PLACEHOLDER_SUFFIX = "<|secret|>"
# Match `<|secret|>NAME<|secret|>`; non-greedy so adjacent placeholders are matched
# individually. The delimiters make any (non-empty) name unambiguous, so no charset
# restriction on names is required.
_PLACEHOLDER_PATTERN = re.compile(
    re.escape(_PLACEHOLDER_PREFIX) + r"(.+?)" + re.escape(_PLACEHOLDER_SUFFIX)
)
# Values shorter than this are not redacted (the placeholder path still works); short
# values risk over-redacting unrelated occurrences in normal text.
_MIN_REDACTION_LENGTH = 4


def _placeholder_for(name: str) -> str:
    """Build the placeholder string for a secret ``name``."""
    return f"{_PLACEHOLDER_PREFIX}{name}{_PLACEHOLDER_SUFFIX}"


class Secret(BaseModel):
    """A named secret value the agent may use but the LLM must never see.

    The agent references the secret in tool calls via its placeholder
    (`<|secret|>NAME<|secret|>`); the real value is substituted at execution time.

    Args:
        name (str): Identifier used in the placeholder `<|secret|>NAME<|secret|>`. Must
            be non-empty.
        value (str | SecretStr): The sensitive value. Accepts a plain `str` (wrapped
            automatically) or a `SecretStr`; stored as a `SecretStr` so it is masked in
            reprs, logs and `model_dump()`/`model_dump_json()`. Substituted into tool
            calls at execution time; never sent to the model. Read the real value via
            `secret.value.get_secret_value()` only where you actually need it.
        description (str, optional): Human-readable hint shown to the model so it knows
            what the placeholder is for (e.g. `"the user's login password"`). The
            description itself IS sent to the model, so it must not contain the secret.
            Defaults to `""`.

    Note:
        The real value is kept out of the model prompt, reporter, logs and cache, but a
        secret typed into a **visible** field can still appear in subsequent screenshots
        sent to the model. On-screen secrets cannot currently be hidden.

    Example:
        ```python
        from askui import ComputerAgent, Secret

        with ComputerAgent(
            secrets=[Secret(name="password", value="hunter2")]
        ) as agent:
            agent.act("Log in as admin using the password")
        ```
    """

    name: str = Field(min_length=1)
    value: SecretStr
    description: str = ""

    @property
    def placeholder(self) -> str:
        """The placeholder string the LLM uses to reference this secret."""
        return _placeholder_for(self.name)


class SecretVault:
    """Holds registered secrets and performs substitution and redaction.

    Real secret values live only here. See the module docstring for the trust model.

    Args:
        secrets (list[Secret] | None, optional): Secrets to register. Later entries win
            on name collision. Defaults to `None` (empty vault).
    """

    def __init__(self, secrets: list[Secret] | None = None) -> None:
        self._secrets: dict[str, Secret] = {
            secret.name: secret for secret in (secrets or [])
        }

    def __bool__(self) -> bool:
        return bool(self._secrets)

    @property
    def names(self) -> list[str]:
        """Names of all registered secrets."""
        return list(self._secrets.keys())

    @property
    def secrets(self) -> list[Secret]:
        """All registered secrets."""
        return list(self._secrets.values())

    def merge(self, other: "SecretVault") -> "SecretVault":
        """Return a new vault combining this vault with `other`.

        Secrets in `other` take precedence on name collision.
        """
        return SecretVault(self.secrets + other.secrets)

    def substitute(self, obj: Any) -> Any:
        """Recursively replace `<|secret|>NAME<|secret|>` placeholders with real values.

        Returns a new object; the input is not mutated. Unknown placeholder names are
        left intact.
        """
        if not self._secrets:
            return obj
        if isinstance(obj, str):
            return self._substitute_str(obj)
        if isinstance(obj, dict):
            return {key: self.substitute(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.substitute(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.substitute(item) for item in obj)
        return obj

    def _substitute_str(self, text: str) -> str:
        def _replace(match: "re.Match[str]") -> str:
            # Tolerate stray whitespace the model may add inside the delimiters
            # (e.g. ``<|secret|> password <|secret|>``).
            name = match.group(1).strip()
            secret = self._secrets.get(name)
            if secret is None:
                logger.debug("Unknown secret placeholder '%s' left unresolved", name)
                return match.group(0)
            return secret.value.get_secret_value()

        return _PLACEHOLDER_PATTERN.sub(_replace, text)

    def redact(self, obj: Any) -> Any:
        """Recursively replace literal secret values with their placeholders.

        Emits a warning (naming the secret, never its value) whenever a literal value is
        found and replaced. Returns a new object; the input is not mutated.
        """
        if not self._secrets:
            return obj
        if isinstance(obj, str):
            return self._redact_str(obj)
        if isinstance(obj, dict):
            return {key: self.redact(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.redact(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact(item) for item in obj)
        return obj

    def _redact_str(self, text: str) -> str:
        result = text
        for secret in self._secrets.values():
            value = secret.value.get_secret_value()
            if len(value) < _MIN_REDACTION_LENGTH:
                continue
            if value in result:
                result = result.replace(value, secret.placeholder)
                logger.warning(
                    "Redacted secret '%s' from content before it left the trusted "
                    "boundary. Reference secrets via their placeholder '%s' instead of "
                    "embedding the value in goals or tool outputs.",
                    secret.name,
                    secret.placeholder,
                )
        return result

    def redact_message(self, message: MessageParam) -> MessageParam:
        """Return a copy of `message` with all text-bearing fields redacted."""
        if not self._secrets:
            return message
        redacted = message.model_copy(deep=True)
        redacted.content = self.redact_content(redacted.content)
        return redacted

    def redact_content(
        self, content: "str | list[ContentBlockParam] | list[Any]"
    ) -> Any:
        """Redact a message/tool-result content (str or list of content blocks).

        Used to scrub tool outputs (`ToolResultBlockParam.content`) before they are
        fed back to the model or recorded, in addition to `redact_message`.
        """
        if not self._secrets:
            return content
        if isinstance(content, str):
            return self._redact_str(content)
        return [self._redact_block(block) for block in content]

    def _redact_block(self, block: Any) -> Any:
        if isinstance(block, TextBlockParam):
            block.text = self._redact_str(block.text)
        elif isinstance(block, ToolResultBlockParam):
            block.content = self.redact_content(block.content)
        elif isinstance(block, ToolUseBlockParam):
            block.input = self.redact(block.input)
        return block

    def system_prompt_section(self) -> str:
        """Build the `<AVAILABLE_SECRETS>` system-prompt block (``""`` if empty)."""
        if not self._secrets:
            return ""
        lines = [
            f"- {secret.placeholder}"
            + (f" — {secret.description}" if secret.description else "")
            for secret in self._secrets.values()
        ]
        listing = "\n".join(lines)
        example = next(iter(self._secrets.values())).placeholder
        return (
            "<AVAILABLE_SECRETS>\n"
            "The following secret placeholders are available. When you need to enter a "
            "sensitive value (e.g. a password), use the EXACT placeholder string shown "
            "below as the value (for example, as the text to type). The real value is "
            "substituted securely at execution time and is hidden from you. "
            "NEVER guess, invent, or ask for the actual value, and never write it out."
            "\n\n"
            f"{listing}\n\n"
            "Example — to enter the first secret with a typing tool, pass the text "
            f"exactly as: {example}\n"
            "</AVAILABLE_SECRETS>"
        )
