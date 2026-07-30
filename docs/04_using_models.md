# Using Models

AskUI supports using various different models

## Default Setup

The code will use the default act and get models. We will update them occassionally to always give you the best balance agent capabilities and cost-efficiency.

```python
from askui import ComputerAgent

with ComputerAgent() as agent:
    agent.act("Open the calculator")
    result = agent.get("What is shown on the display?")
```

---

## Configuring Model IDs

If you want to use another model, you select one of the available ones and set as an environment variable (**currently only supported for vlm_provider!**):
```
VLM_PROVIDER_MODEL_ID=claude-opus-5
```

Alternatively, you can also set it through overriding the model_id in the provider:

```python
from askui import AgentSettings, ComputerAgent
from askui.model_providers import AskUIVlmProvider, AskUIImageQAProvider

with ComputerAgent(settings=AgentSettings(
    vlm_provider=AskUIVlmProvider(model_id="claude-opus-5"),
    image_qa_provider=AskUIImageQAProvider(model_id="gemini-2.5-pro"),
)) as agent:
    agent.act("Complete the checkout process")
```

The following models are available with your AskUI credentials through the AskUI API:

**VLM Provider** (for `act()`):
- `claude-haiku-4-5-20251001` (most cost efficient)
- - `gemini-3.5-flash` (fastest)
- `claude-sonnet-5`(default)
- `claude-opus-5` (most capable)


**Image Q&A Provider** (for `get()`): Gemini models via AskUI's Gemini proxy
- `gemini-2.5-flash` (default)
- `gemini-2.5-pro`

You can find more details on the capabilities and limitations of the Anthropic Claude models [here](https://platform.claude.com/docs/en/about-claude/models/overview), and the Google Gemini models [here](https://modelcards.withgoogle.com/model-cards)
