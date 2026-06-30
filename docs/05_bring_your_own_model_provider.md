# Using Models and BYOMP (Bring Your Own Model Provider)

Our SDK allows you to **use your own model cloud** or models from a 3rd party provider to work with models that are not available through the AskUI API.


Our SDK follows a layered architecture: Provider → Model → MessagesAPI → Client

- **Provider**: User-facing configuration layer (`AskUIVlmProvider`, `AnthropicVlmProvider`, etc.)
- **Model**: Internal implementation handling the agent loop (e.g., `AnthropicActModel`)
- **MessagesAPI**: Converts internal message format to provider-specific format
- **Client**: Underlying HTTP client (`Anthropic`, `OpenAI`, etc.)

**Users typically configure only the Provider**. For advanced use cases and full customization, you can also inject a pre-configured **Client**.

Our SDK supports the following three provider slots:

| Slot | Used by | Default |
|------|---------|---------|
| `vlm_provider` | `act()` — multimodal input + tool-calling | `AskUIVlmProvider` |
| `image_qa_provider` | `get()` — Q&A and structured extraction | `AskUIImageQAProvider` |
| `detection_provider` | `locate()`, `click()`, etc. — element coordinates | `AskUIDetectionProvider` |

**⚠️ Important:** The necessary credentials for each provider are read from environment variables and validated lazily. Please make sure to set the necessary environment variables for the provider you are using!

---

## AskUI Model Providers

Per default, Agents use the following providers to access models from the AskUI API:
- `AskuiVlmProvider` for `act()`
- `AskUIImageQAProvider` for `get()`
- `AskUIDetectionProvider` for `locate()`


```python
from askui import ComputerAgent

with ComputerAgent() as agent:
    agent.act("Open the calculator")
    result = agent.get("What is shown on the display?")
    pos = agent.locate("7")
```

Environment variables:
- `ASKUI_WORKSPACE_ID` (required)
- `ASKUI_TOKEN` (required)

## Anthropic Model Providers

If you want to use Anthropic models directly from the Anthropic API with your Anthropic API key, you can use:
- `AnthropicVlmProvider` for `act()`
- `AnthropicImageQAProvider` for `get()`

```python
import os
from askui import AgentSettings, ComputerAgent
from askui.model_providers import AnthropicVlmProvider

with ComputerAgent(settings=AgentSettings(
    vlm_provider=AnthropicVlmProvider(
        model_id="claude-opus-4-6",
    ),
)) as agent:
    agent.act("Navigate to settings")
    agent.get("What is shown on the display")
```

Environment variables:
- `ANTHROPIC_API_KEY` (optional)
- `ANTHROPIC_AUTH_TOKEN` (optional)
- `ANTHROPIC_BASE_URL` (optional, default=`https://api.anthropic.com`)

_Note: either `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` must be set_

## Google Model Provider

If you want to use Google models directly from the Google API with your Google API key, you can use:
- `GoogleImageQAProvider` for `get()`

```python
from askui import AgentSettings, ComputerAgent
from askui.model_providers import GoogleImageQAProvider

with ComputerAgent(settings=AgentSettings(
    image_qa_provider=GoogleImageQAProvider(
        model_id="gemini-2.5-pro",
    ),
)) as agent:
    agent.get("What is shown on the display")
```

Environment variables:
- `GOOGLE_API_KEY` (required)

or, if you want to use the Vertex AI API, please set:
- `GOOGLE_GENAI_USE_VERTEXAI=true` (required)
- `GOOGLE_CLOUD_PROJECT` (required)
- `GOOGLE_CLOUD_LOCATION` (required)

## Custom Model Providers

For customization and to access to use your own model cloud, you can also implement your own model provider class. The library defines three base classes, that you can overwrite for that.


| Base Class | Method to overwrite | Used by |
|-----------|--------|---------|
| `VlmProvider` | `create_message(...)` | `act()` |
| `ImageQAProvider` | `query(...)` | `get()` |
| `DetectionProvider` | `detect(...)` | `locate()` |

### Example
```python
from typing import Type
from typing_extensions import override
from askui.model_providers import ImageQAProvider
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source


class MyImageQAProvider(ImageQAProvider):
    @property
    def model_id(self) -> str:
        return "my-model-v1"

    @override
    def query(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        return "my answer"  # call your API here
```


### Execution Cost Tracking

The built-in VLM providers include default pricing for supported models. You can override the pricing on any provider by passing `input_cost_per_million_tokens` and `output_cost_per_million_tokens`:

```python
from askui import AgentSettings, ComputerAgent
from askui.model_providers import AnthropicVlmProvider
from askui.reporting import SimpleHtmlReporter

with ComputerAgent(
    reporters=[SimpleHtmlReporter()],
    settings=AgentSettings(
        vlm_provider=AnthropicVlmProvider(
            model_id="claude-sonnet-4-6",
            input_cost_per_million_tokens=3.0,
            output_cost_per_million_tokens=15.0,
        ),
    ),
) as agent:
    agent.act("Open settings")
```

If you implement a fully custom `VlmProvider`, override the `pricing` property to enable cost tracking:

```python
from askui.model_providers import VlmProvider, ModelPricing

class MyVlmProvider(VlmProvider):
    @property
    def pricing(self) -> ModelPricing | None:
        return ModelPricing(
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=5.0,
        )

    # ... rest of implementation
```

---

## Image Scaling

Before a screenshot is sent to a model, it is preprocessed by an **image scaler**. The scaler resizes the image to match the model's optimal input resolution, which affects both token cost and coordinate precision.

All scalers inherit from `ImageScaler`:

| Class | Behaviour | Used by |
|-------|-----------|---------|
| `PatchOptimizedImageScaler` | Finds the largest aspect-preserving size within a patch-based token budget (`max_edge`, `max_tokens`, `patch_size`) | `AskUIVlmProvider`, `AnthropicVlmProvider`, `OpenAIVlmProvider` |
| `ContainedImageScaler` | Fits within `max_width` × `max_height` bounds | Default in `VlmProvider` base class |

### Configuring the Maximum Image Edge

All built-in providers accept an `image_edge_max` parameter that controls the maximum pixel dimension of screenshots sent to the model. You can also set it via the `ASKUI_VLM_MAX_IMAGE_EDGE` environment variable:

```
ASKUI_VLM_MAX_IMAGE_EDGE=1568
```

Or pass it directly:

```python
from askui import AgentSettings, ComputerAgent
from askui.model_providers import AnthropicVlmProvider

with ComputerAgent(settings=AgentSettings(
    vlm_provider=AnthropicVlmProvider(image_edge_max=1568),
)) as agent:
    agent.act("Open settings")
```

### Using a Custom Image Scaler

To fully replace the scaling strategy, pass an `image_scaler` instance. When provided, `image_edge_max` is ignored:

```python
from askui import AgentSettings, ComputerAgent
from askui.model_providers import (
    AnthropicVlmProvider,
    ContainedImageScaler,
)

with ComputerAgent(settings=AgentSettings(
    vlm_provider=AnthropicVlmProvider(
        image_scaler=ContainedImageScaler(max_width=1280, max_height=720),
    ),
)) as agent:
    agent.act("Open settings")
```

### Implementing a Custom Image Scaler

For fully custom scaling logic, subclass `ImageScaler`:

```python
from PIL import Image
from askui.model_providers import ImageScaler

class MyImageScaler(ImageScaler):
    def __call__(self, image: Image.Image) -> Image.Image:
        # Your custom scaling logic
        return image.resize((1024, 768), Image.Resampling.LANCZOS)
```

---

## Advanced: Injecting a Custom Client

For full control over HTTP settings (timeouts, proxies, retries), you can inject a pre-configured client:

```python
import os
from anthropic import Anthropic
from askui import AgentSettings, ComputerAgent
from askui.model_providers import AnthropicVlmProvider

client = Anthropic(
    timeout=60.0,
    max_retries=3,
)

with ComputerAgent(settings=AgentSettings(
    vlm_provider=AnthropicVlmProvider(client=client),
)) as agent:
    agent.act("Process the document")
```
