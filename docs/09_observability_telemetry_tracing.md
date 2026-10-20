# Observability, Telemetry, and Tracing

## OpenTelemetry Tracing

AskUI supports exporting traces via [OpenTelemetry](https://opentelemetry.io/) (OTLP/HTTP) for integration with observability backends like Grafana, Jaeger, or Datadog.

### Setup

To use OpenTelemetry Tracing, you first need to install the optional tracing dependencies
```
pip install askui[otel]
```


### Configuration via Environment Variables

Set the following environment variables to configure tracing. All variables use the `ASKUI__OTEL_` prefix.

| Environment Variable | Description | Default |
|---|---|---|
| `ASKUI__OTEL_ENABLED` | Enable or disable OpenTelemetry tracing | `False` |
| `ASKUI__OTEL_USER` | user for OTLP authentication (required when enabled) | — |
| `ASKUI__OTEL_SECRET` | secret for OTLP authentication (required when enabled) | — |
| `ASKUI__OTEL_ENDPOINT` | OTLP HTTP endpoint URL | — |
| `ASKUI__OTEL_SERVICE_NAME` | Service name reported in traces | `askui-python-sdk` |
| `ASKUI__OTEL_SERVICE_VERSION` | Service version reported in traces | Current package version |

#### Linux & MacOS
```bash
export ASKUI__OTEL_ENABLED=True
export ASKUI__OTEL_USER="your-user"
export ASKUI__OTEL_SECRET="your-secret"
export ASKUI__OTEL_ENDPOINT="https://your-otlp-endpoint/v1/traces"
```

#### Windows PowerShell
```powershell
$env:ASKUI__OTEL_ENABLED="True"
$env:ASKUI__OTEL_USER="your-user"
$env:ASKUI__OTEL_SECRET="your-secret"
$env:ASKUI__OTEL_ENDPOINT="https://your-otlp-endpoint/v1/traces"
```

### Usage

Once environment variables are set, pass `OtelSettings` to the `act()` method. Settings are automatically read from the environment:

```python
from askui import ComputerAgent
from askui.telemetry.otel import OtelSettings

with ComputerAgent() as agent:
    agent.act(
        goal="Open Chrome and navigate to askui.com",
        tracing_settings=OtelSettings(enabled=True),
    )
```

You can also override individual settings in code:

```python
from askui.telemetry.otel import OtelSettings

settings = OtelSettings(
    enabled=True,
    service_name="my-custom-service"
)
```
