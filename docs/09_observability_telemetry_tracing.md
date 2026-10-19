# Observability, Telemetry, and Tracing

Telemetry collects usage data to help improve AskUI Vision Agent. This data is used to detect bugs, understand usage patterns, and improve the user experience.

### What Data Is Collected

**Disabled by default.** When enabled, telemetry records:

- **Package Version**: `askui` package version in use
- **Environment Information**:
  - Operating system and version
  - System architecture (x86_64, arm64, etc.)
  - Python version
  - Device ID (hashed for privacy)
- **Session Information**:
  - Session ID (random UUID per run)
  - Workspace ID (if authenticated)
  - User ID (if authenticated)
- **Method Calls**:
  - Functions called (e.g., `click()`, `act()`, `get()`)
  - Non-sensitive parameters (e.g., click coordinates)
  - Execution duration
- **Exceptions**:
  - Exception types and messages
  - Stack traces (sanitized)

### What Data Is NOT Collected

- Screenshot content or image data
- Passwords or credentials
- User input text (what you type)
- Query text (what you ask the agent)
- Screen content or extracted data
- File paths or filenames
- Network traffic or API keys

### Privacy Considerations

- **Device ID Hashing**: Your machine's device ID is hashed with an AskUI-specific salt to prevent cross-application tracking
- **No PII**: No personally identifiable information is collected without explicit authentication
- **Workspace Linkage**: If `ASKUI_WORKSPACE_ID` and `ASKUI_TOKEN` are set, telemetry is linked to your workspace for better support
- **Open Source**: Telemetry code is open source and auditable in the repository

### Enabling Telemetry

Telemetry is off unless you opt in by setting an environment variable:

#### Linux & MacOS
```bash
export ASKUI__VA__TELEMETRY__ENABLED=True
```

#### Windows PowerShell
```powershell
$env:ASKUI__VA__TELEMETRY__ENABLED="True"
```

#### Windows Command Prompt
```cmd
set ASKUI__VA__TELEMETRY__ENABLED=True
```

When disabled, no usage data is collected or transmitted.

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
