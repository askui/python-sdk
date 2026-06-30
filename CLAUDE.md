# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AskUI Vision Agent** is a Python desktop and mobile automation framework that enables AI agents to control computers (Windows, macOS, Linux), mobile devices (Android, iOS), and HMI systems. It supports both programmatic UI automation (RPA-like single-step commands) and agentic intent-based instructions using vision/computer vision models.

**Tech Stack:** Python 3.10+, Pydantic 2, Anthropic SDK, OpenTelemetry, Model Context Protocol (MCP), PDM

## Common Commands

### Development Setup
```bash
# Install dependencies
pdm install
```

### Testing
```bash
# Run all tests (parallel execution)
pdm run test

# Run specific test suites
pdm run test:unit          # Unit tests only
pdm run test:integration   # Integration tests only
pdm run test:e2e          # End-to-end tests only

# Run tests with coverage
pdm run test:cov          # All tests with coverage report
pdm run test:cov:view     # View coverage report in browser
```

### Code Quality
```bash
# Quick QA: type check, format, and fix linting issues (run before commits)
pdm run qa:fix

# Individual commands
pdm run typecheck:all     # Type checking with mypy
pdm run format            # Format code with ruff
pdm run lint              # Lint code with ruff
pdm run lint:fix          # Auto-fix linting issues
```

### Code Generation
```bash
# Regenerate gRPC client code from .proto files
pdm run grpc:gen

# Regenerate Pydantic models from JSON schemas
pdm run json:gen
```

## High-Level Architecture

### Core SDK Architecture

```
ComputerAgent (Main SDK Entry Point)
    ↓
Agent (Abstract base class for all agents)
    ├── ComputerAgent (Desktop automation)
    ├── AndroidAgent (Mobile Android automation)
    ├── WebVisionAgent (Web-specific automation)
    └── WebTestingAgent (Web testing framework)

    Uses:
    ├── ModelRouter → Model selection/composition
    ├── AgentToolbox → Tool & OS abstraction
    └── Locators → UI element identification
```

**Key Flow:**
1. User calls `agent.click("Submit button")` on `ComputerAgent`
2. `AgentBase.locate()` routes to appropriate model via `ModelRouter`
3. Model receives screenshot + locator → returns coordinates
4. `AgentToolbox.os.click()` → gRPC call to Agent OS
5. Agent OS performs actual mouse click

### Chat API Architecture

```
FastAPI Chat API (Experimental)
    ├── Assistants (AI agent configurations)
    ├── Threads (Conversation sessions)
    ├── Messages (Chat history)
    ├── Runs (Agent execution iterations)
    ├── Files (Attachments & resources)
    ├── MCP Configs (Tool providers)
    └── Workflows & Scheduled Jobs (Automation triggers)
```

**Key Flow:**
1. User → Chat UI (hub.askui.com) → Chat API (FastAPI)
2. Thread/Messages stored in SQLAlchemy database
3. Runs execute agent steps in a loop
4. Agent uses ModelRouter → Tools (MCP servers or direct) → AgentOS

### Model Router & Composition

The `ModelRouter` provides a flexible abstraction for AI model selection:

```python
# Single model for all tasks
model = "askui"

# Task-specific models (ActModel, GetModel, LocateModel)
model = {
    "act": "claude-sonnet-4-20250514",
    "get": "askui",
    "locate": "askui-combo"
}

# Custom registry
models = ModelRegistry()
models.register("my-model", custom_model_instance)
```

**Supported Model Providers:**
- **AskUI Models** (Primary - internally hosted)
- **Anthropic Claude** (Computer Use, Messages API)
- **Google Gemini** (via OpenRouter)
- **Hugging Face Spaces** (Community models)

### Agent OS Abstraction

`ComputerAgentOS` provides an abstraction layer for OS-level operations:

```
ComputerAgentOS (Abstract Interface)
    ├── MultiComputerTargetAgentOS (gRPC to AskUI Agent OS - primary)
    ├── PlaywrightAgentOs (Web browser automation)
```

### Locator System

Locators identify UI elements in multiple ways:

- **Text**: Match by text content (exact/similar/contains/regex)
- **Image**: Match by image file or base64
- **Prompt**: Natural language description
- **Coordinate**: Absolute (x, y) position
- **Relatable**: Positional relationships (right_of, below, etc.)

Serialization differs by model type (VLM vs. traditional).

### Tool System (MCP)

Tools follow the Model Context Protocol (MCP) for extensibility:

```
Tools (MCP Servers)
    ├── Computer: screenshot, click, type, mouse, clipboard
    ├── Android: device control via ADB
    ├── Testing: scenario & feature management
    └── Utility: file ops, data extraction
```

Tools are auto-discovered and can be dynamically loaded via MCP configurations.

## Key Code Locations

### Core SDK
- `src/askui/agent.py` - Main `ComputerAgent` class (user-facing API)
- `src/askui/agent_base.py` - Abstract `Agent` (base) with shared agent logic
- `src/askui/android_agent.py` - Android-specific agent
- `src/askui/web_agent.py` - Web-specific agent

### Models & AI
- `src/askui/models/` - AI model providers & router factory
- `src/askui/models/shared/` - Shared abstractions (`Agent`, `Tool`, `MessagesApi`)
- `src/askui/models/{provider}/` - Provider implementations
- `src/askui/prompts/` - System prompts for different models

### Tools & OS
- `src/askui/tools/agent_os.py` - Abstract `ComputerAgentOS` interface
- `src/askui/tools/askui/` - gRPC client for AskUI Agent OS
- `src/askui/tools/android/` - Android-specific tools
- `src/askui/tools/playwright/` - Web automation tools
- `src/askui/tools/mcp/` - MCP client/server implementations
- `src/askui/tools/testing/` - Test scenario tools

### Locators
- `src/askui/locators/` - UI element selectors
- `src/askui/locators/serializers.py` - Locator serialization for models

### Chat API
- `src/askui/chat/` - FastAPI-based Chat API
- `src/askui/chat/api/` - REST API routes
- `src/askui/chat/migrations/` - Alembic migrations & ORM models

### Utilities
- `src/askui/utils/` - Image processing, API utilities, caching, annotations
- `src/askui/reporting.py` - Reporting & logging
- `src/askui/retry.py` - Retry logic with exponential backoff
- `src/askui/telemetry/` - OpenTelemetry tracing & analytics

## Code Style & Conventions

### General Python Style
- **Private members**: Use `_` prefix for all private variables, functions, methods, etc. Mark everything private that doesn't need external access.
- **Type hints**: Required everywhere. Use built-in types (`list`, `dict`, `str | None`) instead of `typing` module types (`List`, `Dict`, `Optional`).
- **Overrides**: Use `@override` decorator from `typing_extensions` for all overridden methods.
- **Exceptions**: Never pass literals to exceptions. Assign to variables first:
  ```python
  # Good
  error_msg = f"Thread {thread_id} not found"
  raise FileNotFoundError(error_msg)

  # Bad
  raise FileNotFoundError(f"Thread {thread_id} not found")
  ```
- **File operations**: Always specify `encoding="utf-8"` for file read/write operations.
- **Init files**: Create `__init__.py` in each folder.

### FastAPI Specific
- Use response type in function signature instead of `response_model` in route annotation.
- Dependencies without defaults should come before arguments with defaults.

### Testing
- Use `pytest-mock` for mocking wherever possible.
- Test files in `tests/` follow structure: `test_*.py` with `Test*` classes and `test_*` functions.
- Timeout: 60 seconds per test (configured in `pyproject.toml`).

### Git Conventions
- **Never** use `git add .` - explicitly add files related to the task.
- Use conventional commits format: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`.
- **Before committing**, always run: `pdm run qa:fix` (or individually: `typecheck:all`, `format`, `lint:fix`).

### Docstrings
- All public functions, classes, and constants require docstrings.
- Document constructor args in class docstring, omit `__init__` docstring.
- Use backticks for code references (variables, types, functions).
- Function references: `click()`, Class references: `ComputerAgent`, Method references: `VisionAgent.click()`
- Include sections: `Args`, `Returns`, `Raises`, `Example`, `Notes`, `See Also` as needed.
- Document parameter types in parentheses, add `, optional` for defaults.

### Documentation (docs/)
When writing or updating documentation in `docs/`:
- **Never show setting environment variables in Python code** (e.g., `os.environ["ASKUI_WORKSPACE_ID"] = "..."`). This is bad practice. Always instruct users to set environment variables via their shell or system settings.
- Keep examples concise and focused on the feature being documented.
- Test all code examples before including them.
- Use `ComputerAgent` (not `VisionAgent`) in examples.

## Important Patterns

### Composition over Inheritance
- `AgentToolbox` wraps `ComputerAgentOS` implementations
- `ModelRouter` composes multiple model providers
- `CompositeReporter` aggregates multiple reporters

### Factory Pattern
- `ModelRouter.initialize_default_model_registry()` creates model registry
- Model providers use factory functions for lazy-loading

### Strategy Pattern
- Truncation strategies for message history
- Different locator serializers for model types
- Retry strategies with exponential backoff

### Adapter Pattern
- `ComputerAgentOS` abstraction bridges OS implementations (gRPC, Playwright, ADB)
- `ModelFacade` adapts models to `ActModel`/`GetModel`/`LocateModel` interfaces

### Dependency Injection
- Constructor-based DI throughout
- FastAPI dependencies for Chat API routes
- `@auto_inject_agent_os` decorator for tools

### Template Method Pattern
- `Agent._step()` orchestrates tool-calling loop
- `Agent` provides common structure for all agents

## Database & Observability

### Alembic Migrations
- Schema versioning in `src/askui/chat/migrations/`
- ORM models in `migrations/shared/{entity}/models.py`
- Auto-migration on startup (configurable)
- SQLAlchemy with async support

### Telemetry
- OpenTelemetry integration (FastAPI, HTTPX, SQLAlchemy)
- Structured logging with structlog
- Correlation IDs for request tracing
- Prometheus metrics via FastAPI instrumentator
- Segment Analytics for usage tracking

## Extending the Framework

### Adding Custom Models
1. Inherit from `ActModel`, `GetModel`, or `LocateModel`
2. Implement message creation via `MessagesApi`
3. Register in `ModelRegistry`
4. Use appropriate locator serializer

### Adding Custom Tools
1. Implement `Tool` protocol in `models/shared/tools.py`
2. Register in appropriate MCP server (`api/mcp_servers/{type}.py`)
3. Use `@auto_inject_agent_os` for ComputerAgentOS dependency
4. Follow Pydantic schema validation

### Adding New Agent Types
1. Inherit from `Agent`
2. Implement required abstract methods
3. Provide appropriate `ComputerAgentOS` implementation
4. Register in agent factory if needed

## Performance & Caching

- Screenshot caching for multi-step operations
- Token counting before API calls
- Cached trajectory execution (replay previous interactions)
- Image downsampling & compression
- Lazy model initialization (`@functools.cache`)

## Error Handling

Custom exceptions:
- `ElementNotFoundError` - UI element not found
- `WaitUntilError` - Timeout waiting for condition
- `MaxTokensExceededError` - Token limit exceeded
- `ModelRefusalError` - Model refused to execute

Retry logic with configurable strategies via `src/askui/retry.py`.

## Documentation References

Additional documentation in `docs/`:
- `chat.md` - Chat API usage
- `direct-tool-use.md` - Direct tool usage
- `extracting-data.md` - Data extraction
- `mcp.md` - MCP servers
- `observability.md` - Logging and reporting
- `telemetry.md` - Telemetry data
- `using-models.md` - Model usage and custom models

Official docs: https://docs.askui.com
Discord: https://discord.gg/Gu35zMGxbx


## Conding Standards
### Anti-Patterns and Bad Examples
1) Setting Env Variables In-Code
```python
os.environ.set("ANTHROPIC_API_KEY")
````
=> we never want to set env variables by the process itself in-code. We expect them to be set in the environment directly hence explicitly setting is not necessary, or if still necessary, please pass them directly to the Client/... that requires the value.

2) Don't Use Lazy Loading
=> we want to have imports at the top of files. Use lazy-loading only in very rare edge-cases, e.g. if you have to check with a try-except if a package is available (in this case it should be an optional dependency)

3) Client Config
All lazy initialized clients should be configurable in the init method

4) Be consisted with the variable namings within one classes (and its subclasses)!
For example, if a parameter is named client, then the member variable that is passed to it should also be named client
