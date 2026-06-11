# Tools

Tools extend the capabilities of your agents, allowing them to interact with the operating system, perform complex operations, and integrate with external services.
They extend the agent’s capabilities beyond basic UI automation, allowing you to:
- Integrate external APIs and services
- Process data and perform calculations
- Manage files and system operations
- Handle complex business logic that goes beyond UI interactions
- Create reusable functionality across different test scenarios

Here we cover three ways of augmenting your agents capabilities with tools: 1) by using bre-built tools from our tool store 2) by adding tools from MCP servers, and 3) by implementing your own tools.

## Part 1: Tool Store

### Overview

The Tool Store provides pre-built, ready-to-use tools that extend your agents' capabilities beyond the default computer control operations. These tools are organized by category and can be easily imported and integrated into your automation workflows.

### How to Use Pre-Built Tools

Import tools from `askui.tools.store` and pass them to your agent in one of two ways:

**Option 1: Pass tools to `agent.act()`:**

```python
from askui import ComputerAgent
from askui.tools.store.computer import ComputerSaveScreenshotTool
from askui.tools.store.universal import PrintToConsoleTool

with ComputerAgent() as agent:
    agent.act(
        "Take a screenshot and save it as demo/demo.png, then print a status message",
        tools=[
            ComputerSaveScreenshotTool(base_dir="./screenshots"),
            PrintToConsoleTool()
        ]
    )
```

**Option 2: Pass tools to the agent constructor:**

```python
from askui import ComputerAgent
from askui.tools.store.computer import ComputerSaveScreenshotTool
from askui.tools.store.universal import PrintToConsoleTool

with ComputerAgent(act_tools=[
    ComputerSaveScreenshotTool(base_dir="./screenshots"),
    PrintToConsoleTool()
]) as agent:
    agent.act("Take a screenshot and save it as demo/demo.png, then print a status message")
```

### Tool Categories

Tools are organized into three main categories based on their dependencies and use cases:

#### Universal Tools (`universal/`)

Work with any agent type, no special dependencies required.

**Examples:**
- `PrintToConsoleTool()` - Print messages to console output
- Data processing and formatting tools
- General utility functions

**Import from:** `askui.tools.store.universal`

#### Computer Tools (`computer/`)

Require `ComputerAgentOS` and work with `ComputerAgent` for desktop automation.

**Examples:**
- `ComputerSaveScreenshotTool(base_dir)` - Save screenshots to disk
- Window management
- Device Automation

**Import from:** `askui.tools.store.computer`

**Requirements:** Only available with `ComputerAgent`

#### Android Tools (`android/`)

Require `AndroidAgentOs` and work with `AndroidAgent` for mobile automation.

**Examples:**
- Device information retrieval
- App management operations
- Mobile-specific interactions
- ADB command execution

**Import from:** `askui.tools.store.android`

**Requirements:** Only available with `AndroidAgent`

---

## Part 2: Extending with MCP

The Model Context Protocol (MCP) is a standardized way to provide context and tools to Large Language Models (LLMs) through a standardized interface. For more information, see the [MCP specification](https://modelcontextprotocol.io/docs/getting-started/intro).

AskUI supports the use of MCP tools in the library (`ComputerAgent.act()`, `AndroidAgent.act()`). Tool usage comprises:
1. Listing available tools from MCP servers
2. Passing tool definitions to the model
3. Calling tools when the model requests them

The implementation uses [`fastmcp`](https://gofastmcp.com/getting-started/welcome) as the underlying MCP client. Integrate MCP tools directly into your agents by creating an MCP client and passing it to the `ToolCollection`:

```python
from fastmcp import Client
from fastmcp.mcp_config import MCPConfig, RemoteMCPServer

from askui import ComputerAgent
from askui.models.shared.tools import ToolCollection
from askui.tools.mcp.config import StdioMCPServer

# Create MCP configuration
mcp_config = MCPConfig(
    mcpServers={
        # Make sure to use our patch of StdioMCPServer as we don't support the official one
        "test_stdio_server": StdioMCPServer(
            command="python", args=["-m", "askui.tools.mcp.servers.stdio"]
        ),
        "test_sse_server": RemoteMCPServer(url="http://127.0.0.1:8001/sse/"),
    }
)

# Create MCP client
mcp_client = Client(mcp_config)

# Create tool collection with MCP tools
tools = ToolCollection(mcp_client=mcp_client)

# Use with ComputerAgent
with ComputerAgent() as agent:
    agent.act(
        "Use the `test_stdio_server_test_stdio_tool`",
        tools=tools,
    )
```

**Important notes:**
- Tools are appended to the default tools of the agent, potentially overriding them
- Tool names are prefixed with the server name to avoid conflicts (e.g., `test_stdio_server_test_stdio_tool`)
- For different ways to construct `Client`s, see the [fastmcp documentation](https://gofastmcp.com/clients/client)

**Running the SSE Server Example:**

If you want to try the `test_sse_server`, start it before running your code:

```bash
python -m askui.tools.mcp.servers.sse
```

## Part 3: Building Custom Tools
For personalized functionalities you can add customly tailored tools to your agent. Each tool definition follows a consistent pattern with three essential components:

### 1. Tool Class Definition
```python
from askui.models.shared.tools import Tool

class MyCustomTool(Tool):
    """Brief description of what this tool does."""
```
### 2. Constructor (`__init__`)

The constructor defines the tool’s metadata and input requirements:
- name: Unique identifier (string) - must be unique across all tools
- description: Clear explanation (string) - helps the agent understand when to use this tool
- input_schema: JSON schema defining expected parameters

### 3. Execution Method (`__call__`)

Contains the actual business logic that runs when the tool is invoked.

Tools are flexible — they can return plain values, structured data, or even images.
A tool’s __call__ method may return:
- str
- numbers or other primitive values
- PIL.Image.Image — image output
- None
- a list or tuple containing any of the above

### Complete Example

Here’s a greeting tool that demonstrates all the key concepts:

```python
from askui.models.shared.tools import Tool
from datetime import datetime
from typing import Optional

class GreetingTool(Tool):
    """Creates personalized greeting messages with time-based customization."""

    def __init__(self):
        super().__init__(
            name="greeting_tool",
            description="Creates a personalized greeting message based on time of day and user preferences",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person to greet",
                        "minLength": 1
                    },
                    "time_of_day": {
                        "type": "string",
                        "description": "Time of day: morning, afternoon, or evening",
                        "enum": ["morning", "afternoon", "evening"]
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for the greeting (optional). Default is english.",
                        "enum": ["english", "spanish", "french"],
                        "default": "english"
                    }
                },
                "required": ["name", "time_of_day"]
            }
        )

    def __call__(self, name: str, time_of_day: str, language: Optional[str] = "english") -> str:
            if not name or not name.strip():
                raise ValueError("Name cannot be empty") # The error will be caught by the agent, it will try to fix the error and continue the execution. It's the agent auto-correction feature.

            if time_of_day not in ["morning", "afternoon", "evening"]:
                raise ValueError(f"Time of day must be 'morning', 'afternoon', or 'evening', got '{time_of_day}'") # The error will be caught by the agent, it will try to fix the error and continue the execution. It's the agent auto-correction feature.

            # Create greeting based on language
            greetings = {
                "english": {
                    "morning": "Good morning",
                    "afternoon": "Good afternoon",
                    "evening": "Good evening"
                },
                "spanish": {
                    "morning": "Buenos días",
                    "afternoon": "Buenas tardes",
                    "evening": "Buenas noches"
                },
                "french": {
                    "morning": "Bonjour",
                    "afternoon": "Bon après-midi",
                    "evening": "Bonsoir"
                }
            }

            base_greeting = greetings.get(language, greetings["english"])[time_of_day]
            return f"{base_greeting}, {name}! How are you today?"
```

### Error Handling in Tools

When a tool raises an exception, the agent distinguishes between **fixable** and **unfixable** errors:

- **Fixable errors** (regular exceptions): The error message is returned to the model, which can auto-correct and retry with different parameters. This is the default behavior for any `Exception` raised inside `__call__`.
- **Unfixable errors** (`AutomationError`): The error propagates immediately to the caller, terminating the agent's execution. Use this for errors where retrying cannot help (e.g., missing credentials, unreachable services, invalid environment state).

```python
from askui import AutomationError
from askui.models.shared.tools import Tool


class DatabaseQueryTool(Tool):
    """Queries a database."""

    def __init__(self):
        super().__init__(
            name="database_query",
            description="Executes a read-only SQL query",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"],
            },
        )

    def __call__(self, query: str) -> str:
        if not self._is_connected():
            # Unfixable: no amount of retrying will help
            raise AutomationError("Database connection is not available")

        if "DROP" in query.upper():
            # Fixable: the agent can rephrase the query
            raise ValueError("Only SELECT queries are allowed")

        return self._execute(query)
```

To use this tool with the ComputerAgent, you can run
```python
from askui import ComputerAgent
from helpers.tools.greeting_tool import GreetingTool

with ComputerAgent() as agent:
    agent.act(
        "Greet John in the morning using Spanish",
        tools=[GreetingTool()],
    )
```
