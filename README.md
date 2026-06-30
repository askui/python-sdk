# 🤖 AskUI Python SDK

[![Release Notes](https://img.shields.io/github/release/askui/vision-agent?style=flat-square)](https://github.com/askui/vision-agent/releases)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)

**Enable AI agents to control your desktop (Windows, MacOS, Linux), mobile (Android, iOS) and HMI devices**

Join the [AskUI Discord](https://discord.gg/Gu35zMGxbx).

## Why AskUI?

Traditional UI automation is fragile. Every time a button moves, a label changes, or a layout shifts, your scripts break. You're stuck maintaining brittle selectors, writing conditional logic for edge cases, and constantly updating tests.

**AskUI Agents solve this by combining two powerful approaches:**

1. **Vision-based automation** - Find UI elements by what they look like or say, not by brittle XPath or CSS selectors
2. **AI-powered agents** - Give high-level instructions and let AI figure out the steps
AskUI Python SDK is a powerful automation framework that enables you and AI agents to control your desktop, mobile, and HMI devices and automate tasks. With support for multiple AI models, multi-platform compatibility, and enterprise-ready features,

Whether you're automating desktop apps, testing mobile applications, or building RPA workflows, AskUI adapts to UI changes automatically—saving you hours of maintenance work.

## Key Features

- **Multi-platform** - Works on Windows, Linux, MacOS, and Android
- **Two modes** - Single-step UI commands or agentic intent-based instructions
- **Vision-first** - Find elements by text, images, or natural language descriptions
- **Model flexibility** - Anthropic Claude, Google Gemini, AskUI models, or bring your own
- **Extensible** - Add custom tools and capabilities via Model Context Protocol (MCP)
- **Caching** - Save expensive calls to AI APIs by using them only when really necessary

## Quick Examples

### Agentic UI Automation

Run the script with `python <file path>`, e.g `python test.py` to see if it works.

### 🤖 Let AI agents control your devices

In order to let AI agents control your devices, you need to be able to connect to an AI model (provider). We host some models ourselves and support several other ones, e.g. Anthropic, OpenRouter, Hugging Face, etc. out of the box. If you want to use a model provider or model that is not supported, you can easily plugin your own (see [Custom Models](docs/using-models.md#using-custom-models)).

For this example, we will us AskUI as the model provider to easily get started.

#### 🔐 Sign up with AskUI

Sign up at [hub.askui.com](https://hub.askui.com) to:
- Activate your **free trial** by signing up (no credit card required)
- Get your workspace ID and access token

#### ⚙️ Configure environment variables

<details>
<summary>Linux & MacOS</summary>

```shell
export ASKUI_WORKSPACE_ID=<your-workspace-id-here>
export ASKUI_TOKEN=<your-token-here>
```
</details>

<details>
<summary>Windows PowerShell</summary>

```shell
$env:ASKUI_WORKSPACE_ID="<your-workspace-id-here>"
$env:ASKUI_TOKEN="<your-token-here>"
```

</details>

#### 💻 Example

```python
from askui import ComputerAgent

with ComputerAgent() as agent:
    # Complex multi-step instruction
    agent.act(
        "Open a browser, navigate to GitHub, search for 'askui vision-agent', "
        "and star the repository"
    )

    # Extract information from the screen
    balance = agent.get("What is my current account balance?")
    print(f"Balance: {balance}")

    # Combine both approaches
    agent.act("Find the login form")
    agent.type("user@example.com")
    agent.click("Next")
```

### Extend with Custom Tools

Add new capabilities to your agents:

```python
from askui import ComputerAgent
from askui.tools.store.computer import ComputerSaveScreenshotTool
from askui.tools.store.universal import PrintToConsoleTool

with ComputerAgent() as agent:
    agent.act(
        "Take a screenshot of the current screen and save it, then confirm",
        tools=[
            ComputerSaveScreenshotTool(base_dir="./screenshots"),
            PrintToConsoleTool()
        ]
    )
```

## Getting Started

Ready to build your first agent? Check out our documentation:

0. **[Start Here](docs/00_overview.md)** - Overview and core concepts
1. **[Setup](docs/01_setup.md)** - Installation and configuration
2. **[Using Agents](docs/02_using_agents.md)** - Using the AskUI ComputerAgent and AndroidAgent
3. **[System Prompts](docs/03_prompting.md)** - How to write effective instructions
4. **[Using Models](docs/04_using_models.md)** - Using different models as backbone for act, get, and locate
5. **[BYOM](docs/05_bring_your_own_model_provider.md)** - use your own model cloud by plugging in your own model provider
6. **[Caching](docs/06_caching.md)** - Optimize performance and costs
7. **[Tools](docs/07_tools.md)** - Extend agent capabilities
8. **[Reporting](docs/08_reporting.md)** - Obtain agent logs as execution reports and summaries as test reports
9. **[Observability](docs/09_observability_telemetry_tracing.md)** - Monitor and debug agents
10. **[Extracting Data](docs/10_extracting_data.md)** - Extracting structured data from screenshots and files
11. **[Callbacks](docs/11_callbacks.md)** - Inject custom logic into the control loop
12. **[Secrets](docs/12_secrets.md)** - Let agents use sensitive values without exposing them to the LLM

**Official documentation:** [docs.askui.com](https://docs.askui.com)

## Quick Install

```bash
pip install askui
```

**Requires Python >=3.10*

The default install includes core dependencies (Agent OS client, HTTP, common model APIs, Android tooling, etc.). Add **optional extras** only when you need the features below.

```bash
pip install askui[all]
```

`all` pulls in every optional extra in one command (larger install). Prefer picking individual extras when you know what you need. Or start using AskUI with a minimal install and add extras as you need them.

### Optional install extras

| Extra | When to use it |
| --- | --- |
| *(none)* | Everyday automation. Smallest footprint. |
| `all` | You want every optional integration below without choosing extras (CI images, demos, or “install everything once”). |
| `office-document` | Reading or converting **Office files** (Excel `.xls`/`.xlsx`, Word `.docx`) via MarkItDown—e.g. extracting content from spreadsheets or documents in workflows. To be used in `get()` methods. |
| `bedrock` | Running **Anthropic Claude through AWS Bedrock** (`anthropic[bedrock]`). Use when your org routes Claude via Bedrock, not the direct Anthropic API. |
| `vertex` | Running **Anthropic Claude on Google Vertex AI** plus Vertex client libraries. Use when models are hosted on Vertex, not only the public Anthropic API. |
| `otel` | **OpenTelemetry** export and instrumentation you enable in code: OTLP over HTTP, plus optional instrumentation for `httpx` and SQLAlchemy. Use for production tracing/metrics pipelines—not required for local scripts. |
| `web` | **Browser automation with Playwright** (e.g. web-focused agents and Playwright-backed flows). Core desktop control still uses Agent OS; this extra is for the Playwright stack. |

Combine extras as needed, for example:

```bash
pip install askui[bedrock,otel]
```

You'll also need to install AskUI Agent OS for device control. See [Setup Guide](docs/01_setup.md) for detailed instructions.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
