# Overview

This documentation is intended for **developers** that want build custom implementation with this SDK.
For tutorials, how-to-guides, and more ressources please visit our user documentation [here](https://docs.askui.com/).

## Why AskUI Vision Agent?

AskUI Vision Agent is a powerful automation framework that bridges the gap between human intent and computer control. Whether you need precise UI automation for testing, want AI agents to handle complex workflows, or need to automate tasks across desktop, mobile, and HMI devices, AskUI Vision Agent provides a unified solution.

### The Problem

Traditional automation frameworks require:
- Brittle selectors that break when UI changes
- Separate tools for different platforms (web, desktop, mobile)
- Manual scripting of every action step
- Constant maintenance as applications evolve
- Random application behavior
- External issues like network, rights or installation issues

### The Solution

AskUI Vision Agent offers two complementary approaches:

**1. Agentic Control (Goal-based)**
```python
with ComputerAgent() as agent:
    agent.act(
        "Search for flights from New York to London, "
        "filter by direct flights, and show me the cheapest option"
    )
```
Natural language instructions where AI agents autonomously navigate UI, make decisions, and achieve goals. Perfect for complex workflows and exploratory testing.

### Key Capabilities

- **Multi-Platform**: Windows, MacOS, Linux, Android, Citric & KVM
- **Vision-Powered**: Uses AI models to understand UI visually, no selectors needed
- **Flexible Models**: Hot-swap between models, use AskUI-hosted, Anthropic, OpenRouter, or bring your own
- **Extensible**: Custom tools via MCP, custom models, custom prompts

## How This Documentation Is Structured

This documentation is organized to take you from setup to advanced usage:

### 01 - Setup
**Topics**: Installation, AgentOS setup, environment configuration, authentication

Start here to get AskUI Vision Agent installed and running. Covers Python package installation and the AgentOS component that enables cross-platform device control.

### 02 - Using Agents
**Topics**: ComputerAgent, AndroidAgent, WebVisionAgent, platform-specific tools

Learn about the three predefined agent types and their capabilities for desktop, mobile, and web automation.

### 03 - Prompting
**Topics**: System prompts, customizing agent behavior, prompt engineering best practices

Learn how to guide AI agents with effective prompts. Critical for the `act()` method and achieving reliable agentic behavior.

### 04 - Using Models
**Topics**: Model selection, available models via AskUI API

Understand how to choose and configure models for different tasks.

### 05 - Bring Your Own Model Provider
**Topics**: Custom model providers, Anthropic, Google, third-party integrations

Integrate custom models or use your own API keys with third-party providers.

### 06 - Caching
**Topics**: Recording trajectories, replaying actions, regression testing

Speed up test execution and improve reliability by recording and replaying successful action sequences.

### 07 - Tools
**Topics**: Tool store, Model Context Protocol (MCP), custom tools, extending agents

Access pre-built tools, integrate external services via MCP, and extend agent capabilities with custom tools.

### 08 - Reporting
**Topics**: HTML reports, custom reporters, debugging

Generate human-readable logs of agent actions for debugging and auditing.

### 09 - Observability, Telemetry, Tracing
**Topics**: Telemetry data, privacy, disabling telemetry

Understand what data is collected and how to opt out.

### 10 - Extracting Data
**Topics**: Using `get()`, file support (PDF. Excel, Word, CSV), structured data extraction, response schemas

### 11 - Callbacks
**Topics**: Inject custom logic at different positions of the control loop through callbacks

Extract information from screens and files using the `get()` method with Pydantic models.

## Additional Resources
- **GitHub Issues**: https://github.com/askui/vision-agent/issues
- **Official Docs**: https://docs.askui.com
- **Examples**: Check the `examples/` directory for complete working code

## Quick Start Example

Before starting, need to sign up at www.askui.com to get your credentials and then set them as environment variables (ASKUI_WORKSPACE_ID, and ASKUI_TOKEN)
Here's a complete example to get you started:

```python
import os
from askui import ComputerAgent

with ComputerAgent() as agent:
    agent.act(
        "Log in with email 'user@example.com' and password 'password123', "
        "then tell me what the welcome message says"
    )
```

