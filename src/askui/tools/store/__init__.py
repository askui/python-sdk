"""Tool Store - Optional tools that users can import and use.

Tools are organized by category:
- `android`: Tools specific to Android agents (require AndroidAgentOs)
- `computer`: Tools specific to Computer/Desktop agents (require ComputerAgentOsFacade)
- `universal`: Tools that work with any agent type (don't require ComputerAgentOS)

Example:
    ```python
    from askui import ComputerAgent
    from askui.tools.store.computer import ComputerSaveScreenshotTool
    from askui.tools.store.universal import PrintToConsoleTool

    with ComputerAgent() as agent:
        agent.act(
            "Save the current screen as demo/demo.png and keep me updated.",
            tools=[ComputerSaveScreenshotTool(base_dir="/path/to/screenshots"),
                PrintToConsoleTool()]
        )
    ```
"""
