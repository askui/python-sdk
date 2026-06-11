import webbrowser

import httpx
import pyperclip

from askui.tools.agent_os import ComputerAgentOS


class AgentToolbox:
    """
    Toolbox for agent.

    Provides access to OS-level actions, clipboard, web browser, HTTP client etc.

    Args:
        agent_os (ComputerAgentOS): The OS interface implementation to use for
            agent actions.

    Attributes:
        webbrowser: Python's built-in `webbrowser` module for opening URLs.
        clipboard: `pyperclip` module for clipboard access.
        agent_os (ComputerAgentOS): The OS interface for mouse, keyboard, and
            screen actions.
        httpx: HTTPX client for HTTP requests.
    """

    def __init__(self, agent_os: ComputerAgentOS):
        self.webbrowser = webbrowser
        self.clipboard = pyperclip
        self.os = agent_os
        self.httpx = httpx
