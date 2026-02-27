from typing_extensions import override

from askui.models.shared.tools import Tool
from askui.tools.playwright.agent_os import PlaywrightAgentOs


class PlaywrightGotoTool(Tool):
    """
    Navigates to a specific URL in the browser.
    """

    def __init__(self, agent_os: PlaywrightAgentOs) -> None:
        super().__init__(
            name="playwright_goto_tool",
            description=(
                """
                Navigates the browser to a specific URL.
                This will load the webpage at the given URL and make it the current
                page. The browser will wait for the page to load completely before
                proceeding.
                """
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The URL to navigate to. Must be a valid URL including "
                            "the protocol (e.g., 'https://example.com')."
                        ),
                    },
                },
                "required": ["url"],
            },
        )
        self._agent_os = agent_os
        self.is_cacheable = True

    @override
    def __call__(self, url: str) -> str:
        self._agent_os.goto(url)
        return f"Navigated to: {url}"


class PlaywrightBackTool(Tool):
    """
    Navigates back to the previous page in the browser history.
    """

    def __init__(self, agent_os: PlaywrightAgentOs) -> None:
        super().__init__(
            name="playwright_back_tool",
            description=(
                """
                Navigates back to the previous page in the browser history.
                This is equivalent to clicking the back button in a browser.
                If there is no previous page in the history, this action will have no
                effect.
                """
            ),
        )
        self._agent_os = agent_os
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        self._agent_os.back()
        return "Navigated back to the previous page"


class PlaywrightForwardTool(Tool):
    """
    Navigates forward to the next page in the browser history.
    """

    def __init__(self, agent_os: PlaywrightAgentOs) -> None:
        super().__init__(
            name="playwright_forward_tool",
            description=(
                """
                Navigates forward to the next page in the browser history.
                This is equivalent to clicking the forward button in a browser.
                If there is no next page in the history, this action will have no
                effect.
                """
            ),
        )
        self._agent_os = agent_os
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        self._agent_os.forward()
        return "Navigated forward to the next page"


class PlaywrightGetPageTitleTool(Tool):
    """
    Gets the title of the current page.
    """

    def __init__(self, agent_os: PlaywrightAgentOs) -> None:
        super().__init__(
            name="playwright_get_page_title_tool",
            description=(
                """
                Retrieves the title of the currently loaded webpage.
                The title is typically displayed in the browser tab and represents
                the main heading or name of the page content.
                """
            ),
        )
        self._agent_os = agent_os
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        title = self._agent_os.get_page_title()
        return f"Page title: {title}"


class PlaywrightGetPageUrlTool(Tool):
    """
    Gets the URL of the current page.
    """

    def __init__(self, agent_os: PlaywrightAgentOs) -> None:
        super().__init__(
            name="playwright_get_page_url_tool",
            description=(
                """
                Retrieves the URL of the currently loaded webpage.
                This returns the full URL including protocol, domain, path, and query
                parameters.
                """
            ),
        )
        self._agent_os = agent_os
        self.is_cacheable = True

    @override
    def __call__(self) -> str:
        url = self._agent_os.get_page_url()
        return f"Current page URL: {url}"
