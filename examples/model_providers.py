"""Example demonstrating all available model providers.

This example shows how to configure and use each provider type:
- VLM providers for act() - AskUI, Anthropic
- ImageQA providers for get() - AskUI, Anthropic, Google
- Detection providers for locate() - AskUI

Required environment variables (see .env):
- ASKUI_WORKSPACE_ID, ASKUI_TOKEN - for AskUI providers
- ANTHROPIC_API_KEY - for Anthropic providers
- GOOGLE_API_KEY - for Google provider
"""

import logging
import os

from askui import AgentSettings, ComputerAgent
from askui.model_providers import (
    AnthropicImageQAProvider,
    AnthropicVlmProvider,
    AskUIDetectionProvider,
    AskUIImageQAProvider,
    AskUIVlmProvider,
    GoogleImageQAProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


def try_act(agent: ComputerAgent, provider_name: str) -> None:
    """Test the act() method with a simple goal."""
    logger.info(f"Testing act() with {provider_name}")
    agent.act(
        goal=("Open a new Chrome window and navigate to askui.com. "),
    )


def try_locate(agent: ComputerAgent, provider_name: str) -> None:
    """Test the locate() method."""
    logger.info(f"Testing locate() with {provider_name}")
    point = agent.locate("The green button")
    logger.info(f"Located element at: {point}")


def try_get(agent: ComputerAgent, provider_name: str) -> None:
    """Test the get() method."""
    logger.info(f"Testing get() with {provider_name}")
    result = agent.get("What application or website is currently visible?")
    logger.info(f"Result: {result}")


def create_askui_providers() -> AgentSettings:
    """Create settings using AskUI-hosted providers (default)."""
    return AgentSettings(
        vlm_provider=AskUIVlmProvider(model_id="claude-sonnet-4-5-20250929"),
        image_qa_provider=AskUIImageQAProvider(model_id="gemini-3.5-flash"),
        detection_provider=AskUIDetectionProvider(),
    )


def create_anthropic_providers() -> AgentSettings:
    """Create settings using direct Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return AgentSettings(
        vlm_provider=AnthropicVlmProvider(
            api_key=api_key,
            model_id="claude-sonnet-4-5-20250929",
        ),
        image_qa_provider=AnthropicImageQAProvider(
            api_key=api_key,
            model_id="claude-haiku-4-5-20251001",
        ),
        # Detection still uses AskUI (no Anthropic equivalent)
        detection_provider=AskUIDetectionProvider(),
    )


def create_google_providers() -> AgentSettings:
    """Create settings using Google Gemini for image Q&A."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return AgentSettings(
        # VLM uses AskUI (no direct Google VLM provider yet)
        vlm_provider=AskUIVlmProvider(model_id="claude-sonnet-4-5-20250929"),
        image_qa_provider=GoogleImageQAProvider(
            api_key=api_key,
            model_id="gemini-3.5-flash",
        ),
        detection_provider=AskUIDetectionProvider(),
    )


def create_mixed_providers() -> AgentSettings:
    """Create settings mixing different providers for optimal results."""
    return AgentSettings(
        # Anthropic for agentic tasks (strong reasoning)
        vlm_provider=AnthropicVlmProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model_id="claude-sonnet-4-5-20250929",
        ),
        # Google for fast image Q&A
        image_qa_provider=GoogleImageQAProvider(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            model_id="gemini-3.5-flash",
        ),
        # AskUI for element detection
        detection_provider=AskUIDetectionProvider(),
    )


if __name__ == "__main__":
    # Define which provider configurations to test
    provider_configs = {
        # "AskUI (default)": create_askui_providers,
        # "Anthropic (direct)": create_anthropic_providers,
        # "Google (Gemini)": create_google_providers,
        "Mixed providers": create_mixed_providers,
    }

    # Select which configuration to run (change this to test different providers)
    selected_config = "AskUI (default)"

    for selected_config in provider_configs.keys():
        logger.info("#" * 60)
        logger.info(f"Running with provider configuration: {selected_config}")
        logger.info("#" * 60)
        settings = provider_configs[selected_config]()

        with ComputerAgent(settings=settings, display=1) as agent:
            # Test all three methods
            try_act(agent, selected_config)
            try_get(agent, selected_config)
            try_locate(agent, selected_config)

        logger.info("Done!")
