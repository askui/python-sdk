"""AskUI Python SDK"""

__version__ = "0.23.1"

import logging
import os

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

from .agent import ComputerAgent, VisionAgent
from .agent_base import Agent
from .agent_settings import AgentSettings
from .locators import Locator
from .models import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    CitationCharLocationParam,
    CitationContentBlockLocationParam,
    CitationPageLocationParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    OnMessageCb,
    OnMessageCbParam,
    Point,
    PointList,
    TextBlockParam,
    TextCitationParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UrlImageSourceParam,
)
from .models.shared.conversation_callback import ConversationCallback
from .models.shared.settings import (
    DEFAULT_GET_RESOLUTION,
    DEFAULT_LOCATE_RESOLUTION,
    ActSettings,
    GetSettings,
    LocateSettings,
    MessageSettings,
    Resolution,
)
from .models.shared.tools import Tool
from .models.types.response_schemas import ResponseSchema, ResponseSchemaBase
from .retry import ConfigurableRetry, Retry
from .tools import ModifierKey, PcKey
from .utils.image_utils import ImageSource
from .utils.source_utils import InputSource

try:
    from .android_agent import AndroidAgent, AndroidVisionAgent

    _ANDROID_AGENT_AVAILABLE = True
except ImportError:
    _ANDROID_AGENT_AVAILABLE = False

try:
    from .web_agent import WebVisionAgent
    from .web_testing_agent import WebTestingAgent

    _WEB_AGENTS_AVAILABLE = True
except ImportError:
    _WEB_AGENTS_AVAILABLE = False

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Agent",
    "ComputerAgent",
    "VisionAgent",
    "AgentSettings",
    "ActSettings",
    "Base64ImageSourceParam",
    "CacheControlEphemeralParam",
    "CitationCharLocationParam",
    "CitationContentBlockLocationParam",
    "CitationPageLocationParam",
    "ConfigurableRetry",
    "ContentBlockParam",
    "ConversationCallback",
    "DEFAULT_GET_RESOLUTION",
    "DEFAULT_LOCATE_RESOLUTION",
    "GetSettings",
    "ImageBlockParam",
    "ImageSource",
    "InputSource",
    "Locator",
    "LocateSettings",
    "MessageParam",
    "MessageSettings",
    "ModifierKey",
    "OnMessageCb",
    "OnMessageCbParam",
    "PcKey",
    "Point",
    "PointList",
    "Resolution",
    "ResponseSchema",
    "ResponseSchemaBase",
    "Retry",
    "TextBlockParam",
    "TextCitationParam",
    "Tool",
    "ToolResultBlockParam",
    "ToolUseBlockParam",
    "UrlImageSourceParam",
]

if _ANDROID_AGENT_AVAILABLE:
    __all__ += ["AndroidAgent", "AndroidVisionAgent"]

if _WEB_AGENTS_AVAILABLE:
    __all__ += ["WebVisionAgent", "WebTestingAgent"]
