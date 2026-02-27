import abc
import re
from typing import Type

from pydantic import BaseModel, ConfigDict

from askui.locators.locators import Locator
from askui.models.shared.agent_message_param import MessageParam
from askui.models.shared.settings import ActSettings, GetSettings, LocateSettings
from askui.models.shared.tools import ToolCollection
from askui.models.types.geometry import Point, PointList
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.image_utils import ImageSource
from askui.utils.source_utils import Source


class ModelName:
    """Enumeration of all available model names in AskUI.

    This enum is not really an `enum.Enum` but rather a collection of literal strings.
    It provides type-safe access to model identifiers used throughout the
    library. Each model name corresponds to a specific AI model or model composition
    that can be used for different tasks like acting, getting information, or locating
    elements.
    """

    ASKUI = "askui"
    ASKUI__GEMINI__2_5__FLASH = "askui/gemini-2.5-flash"
    ASKUI__GEMINI__2_5__PRO = "askui/gemini-2.5-pro"
    ASKUI__AI_ELEMENT = "askui-ai-element"
    ASKUI__COMBO = "askui-combo"
    ASKUI__OCR = "askui-ocr"
    ASKUI__PTA = "askui-pta"
    CLAUDE__SONNET__4__20250514 = "claude-sonnet-4-20250514"
    GEMINI__2_5__FLASH = "gemini-2.5-flash"
    GEMINI__2_5__PRO = "gemini-2.5-pro"
    HF__SPACES__ASKUI__PTA_1 = "AskUI/PTA-1"
    HF__SPACES__OS_COPILOT__OS_ATLAS_BASE_7B = "OS-Copilot/OS-Atlas-Base-7B"
    HF__SPACES__QWEN__QWEN2_VL_2B_INSTRUCT = "Qwen/Qwen2-VL-2B-Instruct"
    HF__SPACES__QWEN__QWEN2_VL_7B_INSTRUCT = "Qwen/Qwen2-VL-7B-Instruct"
    HF__SPACES__SHOWUI__2B = "showlab/ShowUI-2B"


MODEL_DEFINITION_PROPERTY_REGEX_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class BoundingBox(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
    )

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @staticmethod
    def from_json(data: dict[str, float]) -> "BoundingBox":
        return BoundingBox(
            xmin=int(data["xmin"]),
            ymin=int(data["ymin"]),
            xmax=int(data["xmax"]),
            ymax=int(data["ymax"]),
        )

    def __str__(self) -> str:
        return f"[{self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}]"

    @property
    def width(self) -> int:
        """The width of the bounding box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """The height of the bounding box."""
        return self.ymax - self.ymin

    @property
    def center(self) -> Point:
        """The center point of the bounding box."""
        return int((self.xmin + self.xmax) / 2), int((self.ymin + self.ymax) / 2)


class DetectedElement(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
    )

    name: str
    text: str
    bounding_box: BoundingBox

    @staticmethod
    def from_json(data: dict[str, str | float | dict[str, float]]) -> "DetectedElement":
        return DetectedElement(
            name=str(data["name"]),
            text=str(data["text"]),
            bounding_box=BoundingBox.from_json(data["bndbox"]),  # type: ignore
        )

    def __str__(self) -> str:
        return f"[name={self.name}, text={self.text}, bndbox={str(self.bounding_box)}]"

    @property
    def center(self) -> Point:
        """The center point of the detected element."""
        return self.bounding_box.center

    @property
    def width(self) -> int:
        """The width of the detected element."""
        return self.bounding_box.width

    @property
    def height(self) -> int:
        """The height of the detected element."""
        return self.bounding_box.height


class ActModel(abc.ABC):
    """Abstract base class for models that can execute autonomous actions.

    Models implementing this interface can be used with the `VisionAgent.act()`.

    Example:
        ```python
        from askui import (
            ActModel,
            MessageParam,
            OnMessageCb,
            VisionAgent,
        )
        from typing_extensions import override

        class MyActModel(ActModel):
            @override
            def act(
                self,
                messages: list[MessageParam],
                model: str,
                tools: list[Tool] | None = None,
                settings: AgentSettings | None = None,
            ) -> None:
                pass  # implement action logic here

        with VisionAgent(models={"my-act": MyActModel()}) as agent:
            agent.act("search for flights", model="my-act")
    """

    @abc.abstractmethod
    def act(
        self,
        messages: list[MessageParam],
        act_settings: ActSettings,
        tools: ToolCollection | None = None,
    ) -> None:
        """
        Execute autonomous actions to achieve a goal, using a message history
        and optional callbacks, encoded in the messages. In the simplest case,
        it can be found in the first message `messages[0].content` as a `str`.

        The `messages` usually start with a `"user"` (role) message which is followed by
        alternating `"assistant"` (AI agent) and `"user"` messages (which can be
        automatic tool use, e.g., taking a screenshot) similar how you would
        expect it from a conversation whereby the `"assistant"` determines the next
        actions which are then automatically taking by the `"user"` programmatically
        until it eventually returns, usually with an `"assistant"` message that either
        says that the goal has been achieved or that it failed to achieve the goal.

        Args:
            messages (list[MessageParam]): The message history to start that
                determines the actions and following messages.
            tools (ToolCollection | None, optional): The tools for the agent.
                Defaults to `None`.
            act_settings (ActSettings): The settings for this act operation,
                passed from the agent.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented.
        """  # noqa: E501
        raise NotImplementedError

    def to_telemetry_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation for telemetry.

        For ActModels, this includes the model_id and MessagesApi class name.

        Returns:
            dict[str, str]: A dictionary with telemetry information about this model.
                Includes: "type", "class", "model_id", "messages_api".

        Example:
            ```python
            {
                "type": "act_model",
                "class": "AskUIAgent",
                "model_id": "claude-sonnet-4-20250514",
                "messages_api": "AnthropicMessagesApi"
            }
            ```
        """
        return {
            "type": "act_model",
            "class": self.__class__.__name__,
            "model_id": self.__getattribute__("_model_id" or "-"),
            "messages_api": self._messages_api.__class__.__name__,  # type: ignore[attr-defined]
        }


class GetModel(abc.ABC):
    """Abstract base class for models that can extract information from images and PDFs.

    Models implementing this interface can be used with the `get()` method of
    `VisionAgent`
    to extract information from screenshots, other images or PDFs. These models analyze
    visual content and return structured or unstructured information based on queries.
    Example:
        ```python
        from askui import GetModel, VisionAgent, ResponseSchema, Source
        from typing import Type

        class MyGetModel(GetModel):
            def get(
                self,
                query: str,
                source: Source,
                response_schema: Type[ResponseSchema] | None,
                model: str,
            ) -> ResponseSchema | str:
                # Implement custom get logic
                return "Custom response"

        with VisionAgent(models={"my-get": MyGetModel()}) as agent:
            result = agent.get("what's on screen?", model="my-get")
        ```
    """

    @abc.abstractmethod
    def get(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        """Extract information from a source based on a query.
        Args:
            query (str): A description of what information to extract
            source (Source): The source to analyze (screenshot, image or PDF)
            response_schema (Type[ResponseSchema] | None): Optional Pydantic model class
                defining the expected response structure
            get_settings (GetSettings): The settings for this get operation,
                passed from the agent

        Returns:
            Either a string response or a Pydantic model instance if response_schema is
            provided
        """
        raise NotImplementedError

    def to_telemetry_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation for telemetry.

        For GetModels, this includes the class name.

        Returns:
            dict[str, str]: A dictionary with telemetry information about this model.
                Includes: "type", "class".

        Example:
            ```python
            {
                "type": "get_model",
                "class": "AskUiGeminiGetModel"
            }
            ```
        """
        return {
            "type": "get_model",
            "class": self.__class__.__name__,
        }


class LocateModel(abc.ABC):
    """Abstract base class for models that can locate UI elements in images.

    Models implementing this interface can be used with the `click()`, `locate()`, and
    `mouse_move()` methods of `VisionAgent` to find UI elements on screen. These models
    analyze visual content to determine the coordinates of elements based on
    descriptions or locators.

    Example:
        ```python
        from askui import LocateModel, VisionAgent, Locator, ImageSource, PointList

        class MyLocateModel(LocateModel):
            def locate(
                self,
                locator: str | Locator,
                image: ImageSource,
            ) -> PointList:
                # Implement custom locate logic
                return [(100, 100)]

        with VisionAgent(models={"my-locate": MyLocateModel()}) as agent:
            agent.click("button", model="my-locate")
        ```
    """

    @abc.abstractmethod
    def locate(
        self,
        locator: str | Locator,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> PointList:
        """Find the coordinates of a UI element in an image.

        Args:
            locator (str | Locator): A description or locator object identifying the
                element to find
            image (ImageSource): The image to analyze (screenshot or provided image)
            locate_settings (LocateSettings): The settings for this locate operation,
                passed from the agent

        Returns:
            A list of (x, y) coordinates where the element was found, minimum length 1
        """
        raise NotImplementedError

    def locate_all_elements(
        self,
        image: ImageSource,
        locate_settings: LocateSettings,
    ) -> list[DetectedElement]:
        """Locate all elements in an image.

        Args:
            image (ImageSource): The image to analyze (screenshot or provided image)
            locate_settings (LocateSettings): The settings for this locate operation,
                passed from the agent

        Returns:
            A list of detected elements
        """
        raise NotImplementedError

    def to_telemetry_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation for telemetry.

        For LocateModels, this includes the class name.

        Returns:
            dict[str, str]: A dictionary with telemetry information about this model.
                Includes: "type", "class".

        Example:
            ```python
            {
                "type": "locate_model",
                "class": "AskUiPtaLocateModel"
            }
            ```
        """
        return {
            "type": "locate_model",
            "class": self.__class__.__name__,
        }


Model = ActModel | GetModel | LocateModel
"""Union type of all abstract model classes.

This type represents any model that can be used with `VisionAgent`, whether it's an
`ActModel`, `GetModel`, or `LocateModel`. It's useful for type hints when you need to
work with models in a generic way.
"""
