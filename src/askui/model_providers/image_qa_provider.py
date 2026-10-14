"""ImageQAProvider interface for multimodal image question-answering."""

from abc import ABC, abstractmethod
from typing import Type

from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.utils.source_utils import Source


class ImageQAProvider(ABC):
    """Interface for providers that answer questions about images.

    An `ImageQAProvider` supports multimodal Q&A and structured output extraction
    from images and PDFs. It is used for `agent.get()` and the `GetTool`.

    The provider owns the model selection — the `model_id` is configured on the
    provider instance.

    Example:
        ```python
        from askui import AgentSettings, ComputerAgent
        from askui.model_providers import GoogleImageQAProvider

        provider = GoogleImageQAProvider(
            api_key="...",
            model_id="gemini-3.5-flash",
        )
        agent = ComputerAgent(settings=AgentSettings(image_qa_provider=provider))
        ```
    """

    @abstractmethod
    def query(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        """Extract information from an image or PDF based on a query.

        Args:
            query (str): What information to extract.
            source (Source): The image or document source to analyze.
            response_schema (Type[ResponseSchema] | None): Optional Pydantic model
                defining the expected response structure. If `None`, returns a string.
            get_settings (GetSettings): Settings controlling extraction behavior.

        Returns:
            ResponseSchema | str: Extracted data as a string or structured response.
        """
