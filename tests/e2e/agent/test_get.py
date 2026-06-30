import pathlib
from typing import Literal, Type

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, RootModel
from pytest_mock import MockerFixture
from typing_extensions import override

from askui import AgentSettings, ComputerAgent, ResponseSchemaBase
from askui.model_providers.image_qa_provider import ImageQAProvider
from askui.models import ModelName
from askui.models.anthropic.factory import create_api_client
from askui.models.anthropic.get_model import AnthropicGetModel
from askui.models.anthropic.messages_api import AnthropicMessagesApi
from askui.models.askui.get_model import AskUiGeminiGetModel
from askui.models.askui.inference_api import AskUiInferenceApiSettings
from askui.models.models import GetModel
from askui.models.shared.settings import GetSettings
from askui.models.types.response_schemas import ResponseSchema
from askui.reporting import Reporter
from askui.tools.agent_os import ComputerAgentOS
from askui.utils.source_utils import Source


class _GetModelImageQAProvider(ImageQAProvider):
    """Adapter wrapping a `GetModel` as an `ImageQAProvider` for e2e tests."""

    def __init__(self, get_model: GetModel) -> None:
        self._get_model = get_model

    @override
    def query(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        get_settings: GetSettings,
    ) -> ResponseSchema | str:
        result: ResponseSchema | str = self._get_model.get(
            query=query,
            source=source,
            response_schema=response_schema,
            get_settings=get_settings,
        )
        return result


class UrlResponse(ResponseSchemaBase):
    url: str


class PageContextResponse(UrlResponse):
    title: str


class BrowserContextResponse(ResponseSchemaBase):
    page_context: PageContextResponse
    browser_type: Literal["chrome", "firefox", "edge", "safari"]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(None, id="default"),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
        pytest.param(
            AnthropicGetModel(
                model_id=ModelName.CLAUDE__SONNET__4__20250514,
                messages_api=AnthropicMessagesApi(
                    client=create_api_client(api_provider="anthropic"),
                ),
            ),
            id="claude",
        ),
    ],
)
def test_get(
    vision_agent: ComputerAgent,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel | None,
) -> None:
    if get_model is None:
        url = vision_agent.get(
            "What is the current url shown in the url bar?\nUrl: ",
            source=github_login_screenshot,
        )
    else:
        with ComputerAgent(
            settings=AgentSettings(
                image_qa_provider=_GetModelImageQAProvider(get_model)
            ),
            reporters=[simple_html_reporter],
        ) as agent:
            url = agent.get(
                "What is the current url shown in the url bar?\nUrl: ",
                source=github_login_screenshot,
            )
    assert url in ["github.com/login", "https://github.com/login"]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
    ],
)
def test_get_with_pdf_with_gemini_model(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    get_model: GetModel,
    path_fixtures_dummy_pdf: pathlib.Path,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is in the PDF? explain in 1 sentence",
            source=path_fixtures_dummy_pdf,
        )
    assert isinstance(response, str)
    assert "is a test " in response.lower()


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
    ],
)
def test_get_with_pdf_too_large(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    get_model: GetModel,
    path_fixtures_dummy_pdf: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch("askui.models.askui.get_model.MAX_FILE_SIZE_BYTES", 1)
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        with pytest.raises(ValueError, match="PDF file size exceeds the limit"):
            agent.get(
                "What is in the PDF?",
                source=path_fixtures_dummy_pdf,
            )


def test_get_with_pdf_too_large_with_default_model(
    vision_agent: ComputerAgent,
    path_fixtures_dummy_pdf: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "askui.model_providers.askui_image_qa_provider._MAX_FILE_SIZE_BYTES", 1
    )

    with pytest.raises(ValueError, match="PDF file size exceeds the limit"):
        vision_agent.get("What is in the PDF?", source=path_fixtures_dummy_pdf)


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
    ],
)
def test_get_with_xlsx_with_gemini_model(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    get_model: GetModel,
    path_fixtures_dummy_excel: pathlib.Path,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the salary of Doe?",
            source=path_fixtures_dummy_excel,
        )
    assert isinstance(response, str)
    assert "20000" in response.lower()


class Salary(ResponseSchemaBase):
    salary: int
    name: str


class SalaryResponse(ResponseSchemaBase):
    salaries: list[Salary]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
    ],
)
def test_get_with_xlsx_with_gemini_model_with_response_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    get_model: GetModel,
    path_fixtures_dummy_excel: pathlib.Path,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the salary of Everyone?",
            source=path_fixtures_dummy_excel,
            response_schema=SalaryResponse,
        )
    assert isinstance(response, SalaryResponse)
    # sort salaries by name for easier assertion
    response.salaries.sort(key=lambda x: x.name)
    assert response.salaries[0].name == "Doe"
    assert response.salaries[0].salary == 20000
    assert response.salaries[1].name == "John"
    assert response.salaries[1].salary == 10000


def test_get_with_xlsx_with_default_model_with_chart_data(
    vision_agent: ComputerAgent, path_fixtures_dummy_excel: pathlib.Path
) -> None:
    response = vision_agent.get(
        "What is the salary of John?", source=path_fixtures_dummy_excel
    )
    assert isinstance(response, str)
    assert "10000" in response.lower()


def test_get_with_docs_with_default_model(
    vision_agent: ComputerAgent, path_fixtures_dummy_doc: pathlib.Path
) -> None:
    response = vision_agent.get(
        "At what time in 24h format does the person sleeps?",
        source=path_fixtures_dummy_doc,
    )
    assert isinstance(response, str)
    assert "22:00" in response.lower()


def test_get_with_fallback_model(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
) -> None:
    askui_get_model = AskUiGeminiGetModel(
        model_id=ModelName.GEMINI__2_5__FLASH,
        inference_api_settings=AskUiInferenceApiSettings(),
    )
    with ComputerAgent(
        settings=AgentSettings(
            image_qa_provider=_GetModelImageQAProvider(askui_get_model)
        ),
        reporters=[simple_html_reporter],
    ) as agent:
        url = agent.get(
            "What is the current url shown in the url bar?",
            source=github_login_screenshot,
        )
        assert url in ["github.com/login", "https://github.com/login"]


class UrlResponseBaseModel(BaseModel):
    url: str


def test_get_with_response_schema_without_additional_properties_with_askui_model_raises(
    vision_agent: ComputerAgent, github_login_screenshot: PILImage.Image
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        vision_agent.get(
            "What is the current url shown in the url bar?",
            source=github_login_screenshot,
            response_schema=UrlResponseBaseModel,  # type: ignore[type-var]
        )


class OptionalUrlResponse(ResponseSchemaBase):
    url: str = "github.com"


def test_get_with_response_schema_with_default_value(
    vision_agent: ComputerAgent, github_login_screenshot: PILImage.Image
) -> None:
    response = vision_agent.get(
        "What is the current url shown in the url bar?",
        source=github_login_screenshot,
        response_schema=OptionalUrlResponse,
    )
    assert isinstance(response, OptionalUrlResponse)
    assert "github.com" in response.url


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(None, id="default"),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_response_schema(
    vision_agent: ComputerAgent,
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel | None,
) -> None:
    if get_model is None:
        response = vision_agent.get(
            "What is the current url shown in the url bar?",
            source=github_login_screenshot,
            response_schema=UrlResponse,
        )
    else:
        with ComputerAgent(
            settings=AgentSettings(
                image_qa_provider=_GetModelImageQAProvider(get_model)
            ),
            reporters=[simple_html_reporter],
        ) as agent:
            response = agent.get(
                "What is the current url shown in the url bar?",
                source=github_login_screenshot,
                response_schema=UrlResponse,
            )
    assert isinstance(response, UrlResponse)
    assert response.url in ["https://github.com/login", "github.com/login"]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_nested_and_inherited_response_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the current browser context?",
            source=github_login_screenshot,
            response_schema=BrowserContextResponse,
        )
    assert isinstance(response, BrowserContextResponse)
    assert response.page_context.url in ["https://github.com/login", "github.com/login"]
    assert "GitHub" in response.page_context.title
    assert response.browser_type in ["chrome", "firefox", "edge", "safari"]


class LinkedListNode(ResponseSchemaBase):
    value: str
    next: "LinkedListNode | None"


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_recursive_response_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        with pytest.raises(
            NotImplementedError, match="Recursive response schemas are not supported"
        ):
            agent.get(
                "Can you extract all segments (domain, path etc.) from the url as a linked list, "
                "e.g. 'https://google.com/test' -> 'google.com->test->None'?",
                source=github_login_screenshot,
                response_schema=LinkedListNode,
            )


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_string_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the current url shown in the url bar?",
            source=github_login_screenshot,
            response_schema=str,
        )
    assert response in ["https://github.com/login", "github.com/login"]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_flash",
        ),
    ],
)
def test_get_with_boolean_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "Is this a login page?",
            source=github_login_screenshot,
            response_schema=bool,
        )
    assert isinstance(response, bool)
    assert response is True


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_integer_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "How many input fields are visible on this page?",
            source=github_login_screenshot,
            response_schema=int,
        )
    assert isinstance(response, int)
    assert response > 0


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_float_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "Return a floating point number between 0 and 1 as a rating for how you well this page is designed (0 is the worst, 1 is the best)",
            source=github_login_screenshot,
            response_schema=float,
        )
    assert isinstance(response, float)
    assert response > 0


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_returns_str_when_no_schema_specified(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the display showing?",
            source=github_login_screenshot,
        )
    assert isinstance(response, str)


class Basis(ResponseSchemaBase):
    answer: str


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_basis_schema(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the display showing?",
            source=github_login_screenshot,
            response_schema=Basis,
        )
    assert isinstance(response, Basis)
    assert isinstance(response.answer, str)


class Answer(ResponseSchemaBase):
    answer: str


class BasisWithNestedRootModel(ResponseSchemaBase):
    answer: RootModel[Answer]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__FLASH,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="askui",
        ),
    ],
)
def test_get_with_nested_root_model(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "What is the display showing?",
            source=github_login_screenshot,
            response_schema=BasisWithNestedRootModel,
        )
    assert isinstance(response, BasisWithNestedRootModel)
    assert isinstance(response.answer.root.answer, str)


class PageDomElementLevel4(ResponseSchemaBase):
    tag: str
    text: str | None = None


class PageDomElementLevel3(ResponseSchemaBase):
    tag: str
    children: list["PageDomElementLevel4"]
    text: str | None = None


class PageDomElementLevel2(ResponseSchemaBase):
    tag: str
    children: list["PageDomElementLevel3"]
    text: str | None = None


class PageDomElementLevel1(ResponseSchemaBase):
    tag: str
    children: list["PageDomElementLevel2"]
    text: str | None = None


class PageDom(ResponseSchemaBase):
    children: list[PageDomElementLevel1]


@pytest.mark.parametrize(
    "get_model",
    [
        pytest.param(
            AskUiGeminiGetModel(
                model_id=ModelName.GEMINI__2_5__PRO,
                inference_api_settings=AskUiInferenceApiSettings(),
            ),
            id="gemini_pro",
        ),
    ],
)
def test_get_with_deeply_nested_response_schema_with_model_that_does_not_support_recursion(
    agent_os_mock_patch: ComputerAgentOS,  # noqa: ARG001
    simple_html_reporter: Reporter,
    github_login_screenshot: PILImage.Image,
    get_model: GetModel,
) -> None:
    """Test for deeply nested structure with 4 levels of nesting.

    This test case reproduces an issue reported by a user where they encountered
    problems with a deeply nested structure containing 4 levels of nesting.
    """
    with ComputerAgent(
        settings=AgentSettings(image_qa_provider=_GetModelImageQAProvider(get_model)),
        reporters=[simple_html_reporter],
    ) as agent:
        response = agent.get(
            "Create a possible dom of the page that goes 4 levels deep",
            source=github_login_screenshot,
            response_schema=PageDom,
        )
    assert isinstance(response, PageDom)
