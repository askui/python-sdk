import inspect
import logging
import os
import platform
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import ParamSpec, TypeVar

from askui.telemetry.anonymous_id import get_anonymous_id
from askui.telemetry.context import (
    AppContext,
    CallStack,
    DeviceContext,
    OSContext,
    PlatformContext,
    TelemetryContext,
)
from askui.telemetry.pkg_version import get_pkg_version
from askui.telemetry.processors import SegmentSettings, TelemetryProcessor
from askui.telemetry.user_identification import (
    UserIdentification,
    UserIdentificationSettings,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class TelemetrySettings(BaseModel):
    """Settings for telemetry configuration"""

    user_identification: UserIdentificationSettings | None = (
        UserIdentificationSettings()
    )
    segment: SegmentSettings | None = SegmentSettings()
    app_name: str = "askui-vision-agent"
    app_version: str = get_pkg_version()
    group_id: str | None = Field(
        default=os.environ.get("ASKUI_WORKSPACE_ID"),
        description=(
            'The group ID of the user. Defaults to the "ASKUI_WORKSPACE_ID" '
            "environment variable if set, otherwise `None`."
        ),
    )
    device_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "The device ID of the host machine. "
            "This is used to identify the device and the user (if anynomous) across "
            "AskUI components. "
            "We hash it with an AskUI specific salt to avoid user tracking across "
            "(non-AskUI) applications or exposing the actual machine ID. This is the "
            "trade-off we chose for now to protect user privacy while still being able "
            "to improve the UX across components."
        ),
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "This is used to identify the current (test/automation) session. "
            "Defaults to a random UUID4. "
            "This should be overridden if the test/automation is split across "
            "multiple processes to associate all of them with a single session."
        ),
    )
    enabled: bool = False


class Telemetry:
    _EXCLUDE_MASK = "masked"

    def __init__(self, settings: TelemetrySettings) -> None:
        self._settings = settings
        self._processors: list[TelemetryProcessor] = []
        self._user_identification: UserIdentification | None = None
        if not self._settings.enabled:
            logger.debug(
                "Telemetry is disabled. To enable it, set the "
                "`ASKUI__VA__TELEMETRY__ENABLED` environment variable to `True`."
            )
            return

        logger.debug(
            "Telemetry is enabled. To disable it, set the "
            "`ASKUI__VA__TELEMETRY__ENABLED` environment variable to `False`."
        )
        if self._settings.user_identification:
            self._user_identification = UserIdentification(
                settings=self._settings.user_identification
            )
        self._call_stack = CallStack()
        self._context = self._init_context()

    def set_processors(self, processors: list[TelemetryProcessor]) -> None:
        """Set the telemetry processors that will be called in order

        *IMPORTANT*: This will replace the existing processors.

        Args:
            processors (list[TelemetryProcessor]): The list of telemetry processors to set (may be empty)
        """
        self._processors = processors

    def add_processor(self, processor: TelemetryProcessor) -> None:
        """Add a telemetry processor that will be called in order of addition

        Args:
            processor (TelemetryProcessor): The telemetry processor to add
        """
        self._processors.append(processor)

    def _init_context(self) -> TelemetryContext:
        context = TelemetryContext(
            anonymous_id=get_anonymous_id(),
            app=AppContext(
                name=self._settings.app_name,
                version=self._settings.app_version,
            ),
            os=OSContext(
                name=platform.system(),
                version=platform.version(),
                release=platform.release(),
            ),
            platform=PlatformContext(
                arch=platform.machine(),
                python_version=platform.python_version(),
            ),
            session_id=self._settings.session_id,
        )
        if self._settings.group_id:
            context["group_id"] = self._settings.group_id
        if self._user_identification:
            user_id = self._user_identification.get_user_id()
            if user_id:
                context["user_id"] = user_id
        if self._settings.device_id:
            context["device"] = DeviceContext(id=self._settings.device_id)
        return context

    def record_call(  # noqa: C901
        self,
        exclude: set[str] | None = None,
        exclude_first_arg: bool = True,
        exclude_response: bool = True,
        exclude_exception: bool = False,
        exclude_start: bool = True,
        flush: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to record calls to functions and methods

        IMPORTANT: Parameters, responses and exceptions recorded must be serializable
        to JSON. Either make sure that they are serializable or exclude them using
        the `exclude` parameter or use the `exclude_response` and `exclude_exception`
        parameters.

        Args:
            exclude (set[str] | None, optional): Set of parameters whose values are to
                be excluded from tracking (masked to retain structure of the call).
                Defaults to `None`.
            exclude_first_arg (bool, optional): Whether to exclude the first argument,
                e.g., `self` or `cls` for instance and class methods. Defaults to
                `True`, for functions and static methods, it should be set to `False`.
            exclude_response (bool, optional): Whether to exclude the response of the
                function in the telemetry event. Defaults to `True`.
            exclude_exception (bool, optional): Whether to exclude the exception if one
                is raised in the telemetry event. Defaults to `False`.
            exclude_start (bool, optional): Whether to exclude the start of the function
                call as a telemetry event. Defaults to `True`.
            flush (bool, optional): Whether to flush the telemetry data to the backend(s)
                after recording an event. Defaults to `False`. Should be set to `True` if
                an event. Defaults to `False`. Should be set to `True` if the process is
                expected to exit afterwards. Setting it to `True` can have a slightly
                negative impact on performance but ensures that telemetry data is not
                lost in case of a crash.
        """

        _exclude = exclude or set()

        def decorator(func: Callable[P, R]) -> Callable[P, R]:  # noqa: C901
            param_names_sorted = list(inspect.signature(func).parameters.keys())

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # noqa: C901
                if not self._settings.enabled:
                    return func(*args, **kwargs)

                self._call_stack.push_call()
                module = func.__module__
                fn_name = func.__qualname__
                logger.debug(
                    "Record call",
                    extra={"fn_name": fn_name, "module_": module},
                )
                processed_args: tuple[Any, ...] = tuple(
                    arg if param_names_sorted[i] not in _exclude else self._EXCLUDE_MASK
                    for i, arg in enumerate(args)
                )
                if exclude_first_arg:
                    processed_args = processed_args[1:] if processed_args else ()

                def _serialize_arg(arg: Any) -> Any:
                    if to_telemetry := getattr(arg, "to_telemetry_dict", None):
                        if callable(to_telemetry):
                            return to_telemetry()
                    if isinstance(arg, BaseModel):
                        return arg.model_dump()
                    if inspect.isclass(arg):
                        return str(arg)
                    return arg

                processed_args = tuple(_serialize_arg(arg) for arg in processed_args)
                processed_kwargs = {
                    k: v if k not in _exclude else self._EXCLUDE_MASK
                    for k, v in kwargs.items()
                }
                processed_kwargs = {
                    k: _serialize_arg(v) for k, v in processed_kwargs.items()
                }
                attributes: dict[str, Any] = {
                    "module": module,
                    "fn_name": fn_name,
                    "args": processed_args,
                    "kwargs": processed_kwargs,
                    "call_id": self._call_stack.current[-1],
                }
                self._context["call_stack"] = self._call_stack.current
                if not exclude_start:
                    for processor in self._processors:
                        processor.record_event(
                            name="Function call started",
                            attributes=attributes,
                            context=self._context,
                        )
                start_time = time.time()
                try:
                    response = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    attributes["duration_ms"] = duration_ms
                    if not exclude_response:
                        attributes["response"] = response
                    for processor in self._processors:
                        processor.record_event(
                            name="Function called",
                            attributes=attributes,
                            context=self._context,
                        )
                    if flush:
                        self.flush()
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    attributes["duration_ms"] = duration_ms
                    if not exclude_exception:
                        attributes["exception"] = {
                            "type": type(e).__name__,
                            "message": str(e),
                        }
                    for processor in self._processors:
                        processor.record_event(
                            name="Function called",
                            attributes=attributes,
                            context=self._context,
                        )
                    if flush:
                        self.flush()
                    raise
                else:
                    return response
                finally:
                    self._call_stack.pop_call()

            return wrapper

        return decorator

    def flush(self) -> None:
        """Flush the telemetry data to the backend"""
        for processor in self._processors:
            processor.flush()
