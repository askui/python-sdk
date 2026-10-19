from typing import Any

import pytest

from askui.telemetry import InMemoryProcessor, Telemetry, TelemetrySettings


def test_telemetry_disabled() -> None:
    settings = TelemetrySettings(enabled=False)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call()
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10
    assert len(processor.get_events()) == 0


def test_telemetry_disabled_by_default() -> None:
    assert TelemetrySettings().enabled is False


def test_telemetry_enabled() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(
        exclude_first_arg=False, exclude_response=False, exclude_start=False
    )
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2

    start_event = events[0]
    assert start_event["name"] == "Function call started"
    assert start_event["attributes"]["fn_name"].endswith("test_func")
    assert start_event["attributes"]["args"] == (5,)
    assert start_event["attributes"]["kwargs"] == {}

    end_event = events[1]
    assert end_event["name"] == "Function called"
    assert end_event["attributes"]["fn_name"].endswith("test_func")
    assert end_event["attributes"]["args"] == (5,)
    assert end_event["attributes"]["kwargs"] == {}
    assert end_event["attributes"]["response"] == 10
    assert end_event["attributes"]["duration_ms"] is not None
    assert end_event["attributes"]["duration_ms"] >= 0


def test_telemetry_error() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(
        exclude_first_arg=False, exclude_start=False, exclude_exception=False
    )
    def test_func(x: int) -> int:  # noqa: ARG001
        error_msg = "Test error"
        raise ValueError(error_msg)

    with pytest.raises(ValueError):
        test_func(5)

    events = processor.get_events()
    assert len(events) == 2

    start_event = events[0]
    assert start_event["name"] == "Function call started"
    assert start_event["attributes"]["fn_name"].endswith("test_func")
    assert start_event["attributes"]["args"] == (5,)
    assert start_event["attributes"]["kwargs"] == {}

    error_event = events[1]
    assert error_event["name"] == "Function called"
    assert error_event["attributes"]["fn_name"].endswith("test_func")
    assert error_event["attributes"]["args"] == (5,)
    assert error_event["attributes"]["kwargs"] == {}
    assert error_event["attributes"]["exception"]["type"] == "ValueError"
    assert error_event["attributes"]["exception"]["message"] == "Test error"
    assert error_event["attributes"]["duration_ms"] is not None
    assert error_event["attributes"]["duration_ms"] >= 0


def test_multiple_processors() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor1 = InMemoryProcessor()
    processor2 = InMemoryProcessor()
    telemetry.add_processor(processor1)
    telemetry.add_processor(processor2)

    @telemetry.record_call(exclude_start=False)
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events1 = processor1.get_events()
    events2 = processor2.get_events()
    assert len(events1) == 2
    assert len(events2) == 2
    for e1, e2 in zip(events1, events2, strict=False):
        assert e1["name"] == e2["name"]
        assert e1["attributes"]["fn_name"] == e2["attributes"]["fn_name"]
        assert e1["attributes"]["args"] == e2["attributes"]["args"]
        assert e1["attributes"]["kwargs"] == e2["attributes"]["kwargs"]
        if "response" in e1["attributes"]:
            assert e1["attributes"]["response"] == e2["attributes"]["response"]
        if "exception" in e1["attributes"]:
            assert e1["attributes"]["exception"] == e2["attributes"]["exception"]
        assert e1["attributes"]["duration_ms"] == e2["attributes"]["duration_ms"]
        assert e1["timestamp"] <= e2["timestamp"]


def test_function_tracking() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_start=False)
    def standalone_function(x: int) -> int:
        return x * 2

    result = standalone_function(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2
    assert events[0]["attributes"]["fn_name"].endswith("standalone_function")
    assert events[1]["attributes"]["fn_name"].endswith("standalone_function")


def test_instance_method_tracking() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        @telemetry.record_call(exclude_start=False)
        def instance_method(self, x: int) -> int:
            return x * 2

    obj = TestClass()
    result = obj.instance_method(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2
    assert events[0]["attributes"]["fn_name"].endswith("TestClass.instance_method")
    assert events[1]["attributes"]["fn_name"].endswith("TestClass.instance_method")


def test_class_method_tracking() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        @classmethod
        @telemetry.record_call(exclude_start=False)
        def class_method(cls, x: int) -> int:
            return x * 3

    result = TestClass.class_method(5)
    assert result == 15

    events = processor.get_events()
    assert len(events) == 2
    assert events[0]["attributes"]["fn_name"].endswith("TestClass.class_method")
    assert events[1]["attributes"]["fn_name"].endswith("TestClass.class_method")


def test_static_method_tracking() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        @staticmethod
        @telemetry.record_call(exclude_start=False)
        def static_method(x: int) -> int:
            return x * 4

    result = TestClass.static_method(5)
    assert result == 20

    events = processor.get_events()
    assert len(events) == 2
    assert events[0]["attributes"]["fn_name"].endswith("TestClass.static_method")
    assert events[1]["attributes"]["fn_name"].endswith("TestClass.static_method")


def test_nested_class_tracking() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class Outer:
        class Inner:
            @telemetry.record_call(exclude_start=False)
            def nested_method(self, x: int) -> int:
                return x * 2

    result = Outer.Inner().nested_method(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2
    assert events[0]["attributes"]["fn_name"].endswith("Outer.Inner.nested_method")
    assert events[1]["attributes"]["fn_name"].endswith("Outer.Inner.nested_method")


def test_exclude_parameter() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(
        exclude={"password", "token"}, exclude_first_arg=False, exclude_start=False
    )
    def sensitive_function(username: str, password: str, token: str) -> str:  # noqa: ARG001
        return f"User: {username}"

    result = sensitive_function("test_user", "secret_password", "private_token")
    assert result == "User: test_user"

    events = processor.get_events()
    assert len(events) == 2

    # Check that excluded parameters are masked
    start_event = events[0]
    assert start_event["attributes"]["args"][0] == "test_user"  # username is included
    assert start_event["attributes"]["args"][1] == "masked"  # password is masked
    assert start_event["attributes"]["args"][2] == "masked"  # token is masked

    end_event = events[1]
    assert end_event["attributes"]["args"][0] == "test_user"
    assert end_event["attributes"]["args"][1] == "masked"
    assert end_event["attributes"]["args"][2] == "masked"


def test_exclude_kwargs() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(
        exclude={"password", "token"}, exclude_first_arg=False, exclude_start=False
    )
    def sensitive_function(username: str, **kwargs: Any) -> str:  # noqa: ARG001, ARG002
        return f"User: {username}"

    result = sensitive_function(
        "test_user", password="secret_password", token="private_token", visible="ok"
    )
    assert result == "User: test_user"

    events = processor.get_events()
    assert len(events) == 2

    # Check that excluded kwargs are masked but others aren't
    start_event = events[0]
    assert start_event["attributes"]["args"][0] == "test_user"
    assert start_event["attributes"]["kwargs"]["password"] == "masked"
    assert start_event["attributes"]["kwargs"]["token"] == "masked"
    assert start_event["attributes"]["kwargs"]["visible"] == "ok"


def test_include_first_arg_function() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_first_arg=False, exclude_start=False)
    def test_func(first: str, second: str) -> str:
        return f"{first}-{second}"

    result = test_func("one", "two")
    assert result == "one-two"

    events = processor.get_events()
    assert len(events) == 2

    # Check that first argument is included
    assert events[0]["attributes"]["args"] == ("one", "two")
    assert events[1]["attributes"]["args"] == ("one", "two")


def test_include_first_arg_method() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        def __init__(self, name: str):
            self.name = name

        @telemetry.record_call(exclude_first_arg=False, exclude_start=False)
        def method_with_self(self, param: str) -> str:
            return f"{self.name}-{param}"

    obj = TestClass("test")
    result = obj.method_with_self("param")
    assert result == "test-param"

    events = processor.get_events()
    assert len(events) == 2

    # Check that self is included as first argument
    assert len(events[0]["attributes"]["args"]) == 2
    assert events[0]["attributes"]["args"][1] == "param"  # Second arg should be param
    # Can't directly check self, but we can check it's not removed


def test_default_exclude_self_method() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        def __init__(self, name: str):
            self.name = name

        @telemetry.record_call(exclude_start=False)  # Default exclude_first_arg=True
        def method_with_self(self, param: str) -> str:
            return f"{self.name}-{param}"

    obj = TestClass("test")
    result = obj.method_with_self("param")
    assert result == "test-param"

    events = processor.get_events()
    assert len(events) == 2

    # Check that self is excluded
    assert events[0]["attributes"]["args"] == ("param",)
    assert events[1]["attributes"]["args"] == ("param",)


def test_combined_exclude_and_include_first_arg() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class User:
        @telemetry.record_call(
            exclude={"password"}, exclude_first_arg=False, exclude_start=False
        )
        def authenticate(self, username: str, password: str) -> bool:
            return username == "valid" and password == "correct"

    user = User()
    result = user.authenticate("valid", "correct")
    assert result is True

    events = processor.get_events()
    assert len(events) == 2

    # First arg (self) should be included, password should be masked
    assert len(events[0]["attributes"]["args"]) == 3
    assert events[0]["attributes"]["args"][1] == "valid"  # username is included
    assert events[0]["attributes"]["args"][2] == "masked"  # password is masked

    assert len(events[1]["attributes"]["args"]) == 3
    assert events[1]["attributes"]["args"][1] == "valid"
    assert events[1]["attributes"]["args"][2] == "masked"


def test_static_method_with_include_first_arg() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    class TestClass:
        @staticmethod
        @telemetry.record_call(exclude_first_arg=False, exclude_start=False)
        def static_method(first: str, second: str) -> str:
            return f"{first}-{second}"

    result = TestClass.static_method("one", "two")
    assert result == "one-two"

    events = processor.get_events()
    assert len(events) == 2

    # Check that all arguments are included
    assert events[0]["attributes"]["args"] == ("one", "two")
    assert events[1]["attributes"]["args"] == ("one", "two")


# Tests for new parameters
def test_exclude_response() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_response=True, exclude_start=False)
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2

    # Check that response is excluded
    end_event = events[1]
    assert "response" not in end_event["attributes"]


def test_include_response() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_response=False, exclude_start=False)
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2

    # Check that response is included
    end_event = events[1]
    assert "response" in end_event["attributes"]
    assert end_event["attributes"]["response"] == 10


def test_exclude_exception() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_exception=True, exclude_start=False)
    def test_func(x: int) -> int:  # noqa: ARG001
        error_msg = "Test error"
        raise ValueError(error_msg)

    with pytest.raises(ValueError):
        test_func(5)

    events = processor.get_events()
    assert len(events) == 2

    # Check that exception is excluded
    end_event = events[1]
    assert "exception" not in end_event["attributes"]


def test_include_exception() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_exception=False, exclude_start=False)
    def test_func(x: int) -> int:  # noqa: ARG001
        error_msg = "Test error"
        raise ValueError(error_msg)

    with pytest.raises(ValueError):
        test_func(5)

    events = processor.get_events()
    assert len(events) == 2

    # Check that exception is included
    end_event = events[1]
    assert "exception" in end_event["attributes"]
    assert end_event["attributes"]["exception"]["type"] == "ValueError"
    assert end_event["attributes"]["exception"]["message"] == "Test error"


def test_exclude_start() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_start=True)
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 1

    # Check that only the end event is recorded
    end_event = events[0]
    assert end_event["name"] == "Function called"


def test_include_start() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(exclude_start=False)
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10

    events = processor.get_events()
    assert len(events) == 2

    # Check that both start and end events are recorded
    start_event = events[0]
    end_event = events[1]
    assert start_event["name"] == "Function call started"
    assert end_event["name"] == "Function called"


def test_all_new_parameters_together() -> None:
    settings = TelemetrySettings(enabled=True)
    telemetry = Telemetry(settings)
    processor = InMemoryProcessor()
    telemetry.add_processor(processor)

    @telemetry.record_call(
        exclude_response=False, exclude_exception=False, exclude_start=False
    )
    def test_func(x: int) -> int:
        if x < 0:
            error_msg = "Negative input"
            raise ValueError(error_msg)
        return x * 2

    # Successful case
    result = test_func(5)
    assert result == 10

    # Error case
    with pytest.raises(ValueError):
        test_func(-5)

    events = processor.get_events()
    assert len(events) == 4

    # First pair of events (successful call)
    assert events[0]["name"] == "Function call started"
    assert events[1]["name"] == "Function called"
    assert "response" in events[1]["attributes"]
    assert events[1]["attributes"]["response"] == 10

    # Second pair of events (error call)
    assert events[2]["name"] == "Function call started"
    assert events[3]["name"] == "Function called"
    assert "exception" in events[3]["attributes"]
    assert events[3]["attributes"]["exception"]["type"] == "ValueError"
    assert events[3]["attributes"]["exception"]["message"] == "Negative input"
