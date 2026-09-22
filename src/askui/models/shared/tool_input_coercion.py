"""Coercion of model-provided tool inputs against a tool's JSON input schema.

Models occasionally emit tool arguments with the wrong JSON type, most commonly
integers as strings (`{"x": "726"}` instead of `{"x": 726}`). Passing such values
straight into a tool implementation typically fails with an opaque `TypeError`
(e.g. `'<' not supported between instances of 'str' and 'int'`). The functions in
this module coerce scalar values to the type declared in the tool's
`input_schema` wherever that is possible without ambiguity, and leave everything
else untouched so the tool's own validation still applies.
"""

import logging
from typing import Any

import jsonref

logger = logging.getLogger(__name__)

_TRUE_STRINGS = frozenset({"true", "1", "yes"})
_FALSE_STRINGS = frozenset({"false", "0", "no"})


def coerce_tool_input(
    tool_input: dict[str, Any], input_schema: dict[str, Any]
) -> dict[str, Any]:
    """Coerce the values of `tool_input` to the types declared in `input_schema`.

    Only lossless, unambiguous conversions are performed:

    - `"42"` -> `42` for `integer`
    - `3.0` -> `3` for `integer`
    - `"3.5"` -> `3.5` for `number`
    - `"true"` / `"false"` (and `1` / `0`) -> `bool` for `boolean`
    - `42` -> `"42"` for `string`

    Values that cannot be converted, keys that are not described by the schema,
    and schemas without a recognizable scalar `type` are returned unchanged. The
    input dictionary is never mutated.

    Args:
        tool_input (dict[str, Any]): The arguments the model passed to the tool.
        input_schema (dict[str, Any]): The tool's JSON schema (`type: object`).
            `$ref`s are resolved before coercion.

    Returns:
        dict[str, Any]: A new dictionary with coerced values.
    """
    try:
        resolved_schema = jsonref.replace_refs(
            input_schema, lazy_load=False, proxies=False
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve refs in tool input schema", exc_info=True)
        resolved_schema = input_schema
    coerced = _coerce_value(tool_input, resolved_schema)
    return coerced if isinstance(coerced, dict) else tool_input


def _coerce_value(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    declared_types = _declared_types(schema)

    if isinstance(value, dict) and "object" in declared_types:
        return _coerce_object(value, schema)
    if isinstance(value, list) and "array" in declared_types:
        return _coerce_array(value, schema)
    if isinstance(value, (dict, list)):
        return value
    return _coerce_scalar(value, declared_types)


def _coerce_object(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return value
    return {
        key: _coerce_value(item, properties.get(key)) for key, item in value.items()
    }


def _coerce_array(value: list[Any], schema: dict[str, Any]) -> list[Any]:
    items_schema = schema.get("items")
    if not isinstance(items_schema, dict):
        return value
    return [_coerce_value(item, items_schema) for item in value]


def _declared_types(schema: dict[str, Any]) -> set[str]:
    """Collect the JSON types a schema accepts, including `anyOf`/`oneOf` variants."""
    declared: set[str] = set()
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        declared.add(schema_type)
    elif isinstance(schema_type, list):
        declared.update(t for t in schema_type if isinstance(t, str))
    for combinator in ("anyOf", "oneOf"):
        variants = schema.get(combinator)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    declared.update(_declared_types(variant))
    return declared


def _coerce_scalar(value: Any, declared_types: set[str]) -> Any:
    if value is None or not declared_types:
        return value
    if _matches_declared_type(value, declared_types):
        return value
    if "integer" in declared_types:
        as_int = _to_int(value)
        if as_int is not None:
            return as_int
    if "number" in declared_types:
        as_float = _to_float(value)
        if as_float is not None:
            return as_float
    if "boolean" in declared_types:
        as_bool = _to_bool(value)
        if as_bool is not None:
            return as_bool
    if "string" in declared_types and isinstance(value, (int, float)):
        return str(value)
    return value


def _matches_declared_type(value: Any, declared_types: set[str]) -> bool:
    if isinstance(value, bool):
        return "boolean" in declared_types
    if isinstance(value, int):
        return "integer" in declared_types or "number" in declared_types
    if isinstance(value, float):
        return "number" in declared_types
    if isinstance(value, str):
        return "string" in declared_types
    return True


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            as_float = float(stripped)
        except ValueError:
            return None
        return int(as_float) if as_float.is_integer() else None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return None
