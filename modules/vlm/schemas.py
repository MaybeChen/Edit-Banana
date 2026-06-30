"""Schemas and lightweight validation for VLM JSON responses."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

JsonSchema = Mapping[str, Any]

ELEMENT_CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["element_type", "confidence"],
    "properties": {
        "element_type": {"type": "string"},
        "line_style": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}

PROMPT_GROUPS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["image", "shape", "arrow", "background"],
    "properties": {
        "image": {"type": "array", "items": {"type": "string"}},
        "shape": {"type": "array", "items": {"type": "string"}},
        "arrow": {"type": "array", "items": {"type": "string"}},
        "background": {"type": "array", "items": {"type": "string"}},
    },
}

LAYOUT_RELATIONS_SCHEMA: Dict[str, Any] = {"type": "object"}
EXPORT_VALIDATION_SCHEMA: Dict[str, Any] = {"type": "object"}
REGION_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["confidence", "elements"],
    "properties": {
        "confidence": {"type": "number"},
        "elements": {"type": "array", "items": {"type": "object"}},
    },
}


def validate_json_schema(data: Any, schema: Optional[JsonSchema]) -> Dict[str, Any]:
    """Validate a JSON object against the subset of JSON Schema used here."""
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    if not schema:
        return data
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(data, dict):
        raise ValueError("VLM response schema requires an object")
    for key in schema.get("required", []) or []:
        if key not in data:
            raise ValueError(f"VLM response missing required key: {key}")
    properties = schema.get("properties", {}) or {}
    for key, value in data.items():
        prop = properties.get(key)
        if not prop:
            continue
        _validate_value(key, value, prop)
    return data


def _validate_value(key: str, value: Any, schema: Mapping[str, Any]) -> None:
    allowed = schema.get("type")
    if isinstance(allowed, str):
        allowed = [allowed]
    if allowed and not any(_matches_type(value, item) for item in allowed):
        raise ValueError(f"VLM response key '{key}' has invalid type")
    if isinstance(value, list) and "items" in schema:
        for item in value:
            _validate_value(f"{key}[]", item, schema["items"])


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    return True
