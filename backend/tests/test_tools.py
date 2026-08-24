"""Tool declaration and validation tests. No database, no network."""

from __future__ import annotations

from app.llm.schema import flatten_schema
from app.llm.tools import TOOLS, declarations

FORBIDDEN = {"$defs", "$ref", "additionalProperties", "title", "anyOf", "default"}


def _keys(node) -> set[str]:
    found = set()
    if isinstance(node, dict):
        found |= set(node)
        for value in node.values():
            found |= _keys(value)
    elif isinstance(node, list):
        for item in node:
            found |= _keys(item)
    return found


def test_no_schema_contains_keys_gemini_rejects():
    """The gotcha that costs an evening: Pydantic emits $defs/$ref for nested
    models and the API rejects the request outright."""
    for name, (model, _, _) in TOOLS.items():
        flat = flatten_schema(model.model_json_schema())
        offending = _keys(flat) & FORBIDDEN
        assert not offending, f"{name} schema still contains {offending}"


def test_optional_field_becomes_a_bare_type():
    """`str | None` renders as anyOf[string, null]; Gemini wants 'string'."""
    from app.llm.tools import SpecialtyArgs

    flat = flatten_schema(SpecialtyArgs.model_json_schema())
    assert flat["properties"]["disease_slug"]["type"] == "string"


def test_every_tool_declares_a_description():
    for decl in declarations():
        assert decl.description and len(decl.description) > 20


def test_every_property_is_described():
    """Undescribed parameters are the main cause of the model guessing."""
    for name, (model, _, _) in TOOLS.items():
        flat = flatten_schema(model.model_json_schema())
        for prop, spec in flat.get("properties", {}).items():
            assert spec.get("description"), f"{name}.{prop} has no description"


def test_declaration_count_matches_registry():
    assert len(declarations()) == len(TOOLS)


def test_required_fields_are_marked():
    from app.llm.tools import DiagnoseArgs

    flat = flatten_schema(DiagnoseArgs.model_json_schema())
    assert flat["required"] == ["symptom_slugs"]
