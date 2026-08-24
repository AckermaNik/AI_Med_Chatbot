"""Convert Pydantic schemas into the subset Gemini accepts.

Pydantic's model_json_schema() emits `$defs` and `$ref` for nested models, plus
keys like `additionalProperties`, `title` and `anyOf`, which the Gemini API
rejects outright. This flattens refs inline and strips what it will not take, so
one Pydantic model can serve as BOTH the argument validator and the declared
schema — no second, drifting copy of each tool signature.
"""

from __future__ import annotations

from typing import Any

# Keys Gemini's schema subset does not accept.
STRIP = {"additionalProperties", "title", "$schema", "definitions", "$defs", "default"}


def flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {}) or schema.get("definitions", {})
    return _walk(schema, defs)


def _walk(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_walk(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        target = defs.get(name, {})
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return _walk(merged, defs)

    # Optional fields become anyOf[T, null]; Gemini wants the bare type.
    if "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(options) == 1:
            merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **options[0]}
            return _walk(merged, defs)

    return {k: _walk(v, defs) for k, v in node.items() if k not in STRIP}
