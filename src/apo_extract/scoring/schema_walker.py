"""Recursive JSON Schema traversal binding gold/extracted values onto AST nodes.

Reuses extract-bench's schema AST + evaluation_config resolution (compliance by
construction) but binds values without raising on missing required keys, so bad
predictions score as omissions instead of crashing the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from extract_bench.evaluation.schema_config_helpers import get_evaluation_config
from extract_bench.evaluation.schema_value_instantiator import MISSING
from extract_bench.evaluation.structured_evaluator import _resolve_evaluation_configs
from extract_bench.infra.construct_ast import construct_ast
from extract_bench.infra.nodes import (
    AnyOfSchema,
    ArraySchema,
    ObjectSchema,
    RootSchema,
    Schema,
)
from extract_bench.infra.ref_expander import expand_refs


@dataclass
class WalkNode:
    path: str
    node: Schema
    gold: Any
    extracted: Any
    is_array_item: bool = False

    @property
    def gold_absent(self) -> bool:
        if self.gold is MISSING or self.gold is None:
            return True
        if isinstance(self.node, ArraySchema) and self.gold == []:
            return True
        return False

    @property
    def extracted_absent(self) -> bool:
        if self.extracted is MISSING or self.extracted is None:
            return True
        if isinstance(self.node, ArraySchema) and self.extracted == []:
            return True
        return False


def build_schema_tree(schema_dict: dict) -> Schema:
    tree = expand_refs(construct_ast(schema_dict))
    tree = _resolve_evaluation_configs(tree)
    return tree


class SchemaWalker:
    """Walks a resolved schema AST in lockstep with gold/extracted values."""

    def __init__(self, schema_dict: dict | None = None, tree: Schema | None = None):
        if tree is None:
            if schema_dict is None:
                raise ValueError("SchemaWalker needs either schema_dict or tree")
            tree = build_schema_tree(schema_dict)
        self.tree = tree

    def walk(self, gold: Any, extracted: Any) -> Iterator[WalkNode]:
        yield from self._walk(self.tree, "$", gold, extracted)

    def walk_from(self, node: Schema, path: str, gold: Any, extracted: Any) -> Iterator[WalkNode]:
        yield from self._walk(node, path, gold, extracted)

    def _walk(self, node: Schema, path: str, gold: Any, extracted: Any) -> Iterator[WalkNode]:
        nullable = getattr(node, "nullable", False)
        if nullable and (gold is None or extracted is None):
            node.set_gold_value(gold)
            node.set_extracted_value(extracted)
            yield WalkNode(path, node, gold, extracted)
            return

        if isinstance(node, (ObjectSchema, RootSchema)):
            node.set_gold_value(gold)
            node.set_extracted_value(extracted)
            yield WalkNode(path, node, gold, extracted)
            properties = getattr(node, "properties", None) or {}
            for key, child in properties.items():
                gold_child = gold.get(key, MISSING) if isinstance(gold, dict) else MISSING
                extracted_child = (
                    extracted.get(key, MISSING) if isinstance(extracted, dict) else MISSING
                )
                yield from self._walk(child, f"{path}.{key}", gold_child, extracted_child)
            return

        if isinstance(node, ArraySchema):
            node.set_gold_value(gold)
            node.set_extracted_value(extracted)
            yield WalkNode(path, node, gold, extracted)
            return

        if isinstance(node, AnyOfSchema):
            node.set_gold_value(gold)
            node.set_extracted_value(extracted)
            yield WalkNode(path, node, gold, extracted)
            return

        node.set_gold_value(gold)
        node.set_extracted_value(extracted)
        yield WalkNode(path, node, gold, extracted)

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def type_matches(value: Any, type_name: str) -> bool:
    if value is MISSING or value is None:
        return type_name == "null"
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _TYPE_CHECKS.get(type_name, ()))


def anyof_branch_specs(node: Schema) -> list[dict]:
    """Per-branch metric specs for an anyOf node without its own config.

    Implements the inheritance rule: a config-less anyOf inherits the
    evaluation_config of the non-null child branch that matches the value type.
    """
    branches = []
    for child in getattr(node, "any_of", []):
        child_type = child.get_type()
        if child_type == "null":
            continue
        config = get_evaluation_config(child)
        if not config.metrics:
            continue
        primary = config.metrics[0]
        branches.append(
            {
                "type": child_type,
                "metric_id": primary.metric_id,
                "params": dict(primary.params) if primary.params else None,
            }
        )
    return branches


def resolve_anyof_branch(branches: list[dict], gold: Any, pred: Any) -> dict | None:
    if not branches:
        return None
    for value in (gold, pred):
        for branch in branches:
            if type_matches(value, branch["type"]):
                return branch
    return branches[0]


def item_leaf_specs(
    array_node: Schema, base_path: str
) -> list[tuple[str, str, dict | None, list[str]]]:
    """Enumerate scoreable leaves under an array's items schema.

    Returns ``(leaf_path, metric_id, params, relative_keys)`` where
    ``relative_keys`` is the key chain into a single item value (``"*"`` for
    nested array indices).
    """
    items = getattr(array_node, "items", None)
    if items is None:
        return []
    specs: list[tuple[str, str, dict | None, list[str]]] = []

    def emit(node: Schema, keys: list[str]) -> None:
        config = get_evaluation_config(node)
        for metric_config in config.metrics:
            params = dict(metric_config.params) if metric_config.params else None
            joined = ".".join(keys) or "$"
            specs.append(
                (f"{base_path}[*].{joined}", metric_config.metric_id, params, list(keys))
            )
        if not config.metrics and isinstance(node, AnyOfSchema):
            branches = anyof_branch_specs(node)
            if branches:
                joined = ".".join(keys) or "$"
                specs.append(
                    (
                        f"{base_path}[*].{joined}",
                        "anyof_typed",
                        {"branches": branches},
                        list(keys),
                    )
                )

    def visit(node: Schema, keys: list[str]) -> None:
        if isinstance(node, ObjectSchema):
            for key, child in (node.properties or {}).items():
                visit(child, keys + [key])
            return
        if isinstance(node, ArraySchema):
            if node.items is not None:
                visit(node.items, keys + ["*"])
            return
        emit(node, keys)

    visit(items, [])
    return specs
