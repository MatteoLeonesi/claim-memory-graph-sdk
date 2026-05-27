from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cmg.nodes import (
    INVALIDATION_RESULTS,
    KIND_TO_PREFIX,
    PREFIX_COMMITMENT,
    PREFIX_SUPPORT,
    Commitment,
    Decision,
    Invalidation,
    Node,
    Support,
    Violation,
    now_iso,
)


@dataclass(frozen=True, slots=True)
class GraphState:
    nodes_by_id: Mapping[str, Node]
    active_commitment_ids: frozenset[str]
    last_decision: Decision | None


def _v(node_id: str, code: str, **detail: object) -> Violation:
    return Violation(node_id=node_id, code=code, detail=detail, created_at=now_iso())


def _id_kind_check(node: Node) -> tuple[Violation, ...]:
    expected = KIND_TO_PREFIX[node.kind]
    if not node.node_id.startswith(expected):
        return (_v(node.node_id, "id_kind_mismatch", kind=node.kind, expected_prefix=expected),)
    return ()


def _check_refs(
    node_id: str,
    refs: tuple[str, ...],
    *,
    expected_prefix: str,
    nodes_by_id: Mapping[str, Node],
    active: frozenset[str] | None = None,
) -> list[Violation]:
    out: list[Violation] = []
    for ref in refs:
        if not ref.startswith(expected_prefix):
            out.append(_v(node_id, "wrong_ref_kind", ref=ref, expected_prefix=expected_prefix))
        elif ref not in nodes_by_id:
            out.append(_v(node_id, "unknown_ref", ref=ref))
        elif active is not None and ref not in active:
            out.append(_v(node_id, "ref_not_active", ref=ref))
    return out


def check_support(node: Support, state: GraphState) -> tuple[Violation, ...]:
    out: list[Violation] = list(_id_kind_check(node))
    if not node.content.strip():
        out.append(_v(node.node_id, "empty_field", field="content"))
    # Support refs are free-form (rarely used). We do not validate their kind/existence
    # because Support evidence may legitimately reference external artifacts.
    return tuple(out)


def check_commitment(node: Commitment, state: GraphState) -> tuple[Violation, ...]:
    out: list[Violation] = list(_id_kind_check(node))
    if not node.content.strip():
        out.append(_v(node.node_id, "empty_field", field="content"))
    if not node.refs:
        out.append(_v(node.node_id, "empty_refs", relation="commitment.refs"))
    out.extend(_check_refs(
        node.node_id, node.refs,
        expected_prefix=PREFIX_SUPPORT, nodes_by_id=state.nodes_by_id,
    ))
    return tuple(out)


def check_decision(node: Decision, state: GraphState) -> tuple[Violation, ...]:
    out: list[Violation] = list(_id_kind_check(node))
    if not node.content.strip():
        out.append(_v(node.node_id, "empty_field", field="content"))
    if not node.refs:
        out.append(_v(node.node_id, "empty_refs", relation="decision.refs"))
    out.extend(_check_refs(
        node.node_id, node.refs,
        expected_prefix=PREFIX_COMMITMENT,
        nodes_by_id=state.nodes_by_id,
        active=state.active_commitment_ids,
    ))
    out.extend(_decision_transition_violations(node, state))
    return tuple(out)


def _decision_transition_violations(
    node: Decision, state: GraphState
) -> tuple[Violation, ...]:
    prev = state.last_decision
    if prev is None:
        return ()
    new_refs = set(node.refs)
    prior_active = {r for r in prev.refs if r in state.active_commitment_ids}
    verdict_changed = node.content.strip().casefold() != prev.content.strip().casefold()
    if verdict_changed and prior_active:
        return (
            _v(
                node.node_id,
                "verdict_flip_without_invalidation",
                prior_active=sorted(prior_active),
                previous_decision=prev.node_id,
            ),
        )
    if not verdict_changed:
        dropped = prior_active - new_refs
        if dropped:
            return (
                _v(
                    node.node_id,
                    "silent_commitment_drop",
                    dropped=sorted(dropped),
                    previous_decision=prev.node_id,
                ),
            )
    return ()


def check_invalidation(node: Invalidation, state: GraphState) -> tuple[Violation, ...]:
    out: list[Violation] = list(_id_kind_check(node))
    if not node.new_information.strip():
        out.append(_v(node.node_id, "empty_field", field="new_information"))
    if not node.contrast_test.strip():
        out.append(_v(node.node_id, "empty_field", field="contrast_test"))
    if node.result not in INVALIDATION_RESULTS:
        out.append(_v(node.node_id, "invalidation_result_unknown", result=node.result))
    if not node.previous_support:
        out.append(_v(node.node_id, "empty_refs", relation="invalidation.previous_support"))

    target = state.nodes_by_id.get(node.previous_commitment)
    if not node.previous_commitment.startswith(PREFIX_COMMITMENT):
        out.append(
            _v(
                node.node_id,
                "wrong_ref_kind",
                ref=node.previous_commitment,
                expected_prefix=PREFIX_COMMITMENT,
            )
        )
    elif target is None:
        out.append(_v(node.node_id, "unknown_ref", ref=node.previous_commitment))
    elif node.previous_commitment not in state.active_commitment_ids:
        out.append(
            _v(node.node_id, "invalidation_target_inactive", ref=node.previous_commitment)
        )

    target_refs: set[str] = set()
    if isinstance(target, Commitment):
        target_refs = set(target.refs)

    for sid in node.previous_support:
        if not sid.startswith(PREFIX_SUPPORT):
            out.append(_v(node.node_id, "wrong_ref_kind", ref=sid, expected_prefix=PREFIX_SUPPORT))
        elif sid not in state.nodes_by_id:
            out.append(_v(node.node_id, "unknown_ref", ref=sid))
        elif target_refs and sid not in target_refs:
            out.append(
                _v(
                    node.node_id,
                    "invalidation_support_not_cited",
                    ref=sid,
                    commitment=node.previous_commitment,
                )
            )
    return tuple(out)


def check_node(node: Node, state: GraphState) -> tuple[Violation, ...]:
    match node:
        case Support():
            return check_support(node, state)
        case Commitment():
            return check_commitment(node, state)
        case Decision():
            return check_decision(node, state)
        case Invalidation():
            return check_invalidation(node, state)


__all__ = [
    "GraphState",
    "check_commitment",
    "check_decision",
    "check_invalidation",
    "check_node",
    "check_support",
]
