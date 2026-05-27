from __future__ import annotations

from cmg.checks import GraphState, check_node
from cmg.nodes import (
    INVALIDATION_ACCEPTED,
    INVALIDATION_REJECTED,
    Commitment,
    Decision,
    Invalidation,
    Node,
    Support,
    mint_id,
    now_iso,
)


def _state(*nodes: Node, last_decision: Decision | None = None) -> GraphState:
    by_id = {n.node_id: n for n in nodes}
    retired: set[str] = {
        n.previous_commitment
        for n in nodes
        if isinstance(n, Invalidation) and n.result == INVALIDATION_ACCEPTED
    }
    active = frozenset(
        n.node_id for n in nodes if isinstance(n, Commitment) and n.node_id not in retired
    )
    return GraphState(by_id, active, last_decision)


def _support(content: str = "evidence") -> Support:
    return Support(node_id=mint_id("support"), content=content, created_at=now_iso())


def _commitment(refs: tuple[str, ...], content: str = "claim") -> Commitment:
    return Commitment(node_id=mint_id("commitment"), content=content, created_at=now_iso(), refs=refs)


def _decision(refs: tuple[str, ...], content: str = "approve") -> Decision:
    return Decision(node_id=mint_id("decision"), content=content, created_at=now_iso(), refs=refs)


def _invalidation(
    commitment: str,
    support: tuple[str, ...],
    result: str = INVALIDATION_ACCEPTED,
    new_information: str = "new evidence",
    contrast_test: str = "test",
) -> Invalidation:
    return Invalidation(
        node_id=mint_id("invalidation"),
        created_at=now_iso(),
        previous_commitment=commitment,
        previous_support=support,
        new_information=new_information,
        contrast_test=contrast_test,
        result=result,
    )


def _codes(node: Node, state: GraphState) -> list[str]:
    return [v.code for v in check_node(node, state)]


def test_legal_support_has_no_violations() -> None:
    assert _codes(_support(), _state()) == []


def test_id_kind_mismatch() -> None:
    bad = Support(node_id="k-deadbeef", content="x", created_at=now_iso())
    assert "id_kind_mismatch" in _codes(bad, _state())


def test_empty_field_on_support() -> None:
    assert "empty_field" in _codes(_support(content="   "), _state())


def test_empty_field_on_invalidation() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    inv = _invalidation(c.node_id, (s.node_id,), new_information="  ")
    assert "empty_field" in _codes(inv, _state(s, c))


def test_unknown_ref_on_commitment() -> None:
    c = _commitment(refs=("s-deadbeef",))
    assert "unknown_ref" in _codes(c, _state())


def test_wrong_ref_kind_on_commitment() -> None:
    d = _decision(refs=())
    c = _commitment(refs=(d.node_id,))
    assert "wrong_ref_kind" in _codes(c, _state(d))


def test_ref_not_active_on_decision() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    inv = _invalidation(c.node_id, (s.node_id,))
    d = _decision(refs=(c.node_id,))
    assert "ref_not_active" in _codes(d, _state(s, c, inv))


def test_verdict_flip_without_invalidation() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    prev = _decision(refs=(c.node_id,), content="approve")
    new = _decision(refs=(c.node_id,), content="reject")
    assert "verdict_flip_without_invalidation" in _codes(new, _state(s, c, last_decision=prev))


def test_verdict_flip_after_invalidation_is_clean() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    prev = _decision(refs=(c.node_id,), content="approve")
    inv = _invalidation(c.node_id, (s.node_id,))
    new_c = _commitment(refs=(s.node_id,), content="counter-claim")
    state = _state(s, c, inv, new_c, last_decision=prev)
    new = _decision(refs=(new_c.node_id,), content="reject")
    codes = _codes(new, state)
    assert "verdict_flip_without_invalidation" not in codes
    assert "silent_commitment_drop" not in codes


def test_silent_commitment_drop() -> None:
    s = _support()
    c1 = _commitment(refs=(s.node_id,))
    c2 = _commitment(refs=(s.node_id,), content="other claim")
    prev = _decision(refs=(c1.node_id, c2.node_id), content="approve")
    new = _decision(refs=(c1.node_id,), content="approve")
    assert "silent_commitment_drop" in _codes(new, _state(s, c1, c2, last_decision=prev))


def test_invalidation_target_inactive() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    first = _invalidation(c.node_id, (s.node_id,))
    second = _invalidation(c.node_id, (s.node_id,))
    assert "invalidation_target_inactive" in _codes(second, _state(s, c, first))


def test_invalidation_support_not_cited() -> None:
    s_cited = _support()
    s_other = _support(content="unrelated")
    c = _commitment(refs=(s_cited.node_id,))
    inv = _invalidation(c.node_id, (s_other.node_id,))
    assert "invalidation_support_not_cited" in _codes(inv, _state(s_cited, s_other, c))


def test_invalidation_result_unknown() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    inv = _invalidation(c.node_id, (s.node_id,), result="maybe")
    assert "invalidation_result_unknown" in _codes(inv, _state(s, c))


def test_invalidation_rejected_keeps_commitment_active() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    rejected = _invalidation(c.node_id, (s.node_id,), result=INVALIDATION_REJECTED)
    state = _state(s, c, rejected)
    assert c.node_id in state.active_commitment_ids
    d = _decision(refs=(c.node_id,))
    assert "ref_not_active" not in _codes(d, state)


def test_empty_refs_on_commitment() -> None:
    c = _commitment(refs=())
    assert "empty_refs" in _codes(c, _state())


def test_empty_refs_on_decision() -> None:
    d = _decision(refs=())
    assert "empty_refs" in _codes(d, _state())


def test_empty_refs_on_invalidation() -> None:
    s = _support()
    c = _commitment(refs=(s.node_id,))
    inv = _invalidation(c.node_id, ())
    assert "empty_refs" in _codes(inv, _state(s, c))
