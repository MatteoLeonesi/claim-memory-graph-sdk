from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cmg import ClaimGraph, JsonlStorage
from cmg.errors import DuplicateNodeIdError
from cmg.nodes import INVALIDATION_ACCEPTED


async def _fresh(tmp_path: Path) -> ClaimGraph:
    return ClaimGraph(JsonlStorage(tmp_path / "log.jsonl"))


async def test_basic_flow_no_violations(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("loop terminates early")).node
    c = (await graph.add_commitment("the bug is real", refs=(s.node_id,))).node
    d = await graph.add_decision("reject", refs=(c.node_id,))
    assert d.violations == ()
    assert graph.active_commitments() == (c,)
    assert graph.last_decision() == d.node
    await graph.aclose()


async def test_graph_async_context_manager_closes_storage(tmp_path: Path) -> None:
    storage = JsonlStorage(tmp_path / "log.jsonl")
    async with ClaimGraph(storage) as graph:
        await graph.add_support("e")
        assert storage._writer is not None
    assert storage._writer is None


async def test_verdict_flip_observed_but_appended(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("e")).node
    c = (await graph.add_commitment("real", refs=(s.node_id,))).node
    await graph.add_decision("approve", refs=(c.node_id,))
    flipped = await graph.add_decision("reject", refs=(c.node_id,))
    codes = [v.code for v in flipped.violations]
    assert "verdict_flip_without_invalidation" in codes
    assert flipped.node in graph.nodes()
    await graph.aclose()


async def test_incremental_active_set(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("e")).node
    c = (await graph.add_commitment("c", refs=(s.node_id,))).node
    assert c.node_id in graph.active_commitment_ids()
    await graph.add_invalidation(
        previous_commitment=c.node_id,
        previous_support=(s.node_id,),
        new_information="counter-evidence",
        contrast_test="distinguishing test",
        result=INVALIDATION_ACCEPTED,
    )
    assert c.node_id not in graph.active_commitment_ids()
    await graph.aclose()


async def test_active_set_perf_amortized(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("e")).node
    for _ in range(100):
        await graph.add_commitment("c", refs=(s.node_id,))
    assert len(graph.active_commitments()) == 100
    await graph.aclose()


async def test_concurrent_appends_unique_ids(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("e")).node
    results = await asyncio.gather(
        *[graph.add_commitment(f"claim {i}", refs=(s.node_id,)) for i in range(10)]
    )
    ids = {r.node.node_id for r in results}
    assert len(ids) == 10
    assert len(graph.nodes()) == 11
    await graph.aclose()


async def test_replay_reproduces_state(tmp_path: Path) -> None:
    storage = JsonlStorage(tmp_path / "log.jsonl")
    graph = ClaimGraph(storage)
    s = (await graph.add_support("e")).node
    c = (await graph.add_commitment("c", refs=(s.node_id,))).node
    await graph.add_decision("approve", refs=(c.node_id,))
    await graph.aclose()

    replayed = await ClaimGraph.aload(JsonlStorage(tmp_path / "log.jsonl"))
    assert [n.node_id for n in replayed.nodes()] == [s.node_id, c.node_id, replayed.nodes()[2].node_id]
    assert replayed.active_commitment_ids() == frozenset({c.node_id})


async def test_replay_detects_external_tampering(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    graph = ClaimGraph(JsonlStorage(path))
    s = (await graph.add_support("e")).node
    await graph.add_commitment("c", refs=(s.node_id,))
    await graph.aclose()

    import json

    bogus = {
        "record": "node",
        "cmg_schema_version": 1,
        "kind": "decision",
        "node_id": "d-deadbeef",
        "content": "reject",
        "created_at": "2026-01-01T00:00:00+00:00",
        "refs": ["k-nonexistent"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(bogus) + "\n")

    replayed = await ClaimGraph.aload(JsonlStorage(path))
    codes = [v.code for v in replayed.violations()]
    assert "replay_mismatch" in codes


async def test_redaction_hook_applied(tmp_path: Path) -> None:
    graph = ClaimGraph(JsonlStorage(tmp_path / "log.jsonl"), redact_fn=lambda s: s.replace("secret", "[REDACTED]"))
    result = await graph.add_support("contains secret token")
    assert "secret" not in result.node.content
    assert "[REDACTED]" in result.node.content
    await graph.aclose()


async def test_duplicate_node_id_on_replay_raises(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    import json

    rec = {
        "record": "node",
        "cmg_schema_version": 1,
        "kind": "support",
        "node_id": "s-deadbeef",
        "content": "x",
        "created_at": "2026-01-01T00:00:00+00:00",
        "refs": [],
        "source_turn": 0,
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(rec) + "\n")
    with pytest.raises(DuplicateNodeIdError):
        await ClaimGraph.aload(JsonlStorage(path))
