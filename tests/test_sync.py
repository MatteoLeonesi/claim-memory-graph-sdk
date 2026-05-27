from __future__ import annotations

from pathlib import Path

from cmg import JsonlStorage
from cmg.sync import ClaimGraphSync


def test_sync_basic_flow(tmp_path: Path) -> None:
    graph = ClaimGraphSync(JsonlStorage(tmp_path / "log.jsonl"))
    try:
        s = graph.add_support("loop terminates").node
        c = graph.add_commitment("bug real", refs=(s.node_id,)).node
        d = graph.add_decision("reject", refs=(c.node_id,))
        assert d.violations == ()
        assert graph.active_commitments() == (c,)
    finally:
        graph.close()


def test_sync_context_manager_and_idempotent_close(tmp_path: Path) -> None:
    with ClaimGraphSync(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        s = graph.add_support("loop terminates").node
        c = graph.add_commitment("bug real", refs=(s.node_id,)).node
        assert graph.active_commitment_ids() == frozenset({c.node_id})
        assert graph.violations_for(c.node_id) == ()
    graph.close()


def test_sync_load_replays(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    graph = ClaimGraphSync(JsonlStorage(path))
    try:
        s = graph.add_support("ev").node
        graph.add_commitment("c", refs=(s.node_id,))
    finally:
        graph.close()

    reloaded = ClaimGraphSync.load(JsonlStorage(path))
    try:
        assert len(reloaded.nodes()) == 2
    finally:
        reloaded.close()


def test_sync_run_turn(tmp_path: Path) -> None:
    graph = ClaimGraphSync(JsonlStorage(tmp_path / "log.jsonl"))
    try:
        s = graph.add_support("ev").node

        async def llm(_messages: list[dict[str, str]]) -> str:
            return (
                "ok\n"
                "```cmg\n"
                f'{{"ops": [{{"op": "commitment", "content": "x", "refs": ["{s.node_id}"]}}]}}\n'
                "```"
            )

        result = graph.run_turn(llm, [])
        assert len(result.applied) == 1
        assert result.applied[0].node.kind == "commitment"
    finally:
        graph.close()
