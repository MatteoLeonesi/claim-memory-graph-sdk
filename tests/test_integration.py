from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from cmg import (
    ClaimGraph,
    JsonlStorage,
    Message,
    StreamChunk,
    arun_turn,
    astream_turn,
    build_annotation_system_prompt,
)


async def _fresh(tmp_path: Path) -> ClaimGraph:
    return ClaimGraph(JsonlStorage(tmp_path / "log.jsonl"))


async def test_arun_turn_with_no_annotations(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)

    async def llm(_messages: list[Message]) -> str:
        return "Looks good to me."

    result = await arun_turn(graph, llm, [{"role": "user", "content": "review"}])
    assert result.visible_text == "Looks good to me."
    assert result.applied == ()
    assert result.violations() == ()
    await graph.aclose()


async def test_arun_turn_applies_ops(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    support = (await graph.add_support("loop terminates early")).node

    response = (
        f"The bug is real.\n"
        f"```cmg\n"
        f'{{"ops": [{{"op": "commitment", "content": "bug is real", "refs": ["{support.node_id}"]}}]}}\n'
        f"```"
    )

    async def llm(_messages: list[Message]) -> str:
        return response

    result = await arun_turn(graph, llm, [])
    assert len(result.applied) == 1
    assert result.applied[0].node.kind == "commitment"
    assert result.applied[0].violations == ()
    assert "```cmg" not in result.visible_text
    await graph.aclose()


async def test_arun_turn_surfaces_violations(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    s = (await graph.add_support("e")).node
    c = (await graph.add_commitment("real", refs=(s.node_id,))).node
    await graph.add_decision("approve", refs=(c.node_id,))

    response = (
        f"```cmg\n"
        f'{{"ops": [{{"op": "decision", "content": "reject", "refs": ["{c.node_id}"]}}]}}\n'
        f"```"
    )

    async def llm(_messages: list[Message]) -> str:
        return response

    captured: list[str] = []
    result = await arun_turn(
        graph,
        llm,
        [],
        on_violation=lambda v: captured.append(v.code),
    )
    assert any(v.code == "verdict_flip_without_invalidation" for v in result.violations())
    assert "verdict_flip_without_invalidation" in captured
    await graph.aclose()


async def test_arun_turn_unknown_op_warns(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)

    async def llm(_messages: list[Message]) -> str:
        return '```cmg\n{"ops": [{"op": "thinking", "content": "hmm"}]}\n```'

    result = await arun_turn(graph, llm, [])
    assert result.applied == ()
    assert any("unknown op" in w for w in result.parse_warnings)
    await graph.aclose()


async def test_astream_turn_chunks_and_final(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)
    support = (await graph.add_support("ev")).node

    parts = [
        "Looks like ",
        "a bug.\n",
        "```cmg\n",
        f'{{"ops": [{{"op": "commitment", "content": "bug", "refs": ["{support.node_id}"]}}]}}\n',
        "```\n",
        "Verdict pending.",
    ]

    async def stream(_messages: list[Message]) -> AsyncIterator[str]:
        for p in parts:
            yield p

    chunks: list[StreamChunk] = []
    async for chunk in astream_turn(graph, stream, []):
        chunks.append(chunk)

    assert chunks[-1].is_final is True
    assert chunks[-1].result is not None
    assert len(chunks[-1].result.applied) == 1
    visible_concat = "".join(c.visible_text_delta for c in chunks if not c.is_final)
    assert "```cmg" not in visible_concat
    assert "Looks like a bug" in visible_concat
    assert "Verdict pending" in visible_concat
    await graph.aclose()


def test_build_annotation_system_prompt() -> None:
    prompt = build_annotation_system_prompt()
    assert "```cmg" in prompt
    assert "<cmg" in prompt


async def test_astream_buffer_runaway_is_force_flushed(tmp_path: Path) -> None:
    from cmg.integration import MAX_STREAM_BUFFER_BYTES

    graph = await _fresh(tmp_path)

    async def runaway(_messages: list[Message]) -> AsyncIterator[str]:
        yield "```cmg\n"
        yield "x" * (MAX_STREAM_BUFFER_BYTES + 1024)

    visible_total: list[str] = []
    async for chunk in astream_turn(graph, runaway, []):
        if not chunk.is_final:
            visible_total.append(chunk.visible_text_delta)
    assert sum(len(v) for v in visible_total) >= MAX_STREAM_BUFFER_BYTES
    await graph.aclose()


async def test_arun_rejects_bool_as_int(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)

    async def llm(_messages: list[Message]) -> str:
        return '```cmg\n{"ops": [{"op": "support", "content": "x", "source_turn": true}]}\n```'

    result = await arun_turn(graph, llm, [])
    assert result.applied == ()
    assert any("must be an int" in w for w in result.parse_warnings)
    await graph.aclose()


async def test_arun_rejects_non_string_refs(tmp_path: Path) -> None:
    graph = await _fresh(tmp_path)

    async def llm(_messages: list[Message]) -> str:
        return '```cmg\n{"ops": [{"op": "commitment", "content": "x", "refs": [123]}]}\n```'

    result = await arun_turn(graph, llm, [])
    assert result.applied == ()
    assert any("entries must be strings" in w for w in result.parse_warnings)
    await graph.aclose()
