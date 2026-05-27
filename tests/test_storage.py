from __future__ import annotations

from pathlib import Path

import pytest

from cmg._schema import RECORD_NODE, RECORD_VIOLATION, SCHEMA_VERSION
from cmg.errors import MalformedLogLineError
from cmg.nodes import Support, Violation, mint_id, now_iso
from cmg.storage import JsonlStorage, Storage


async def test_jsonl_round_trip(tmp_path: Path) -> None:
    storage = JsonlStorage(tmp_path / "log.jsonl")
    node = Support(node_id=mint_id("support"), content="x", created_at=now_iso())
    violation = Violation(
        node_id=node.node_id, code="empty_field", detail={"field": "content"}, created_at=now_iso()
    )
    await storage.append_node(node)
    await storage.append_violation(violation)
    await storage.aclose()

    records = list(storage.iter_records())
    assert len(records) == 2
    assert records[0]["record"] == RECORD_NODE
    assert records[0]["cmg_schema_version"] == SCHEMA_VERSION
    assert records[0]["node_id"] == node.node_id
    assert records[1]["record"] == RECORD_VIOLATION
    assert records[1]["code"] == "empty_field"


async def test_jsonl_accepts_string_path_and_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    async with JsonlStorage(str(path)) as storage:
        node = Support(node_id=mint_id("support"), content="x", created_at=now_iso())
        await storage.append_node(node)
        assert storage._writer is not None
    assert storage._writer is None
    assert path.exists()


def test_protocol_conformance() -> None:
    assert isinstance(JsonlStorage(Path("/tmp/_unused.jsonl")), Storage)


async def test_iter_records_empty_file(tmp_path: Path) -> None:
    storage = JsonlStorage(tmp_path / "missing.jsonl")
    assert list(storage.iter_records()) == []


def test_iter_records_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(MalformedLogLineError):
        list(JsonlStorage(path).iter_records())
