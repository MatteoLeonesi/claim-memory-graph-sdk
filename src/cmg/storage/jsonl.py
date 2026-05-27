from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import IO

from cmg._schema import RECORD_NODE, RECORD_VIOLATION, SCHEMA_VERSION
from cmg.errors import MalformedLogLineError
from cmg.nodes import Node, Violation


class JsonlStorage:
    """Append-only JSONL backend. One record per line; UTF-8; schema-versioned."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        self._writer: IO[str] | None = None

    async def __aenter__(self) -> JsonlStorage:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _open_writer(self) -> IO[str]:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self.path.open("a", encoding="utf-8")
        return self._writer

    async def append_node(self, node: Node) -> None:
        payload = asdict(node)
        payload["record"] = RECORD_NODE
        payload["cmg_schema_version"] = SCHEMA_VERSION
        self._write_line(payload)

    async def append_violation(self, violation: Violation) -> None:
        payload = asdict(violation)
        payload["record"] = RECORD_VIOLATION
        payload["cmg_schema_version"] = SCHEMA_VERSION
        self._write_line(payload)

    def _write_line(self, payload: dict[str, object]) -> None:
        writer = self._open_writer()
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        writer.flush()

    def iter_records(self) -> Iterator[dict[str, object]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line_number, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise MalformedLogLineError(f"{self.path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise MalformedLogLineError(
                        f"{self.path}:{line_number}: expected object, got {type(record).__name__}"
                    )
                yield record

    async def aclose(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


__all__ = ["JsonlStorage"]
