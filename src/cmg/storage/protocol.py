from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from cmg.nodes import Node, Violation


@runtime_checkable
class Storage(Protocol):
    """Pluggable persistence for ClaimGraph. Records are JSON-serializable dicts."""

    async def append_node(self, node: Node) -> None: ...

    async def append_violation(self, violation: Violation) -> None: ...

    def iter_records(self) -> Iterator[dict[str, object]]: ...

    async def aclose(self) -> None: ...


__all__ = ["Storage"]
