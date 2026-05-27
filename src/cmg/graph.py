from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from types import TracebackType

from cmg._schema import SCHEMA_VERSION, migrate_record
from cmg.checks import GraphState, check_node
from cmg.errors import DuplicateNodeIdError, MalformedLogLineError
from cmg.nodes import (
    INVALIDATION_ACCEPTED,
    NODE_CLASSES,
    AppendResult,
    Commitment,
    Decision,
    Invalidation,
    Node,
    Support,
    Violation,
    mint_id,
    now_iso,
)
from cmg.storage import Storage


def _identity(s: str) -> str:
    return s


class ClaimGraph:
    """Append-only DAG of typed claims; observes (does not block) semantic deviations."""

    def __init__(
        self,
        storage: Storage,
        *,
        redact_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._storage = storage
        self._redact = redact_fn or _identity
        self._lock = asyncio.Lock()
        self._nodes: list[Node] = []
        self._by_id: dict[str, Node] = {}
        self._active: set[str] = set()
        self._violations: list[Violation] = []
        self._last_decision: Decision | None = None

    @classmethod
    async def aload(
        cls,
        storage: Storage,
        *,
        redact_fn: Callable[[str], str] | None = None,
    ) -> ClaimGraph:
        graph = cls(storage, redact_fn=redact_fn)
        await graph._replay()
        return graph

    async def __aenter__(self) -> ClaimGraph:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def add_support(
        self,
        content: str,
        refs: Iterable[str] = (),
        *,
        source_turn: int = 0,
    ) -> AppendResult:
        async with self._lock:
            node = Support(
                node_id=mint_id("support"),
                content=self._redact(content),
                created_at=now_iso(),
                refs=tuple(refs),
                source_turn=source_turn,
            )
            return await self._commit(node)

    async def add_commitment(self, content: str, refs: Iterable[str]) -> AppendResult:
        async with self._lock:
            node = Commitment(
                node_id=mint_id("commitment"),
                content=self._redact(content),
                created_at=now_iso(),
                refs=tuple(refs),
            )
            return await self._commit(node)

    async def add_decision(self, content: str, refs: Iterable[str]) -> AppendResult:
        async with self._lock:
            node = Decision(
                node_id=mint_id("decision"),
                content=self._redact(content),
                created_at=now_iso(),
                refs=tuple(refs),
            )
            return await self._commit(node)

    async def add_invalidation(
        self,
        *,
        previous_commitment: str,
        previous_support: Iterable[str],
        new_information: str,
        contrast_test: str,
        result: str,
        content: str = "",
        evidence_source_turn: int = 0,
    ) -> AppendResult:
        async with self._lock:
            node = Invalidation(
                node_id=mint_id("invalidation"),
                created_at=now_iso(),
                previous_commitment=previous_commitment,
                previous_support=tuple(previous_support),
                new_information=self._redact(new_information),
                contrast_test=self._redact(contrast_test),
                result=result,
                content=self._redact(content) if content else "",
                evidence_source_turn=evidence_source_turn,
            )
            return await self._commit(node)

    async def _commit(self, node: Node) -> AppendResult:
        if node.node_id in self._by_id:
            raise DuplicateNodeIdError(node.node_id)
        violations = check_node(node, self._snapshot_state())
        await self._storage.append_node(node)
        for v in violations:
            await self._storage.append_violation(v)
        self._apply(node, violations)
        return AppendResult(node=node, violations=violations)

    def _snapshot_state(self) -> GraphState:
        return GraphState(self._by_id, frozenset(self._active), self._last_decision)

    def _apply(self, node: Node, violations: tuple[Violation, ...]) -> None:
        self._nodes.append(node)
        self._by_id[node.node_id] = node
        self._violations.extend(violations)
        match node:
            case Commitment():
                self._active.add(node.node_id)
            case Invalidation() if node.result == INVALIDATION_ACCEPTED:
                self._active.discard(node.previous_commitment)
            case Decision():
                self._last_decision = node

    async def _replay(self) -> None:
        on_disk_violations: dict[str, list[str]] = {}
        nodes_in_order: list[Node] = []
        for record in self._storage.iter_records():
            raw_kind = record.get("record")
            raw_version = record.get("cmg_schema_version", SCHEMA_VERSION)
            version = raw_version if isinstance(raw_version, int) else SCHEMA_VERSION
            if version != SCHEMA_VERSION:
                record = migrate_record(record, version)
            match raw_kind:
                case "node":
                    nodes_in_order.append(_node_from_record(record))
                case "violation":
                    nid = _required_str(record, "node_id")
                    on_disk_violations.setdefault(nid, []).append(_required_str(record, "code"))
                case _:
                    raise MalformedLogLineError(f"unknown record type: {raw_kind!r}")

        for node in nodes_in_order:
            if node.node_id in self._by_id:
                raise DuplicateNodeIdError(node.node_id)
            violations = check_node(node, self._snapshot_state())
            self._apply(node, violations)
            recomputed = [v.code for v in violations]
            expected = on_disk_violations.get(node.node_id, [])
            if recomputed != expected:
                self._violations.append(
                    Violation(
                        node_id=node.node_id,
                        code="replay_mismatch",
                        detail={"on_disk": expected, "recomputed": recomputed},
                        created_at=now_iso(),
                    )
                )

    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def get(self, node_id: str) -> Node | None:
        return self._by_id.get(node_id)

    def active_commitments(self) -> tuple[Commitment, ...]:
        return tuple(
            n for n in self._nodes if isinstance(n, Commitment) and n.node_id in self._active
        )

    def active_commitment_ids(self) -> frozenset[str]:
        return frozenset(self._active)

    def last_decision(self) -> Decision | None:
        return self._last_decision

    def violations(self) -> tuple[Violation, ...]:
        return tuple(self._violations)

    def violations_for(self, node_id: str) -> tuple[Violation, ...]:
        return tuple(v for v in self._violations if v.node_id == node_id)

    async def aclose(self) -> None:
        await self._storage.aclose()


def _node_from_record(record: dict[str, object]) -> Node:
    kind = record.get("kind")
    if not isinstance(kind, str) or kind not in NODE_CLASSES:
        raise MalformedLogLineError(f"missing or unknown kind: {kind!r}")
    match kind:
        case "support":
            return Support(
                node_id=_required_str(record, "node_id"),
                content=_required_str(record, "content"),
                created_at=_required_str(record, "created_at"),
                refs=_str_tuple(record, "refs"),
                source_turn=_required_int(record, "source_turn", default=0),
            )
        case "commitment":
            return Commitment(
                node_id=_required_str(record, "node_id"),
                content=_required_str(record, "content"),
                created_at=_required_str(record, "created_at"),
                refs=_str_tuple(record, "refs"),
            )
        case "decision":
            return Decision(
                node_id=_required_str(record, "node_id"),
                content=_required_str(record, "content"),
                created_at=_required_str(record, "created_at"),
                refs=_str_tuple(record, "refs"),
            )
        case "invalidation":
            return Invalidation(
                node_id=_required_str(record, "node_id"),
                created_at=_required_str(record, "created_at"),
                previous_commitment=_required_str(record, "previous_commitment"),
                previous_support=_str_tuple(record, "previous_support"),
                new_information=_required_str(record, "new_information"),
                contrast_test=_required_str(record, "contrast_test"),
                result=_required_str(record, "result"),
                content=_required_str(record, "content", default=""),
                evidence_source_turn=_required_int(record, "evidence_source_turn", default=0),
            )
        case _:
            raise MalformedLogLineError(f"unhandled kind: {kind!r}")


def _required_str(record: dict[str, object], key: str, *, default: str | None = None) -> str:
    if key not in record and default is not None:
        return default
    value = record.get(key)
    if not isinstance(value, str):
        raise MalformedLogLineError(f"expected str for {key!r}, got {type(value).__name__}")
    return value


def _required_int(record: dict[str, object], key: str, *, default: int) -> int:
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedLogLineError(f"expected int for {key!r}, got {type(value).__name__}")
    return value


def _str_tuple(record: dict[str, object], key: str) -> tuple[str, ...]:
    value = record.get(key, [])
    if not isinstance(value, list):
        raise MalformedLogLineError(f"expected list for {key!r}, got {type(value).__name__}")
    for v in value:
        if not isinstance(v, str):
            raise MalformedLogLineError(f"non-string entry in {key!r}")
    return tuple(value)


__all__ = ["ClaimGraph"]
