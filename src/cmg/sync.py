from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from types import TracebackType
from typing import TypeVar

from cmg.graph import ClaimGraph
from cmg.integration import AsyncLLMFn, Message, TurnResult, arun_turn
from cmg.nodes import AppendResult, Commitment, Decision, Node, Violation
from cmg.storage import Storage

T = TypeVar("T")


class ClaimGraphSync:
    """Synchronous wrapper around ClaimGraph for scripts and notebooks.

    Cannot be used from inside a running event loop; use ClaimGraph directly there.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        redact_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._async: ClaimGraph = ClaimGraph(storage, redact_fn=redact_fn)
        self._closed = False

    @classmethod
    def load(
        cls,
        storage: Storage,
        *,
        redact_fn: Callable[[str], str] | None = None,
    ) -> ClaimGraphSync:
        inst = cls.__new__(cls)
        inst._loop = asyncio.new_event_loop()
        inst._closed = False
        inst._async = inst._run(ClaimGraph.aload(storage, redact_fn=redact_fn))
        return inst

    def _run(self, coro: Awaitable[T]) -> T:
        if self._closed:
            raise RuntimeError("ClaimGraphSync is closed")
        return self._loop.run_until_complete(coro)

    def __enter__(self) -> ClaimGraphSync:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def add_support(
        self,
        content: str,
        refs: Iterable[str] = (),
        *,
        source_turn: int = 0,
    ) -> AppendResult:
        return self._run(self._async.add_support(content, refs, source_turn=source_turn))

    def add_commitment(self, content: str, refs: Iterable[str]) -> AppendResult:
        return self._run(self._async.add_commitment(content, refs))

    def add_decision(self, content: str, refs: Iterable[str]) -> AppendResult:
        return self._run(self._async.add_decision(content, refs))

    def add_invalidation(
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
        return self._run(
            self._async.add_invalidation(
                previous_commitment=previous_commitment,
                previous_support=previous_support,
                new_information=new_information,
                contrast_test=contrast_test,
                result=result,
                content=content,
                evidence_source_turn=evidence_source_turn,
            )
        )

    def run_turn(
        self,
        llm_fn: AsyncLLMFn,
        messages: list[Message],
        *,
        inject_state: bool = True,
        on_violation: Callable[[Violation], None] | None = None,
    ) -> TurnResult:
        return self._run(
            arun_turn(
                self._async,
                llm_fn,
                messages,
                inject_state=inject_state,
                on_violation=on_violation,
            )
        )

    def nodes(self) -> tuple[Node, ...]:
        return self._async.nodes()

    def get(self, node_id: str) -> Node | None:
        return self._async.get(node_id)

    def active_commitments(self) -> tuple[Commitment, ...]:
        return self._async.active_commitments()

    def active_commitment_ids(self) -> frozenset[str]:
        return self._async.active_commitment_ids()

    def last_decision(self) -> Decision | None:
        return self._async.last_decision()

    def violations(self) -> tuple[Violation, ...]:
        return self._async.violations()

    def violations_for(self, node_id: str) -> tuple[Violation, ...]:
        return self._async.violations_for(node_id)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._run(self._async.aclose())
        finally:
            self._closed = True
            self._loop.close()


__all__ = ["ClaimGraphSync"]
