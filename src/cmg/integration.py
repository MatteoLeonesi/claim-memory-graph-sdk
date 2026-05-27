from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from cmg.graph import ClaimGraph
from cmg.nodes import AppendResult, Violation
from cmg.parser import _FENCE_RE, _TAG_RE, parse_turn

Message: TypeAlias = dict[str, str]
AsyncLLMFn: TypeAlias = Callable[[list[Message]], Awaitable[str]]
AsyncLLMStreamFn: TypeAlias = Callable[[list[Message]], AsyncIterator[str]]


@dataclass(frozen=True, slots=True)
class TurnResult:
    visible_text: str
    raw_text: str
    applied: tuple[AppendResult, ...]
    parse_warnings: tuple[str, ...]

    def violations(self) -> tuple[Violation, ...]:
        return tuple(v for a in self.applied for v in a.violations)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    visible_text_delta: str
    is_final: bool
    result: TurnResult | None


async def arun_turn(
    graph: ClaimGraph,
    llm_fn: AsyncLLMFn,
    messages: list[Message],
    *,
    inject_state: bool = True,
    on_violation: Callable[[Violation], None] | None = None,
) -> TurnResult:
    payload = _with_state(graph, messages) if inject_state else list(messages)
    response = await llm_fn(payload)
    return await _ingest_text(graph, response, on_violation)


async def astream_turn(
    graph: ClaimGraph,
    stream_fn: AsyncLLMStreamFn,
    messages: list[Message],
    *,
    inject_state: bool = True,
    on_violation: Callable[[Violation], None] | None = None,
) -> AsyncIterator[StreamChunk]:
    payload = _with_state(graph, messages) if inject_state else list(messages)
    stripper = _StreamingAnnotationStripper()
    raw_parts: list[str] = []
    async for delta in stream_fn(payload):
        raw_parts.append(delta)
        emit = stripper.update(delta)
        if emit:
            yield StreamChunk(visible_text_delta=emit, is_final=False, result=None)
    tail = stripper.finalize()
    if tail:
        yield StreamChunk(visible_text_delta=tail, is_final=False, result=None)
    raw_text = "".join(raw_parts)
    result = await _ingest_text(graph, raw_text, on_violation)
    yield StreamChunk(visible_text_delta="", is_final=True, result=result)


def build_annotation_system_prompt() -> str:
    return (
        "You may annotate your response with cmg ops to track your claims. "
        "Use a fenced block:\n"
        '```cmg\n{"ops": [{"op": "commitment", "content": "...", "refs": ["s-..."]}]}\n```\n'
        "Or a self-closing tag:\n"
        "<cmg ops='[{\"op\": \"commitment\", \"content\": \"...\", \"refs\": [\"s-...\"]}]'/>\n"
        "Op kinds: support, commitment, decision, invalidation. Annotations are optional; "
        "your prose is shown to the user verbatim either way."
    )


async def _ingest_text(
    graph: ClaimGraph,
    text: str,
    on_violation: Callable[[Violation], None] | None,
) -> TurnResult:
    parsed = parse_turn(text)
    applied: list[AppendResult] = []
    warnings = list(parsed.parse_warnings)
    for op in parsed.ops:
        result, warn = await _apply_op(graph, op)
        if warn is not None:
            warnings.append(warn)
        if result is not None:
            applied.append(result)
            if on_violation is not None:
                for v in result.violations:
                    on_violation(v)
    return TurnResult(
        visible_text=parsed.visible_text,
        raw_text=parsed.raw_text,
        applied=tuple(applied),
        parse_warnings=tuple(warnings),
    )


async def _apply_op(
    graph: ClaimGraph,
    op: dict[str, object],
) -> tuple[AppendResult | None, str | None]:
    name = op.get("op")
    try:
        match name:
            case "support":
                return (
                    await graph.add_support(
                        _str(op, "content"),
                        _seq(op, "refs"),
                        source_turn=_int(op, "source_turn", 0),
                    ),
                    None,
                )
            case "commitment":
                return (
                    await graph.add_commitment(_str(op, "content"), _seq(op, "refs")),
                    None,
                )
            case "decision":
                return (
                    await graph.add_decision(_str(op, "content"), _seq(op, "refs")),
                    None,
                )
            case "invalidation":
                return (
                    await graph.add_invalidation(
                        previous_commitment=_str(op, "previous_commitment"),
                        previous_support=_seq(op, "previous_support"),
                        new_information=_str(op, "new_information"),
                        contrast_test=_str(op, "contrast_test"),
                        result=_str(op, "result"),
                        content=_str(op, "content", default=""),
                        evidence_source_turn=_int(op, "evidence_source_turn", 0),
                    ),
                    None,
                )
            case _:
                return None, f"unknown op kind: {name!r}"
    except (KeyError, TypeError) as exc:
        return None, f"malformed {name!r} op: {exc}"


def _str(op: dict[str, object], key: str, *, default: str | None = None) -> str:
    if key not in op:
        if default is not None:
            return default
        raise KeyError(key)
    value = op[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _seq(op: dict[str, object], key: str) -> tuple[str, ...]:
    value = op.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{key} entries must be strings, got {type(item).__name__}")
    return tuple(value)


def _int(op: dict[str, object], key: str, default: int) -> int:
    value = op.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _with_state(graph: ClaimGraph, messages: list[Message]) -> list[Message]:
    return [_state_message(graph), *messages]


def _state_message(graph: ClaimGraph) -> Message:
    parts: list[str] = []
    if active := graph.active_commitments():
        parts.append("Active commitments:\n" + "\n".join(
            f"- {c.node_id}: {c.content}" for c in active
        ))
    if last := graph.last_decision():
        parts.append(f"Last decision ({last.node_id}): {last.content}")
    return {"role": "system", "content": "\n\n".join(parts) or "CMG state: empty graph."}


_PARTIAL_PREFIXES = ("```cmg", "<cmg")
MAX_STREAM_BUFFER_BYTES = 1 << 20


class _StreamingAnnotationStripper:
    def __init__(self) -> None:
        self._buf: str = ""

    def update(self, delta: str) -> str:
        self._buf += delta
        if len(self._buf.encode("utf-8", errors="ignore")) > MAX_STREAM_BUFFER_BYTES:
            flushed = self._buf
            self._buf = ""
            return flushed
        return self._drain(final=False)

    def finalize(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        out: list[str] = []
        while self._buf:
            start = self._earliest_start()
            if start is None:
                out.append(self._safe_flush(final=final))
                break
            if start > 0:
                out.append(self._buf[:start])
                self._buf = self._buf[start:]
            consumed = self._consume_complete_block()
            if consumed:
                continue
            if final:
                out.append(self._buf)
                self._buf = ""
            break
        return "".join(out)

    def _earliest_start(self) -> int | None:
        positions = [self._buf.find(p) for p in _PARTIAL_PREFIXES]
        valid = [p for p in positions if p != -1]
        return min(valid) if valid else None

    def _consume_complete_block(self) -> bool:
        if self._buf.startswith("```cmg"):
            m = _FENCE_RE.match(self._buf)
        else:
            m = _TAG_RE.match(self._buf)
        if m is None:
            return False
        self._buf = self._buf[m.end():]
        return True

    def _safe_flush(self, *, final: bool) -> str:
        if final:
            out = self._buf
            self._buf = ""
            return out
        cut = _max_partial_suffix(self._buf)
        if cut == 0:
            out = self._buf
            self._buf = ""
            return out
        out = self._buf[:-cut]
        self._buf = self._buf[-cut:]
        return out


def _max_partial_suffix(buf: str) -> int:
    for prefix in _PARTIAL_PREFIXES:
        for i in range(min(len(prefix) - 1, len(buf)), 0, -1):
            if buf.endswith(prefix[:i]):
                return i
    return 0


__all__ = [
    "AsyncLLMFn",
    "AsyncLLMStreamFn",
    "Message",
    "StreamChunk",
    "TurnResult",
    "arun_turn",
    "astream_turn",
    "build_annotation_system_prompt",
]
