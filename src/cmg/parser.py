from __future__ import annotations

import json
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```cmg\s*\n(?P<body>.*?)\n```", re.DOTALL)
_TAG_RE = re.compile(
    r"<cmg\s+ops=(?P<q>['\"])(?P<body>.*?)(?P=q)\s*/>",
    re.DOTALL,
)
_BLANKLINES_RE = re.compile(r"\n{3,}")

MAX_PARSE_INPUT_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class ParsedTurn:
    visible_text: str
    raw_text: str
    ops: tuple[dict[str, object], ...]
    parse_warnings: tuple[str, ...]


def parse_turn(text: str) -> ParsedTurn:
    """Extract cmg op annotations from a model turn. Never raises."""
    truncated_warning: list[str] = []
    if len(text.encode("utf-8", errors="ignore")) > MAX_PARSE_INPUT_BYTES:
        text = text.encode("utf-8", errors="ignore")[:MAX_PARSE_INPUT_BYTES].decode(
            "utf-8", errors="ignore"
        )
        truncated_warning.append(f"input truncated to {MAX_PARSE_INPUT_BYTES} bytes")
    spans: list[tuple[int, int, str, str]] = []
    for m in _FENCE_RE.finditer(text):
        spans.append((m.start(), m.end(), "fence", m.group("body")))
    for m in _TAG_RE.finditer(text):
        spans.append((m.start(), m.end(), "tag", m.group("body")))
    spans.sort()

    ops: list[dict[str, object]] = []
    warnings: list[str] = []
    for start, _, source, body in spans:
        ops_from_block, warn = _extract_ops(body, source, start)
        ops.extend(ops_from_block)
        if warn is not None:
            warnings.append(warn)

    visible = _strip_spans(text, [(s, e) for s, e, _, _ in spans])
    return ParsedTurn(
        visible_text=visible,
        raw_text=text,
        ops=tuple(ops),
        parse_warnings=tuple([*truncated_warning, *warnings]),
    )


def _extract_ops(body: str, source: str, offset: int) -> tuple[list[dict[str, object]], str | None]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return [], f"{source} at offset {offset}: invalid JSON ({exc.msg})"

    raw_ops: list[object]
    match data:
        case list():
            raw_ops = data
        case {"ops": list() as lst}:
            raw_ops = lst
        case _:
            return [], f"{source} at offset {offset}: expected list or {{'ops': [...]}}"

    return [entry for entry in raw_ops if isinstance(entry, dict)], None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text.strip()
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return _BLANKLINES_RE.sub("\n\n", "".join(parts)).strip()


__all__ = ["MAX_PARSE_INPUT_BYTES", "ParsedTurn", "parse_turn"]
