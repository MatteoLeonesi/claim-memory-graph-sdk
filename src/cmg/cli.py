from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from textwrap import shorten
from typing import TextIO, cast

from cmg.graph import ClaimGraph
from cmg.nodes import Support
from cmg.report import judge_report
from cmg.storage import JsonlStorage

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
_DEVNULL: TextIO | None = None

OWL = (
    "  ()_()",
    "  (o.o)",
    "  (> <)",
)

FLAG_HINTS = {
    "missing_verdict": "Check the judge response format.",
    "invalid_verdict": "Check the allowed verdict labels.",
    "uncited_verdict": "Check whether the verdict cites active claims.",
    "no_supported_claims": "Check whether the judge made grounded claims.",
    "criterion_citation_gap": "Check whether criterion ids should be cited directly.",
    "rubric_coverage_gap": "Check criteria marked as missing.",
    "reference_ignored": "Check whether the reference answer should be cited.",
    "verdict_flip_without_invalidation": "Check whether old claims need a retraction.",
    "silent_commitment_drop": "Check why an active claim disappeared from the verdict.",
}

HARD_FLAGS = frozenset({
    "empty_refs",
    "unknown_ref",
    "wrong_ref_kind",
    "uncited_verdict",
    "no_supported_claims",
    "invalid_verdict",
    "missing_verdict",
    "verdict_flip_without_invalidation",
    "silent_commitment_drop",
})

SOFT_FLAGS = frozenset({
    "criterion_citation_gap",
    "rubric_coverage_gap",
    "reference_ignored",
})


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, color: str) -> str:
        if not self.enabled:
            return text
        return f"{color}{text}{RESET}"

    def heading(self, text: str) -> str:
        return self.paint(text, BOLD + CYAN)

    def muted(self, text: str) -> str:
        return self.paint(text, DIM)

    def ok(self, text: str) -> str:
        return self.paint(text, GREEN)

    def warn(self, text: str) -> str:
        return self.paint(text, YELLOW)

    def bad(self, text: str) -> str:
        return self.paint(text, RED)

    def link(self, text: str) -> str:
        return self.paint(text, BLUE)


def _case_status(flags: Sequence[str], palette: Palette) -> str:
    if _hard_flags(flags):
        return palette.bad("REVIEW")
    if _soft_flags(flags):
        return palette.warn("REVIEW")
    return palette.ok("CLEAR")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


async def _amain(args: argparse.Namespace) -> int:
    paths = _expand_paths(cast(list[str], args.paths))
    if not paths:
        print("cmg-view: no JSONL files matched", file=sys.stderr)
        return 2

    loaded = await _load_cases(paths)
    if args.json:
        visible = [_case for _case in loaded if _visible(_case, cast(bool, args.flagged_only))]
        _write_output(json.dumps([case["report"] for case in visible], indent=2, ensure_ascii=False))
        return 0

    palette = Palette(_color_enabled(cast(str, args.color), sys.stdout))
    if args.summary:
        _write_output(_render_summary(
            loaded,
            palette=palette,
            flagged_only=cast(bool, args.flagged_only),
            width=cast(int, args.width),
        ))
        return 0

    _write_output(_render_cases(
        loaded,
        palette=palette,
        flagged_only=cast(bool, args.flagged_only),
        show_evidence=cast(bool, args.show_evidence),
        width=cast(int, args.width),
    ))
    return 0


def _write_output(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        global _DEVNULL
        _DEVNULL = os.fdopen(os.open(os.devnull, os.O_WRONLY), "w")
        sys.stdout = _DEVNULL
        raise


async def _load_cases(paths: Sequence[Path]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for path in paths:
        graph = await ClaimGraph.aload(JsonlStorage(path))
        supports = [node for node in graph.nodes() if isinstance(node, Support)]
        cases.append(
            {
                "path": path,
                "case_id": path.stem,
                "report": judge_report(graph),
                "supports": supports,
            }
        )
        await graph.aclose()
    return cases


def _render_cases(
    cases: Sequence[dict[str, object]],
    *,
    palette: Palette,
    flagged_only: bool,
    show_evidence: bool,
    width: int,
) -> str:
    visible = [_case for _case in cases if _visible(_case, flagged_only)]
    flagged = sum(1 for case in cases if _flags(case))
    width = _safe_width(width)
    lines = [
        *_title_block(palette),
        _rule(width),
        (
            f"cases: {len(cases)} loaded | "
            f"{palette.warn(str(flagged) + ' flagged') if flagged else palette.ok('0 flagged')} | "
            f"shown: {len(visible)}"
        ),
        _rule(width, "-"),
        "",
    ]
    if not visible:
        lines.append("No cases to show.")
        return "\n".join(lines)

    for index, case in enumerate(visible, start=1):
        if index > 1:
            lines.append("")
        lines.extend(_render_case(
            case,
            palette=palette,
            show_evidence=show_evidence,
            width=width,
        ))
    return "\n".join(lines)


def _render_summary(
    cases: Sequence[dict[str, object]],
    *,
    palette: Palette,
    flagged_only: bool,
    width: int,
) -> str:
    visible = [_case for _case in cases if _visible(_case, flagged_only)]
    total = len(cases)
    flagged = sum(1 for case in cases if _flags(case))
    width = _safe_width(width)
    lines = [
        *_title_block(palette),
        _rule(width),
        palette.heading("Summary"),
        (
            f"cases: {total} loaded | "
            f"{palette.warn(str(flagged) + ' flagged')} | "
            f"{palette.ok(str(total - flagged) + ' clear')} | "
            f"summarized: {len(visible)}"
        ),
        _rule(width, "-"),
        "",
    ]
    if not visible:
        lines.append("No cases to summarize.")
        return "\n".join(lines)

    lines.extend(_summary_section("Verdicts", _verdict_summary(visible, palette)))
    lines.extend(_summary_section("Hard flags", _flag_summary(visible, HARD_FLAGS, palette.bad)))
    lines.extend(_summary_section("Soft flags", _flag_summary(visible, SOFT_FLAGS, palette.warn)))
    lines.extend(_summary_section("Criteria", _criteria_summary(visible, palette)))
    lines.extend(_summary_section("Top review cases", _top_review_cases(visible, palette)))
    return "\n".join(lines)


def _summary_section(title: str, rows: Sequence[str]) -> list[str]:
    if not rows:
        return []
    return [f"[{title}]", *rows, ""]


def _verdict_summary(cases: Sequence[dict[str, object]], palette: Palette) -> list[str]:
    verdicts = Counter(str(_report(case)["verdict"] or "(none)") for case in cases)
    total = sum(verdicts.values())
    return [
        _summary_row(verdict, count, total, _paint_verdict_label(verdict, palette))
        for verdict, count in sorted(verdicts.items(), key=_sort_verdict)
    ]


def _flag_summary(
    cases: Sequence[dict[str, object]],
    flag_set: frozenset[str],
    paint_label: Callable[[str], str],
) -> list[str]:
    counts = Counter(flag for case in cases for flag in _flags(case) if flag in flag_set)
    if not counts:
        return ["none"]
    total = len(cases)
    return [
        _summary_row(flag, count, total, paint_label)
        for flag, count in counts.most_common()
    ]


def _criteria_summary(cases: Sequence[dict[str, object]], palette: Palette) -> list[str]:
    covered = 0
    citation_gap = 0
    missing = 0
    for case in cases:
        criteria = cast(list[dict[str, object]], _report(case)["criteria"])
        for item in criteria:
            if item["citation_covered"]:
                covered += 1
            elif item["covered"]:
                citation_gap += 1
            else:
                missing += 1
    total = covered + citation_gap + missing
    if total == 0:
        return [palette.muted("No criteria recorded.")]
    return [
        _summary_row("covered", covered, total, palette.ok),
        _summary_row("citation gap", citation_gap, total, palette.warn),
        _summary_row("missing", missing, total, palette.warn),
    ]


def _top_review_cases(cases: Sequence[dict[str, object]], palette: Palette) -> list[str]:
    ranked = sorted(
        (case for case in cases if _flags(case)),
        key=lambda case: (-len(_hard_flags(_flags(case))), -len(_flags(case)), str(case["case_id"])),
    )
    if not ranked:
        return [palette.ok("No review cases.")]
    rows = []
    for case in ranked[:5]:
        flags = _flag_text(_flags(case), palette)
        rows.append(f"- {palette.warn(str(case['case_id']))}: {flags}")
    return rows


def _summary_row(
    label: str,
    count: int,
    total: int,
    paint_label: Callable[[str], str],
) -> str:
    return f"{paint_label(label[:24].ljust(24))} {count:>3}/{total:<3} {_bar(count, total)}"


def _paint_verdict_label(verdict: str, palette: Palette) -> Callable[[str], str]:
    normalized = verdict.casefold()
    if normalized in {"pass", "correct", "yes", "approve"}:
        return palette.ok
    if normalized in {"fail", "incorrect", "no", "reject"}:
        return palette.bad
    return palette.warn


def _bar(count: int, total: int, *, width: int = 24) -> str:
    filled = 0 if total == 0 else round(width * count / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _sort_verdict(item: tuple[str, int]) -> tuple[int, float | str]:
    verdict, _ = item
    try:
        return (0, float(verdict))
    except ValueError:
        return (1, verdict)


def _render_case(
    case: dict[str, object],
    *,
    palette: Palette,
    show_evidence: bool,
    width: int,
) -> list[str]:
    report = cast(dict[str, object], case["report"])
    case_id = cast(str, case["case_id"])
    path = cast(Path, case["path"])
    supports = cast(list[Support], case["supports"])
    flags = cast(list[str], report["human_review_flags"])
    verdict = str(report["verdict"] or "(none)")
    verdict_text = _verdict_text(verdict, palette)
    header = f"{case_id} | {_case_status(flags, palette)}"

    lines = [
        palette.heading(_rule(width, "=")),
        f"{palette.heading(header)}",
        f"file:    {palette.muted(str(path))}",
        f"verdict: {verdict_text}",
        f"flags:   {_flag_text(flags, palette)}",
    ]
    lines.extend(_section("Review hints", _review_hints(flags, palette), palette))
    lines.extend(_section("Verdict errors", _verdict_errors(report, width, palette), palette))
    lines.extend(_section("Judge response", _judge_responses(report, width), palette))
    lines.extend(_section("Claims", _claims(report, width, palette), palette))
    lines.extend(_section("Criteria", _criteria(report, width, palette), palette))
    lines.extend(_section("Retractions", _retractions(report, width), palette))
    lines.extend(_section("Consistency", _violations(report, palette), palette))
    if show_evidence:
        lines.extend(_section("Evidence", _evidence(supports, width, palette), palette))
    return lines


def _title_block(palette: Palette) -> list[str]:
    title = (
        "CMG Judge Audit",
        "claim memory graph",
        "",
    )
    return [
        f"{palette.heading(owl)}  {title_line}" if title_line else palette.heading(owl)
        for owl, title_line in zip(OWL, title, strict=True)
    ]


def _section(title: str, rows: Sequence[str], palette: Palette) -> list[str]:
    if not rows:
        return []
    return ["", palette.heading(f"-- {title} --"), *rows]


def _judge_responses(report: dict[str, object], width: int) -> list[str]:
    rows = []
    for item in cast(list[dict[str, str]], report["judge_responses"]):
        rows.append(_wrap_block(item["content"], width))
    return rows


def _review_hints(flags: Sequence[str], palette: Palette) -> list[str]:
    if not flags:
        return []
    return [
        f"- {_paint_flag(flag, palette)}: {FLAG_HINTS.get(flag, 'Check this case by hand.')}"
        for flag in flags
    ]


def _verdict_errors(
    report: dict[str, object],
    width: int,
    palette: Palette,
) -> list[str]:
    errors = cast(list[dict[str, str]], report.get("verdict_errors", []))
    if not errors:
        return []
    return [f"- {palette.warn(_short(item['content'], width))}" for item in errors]


def _claims(report: dict[str, object], width: int, palette: Palette) -> list[str]:
    claims = cast(list[dict[str, object]], report["claims"])
    if not claims:
        return [palette.muted("No supported claims.")]
    rows = []
    for index, claim in enumerate(claims, start=1):
        evidence = ", ".join(cast(list[str], claim["evidence"]))
        rows.append(f"{index}. {_short(str(claim['claim']), width)}")
        if evidence:
            rows.append(f"   evidence: {palette.link(evidence)}")
    return rows


def _criteria(report: dict[str, object], width: int, palette: Palette) -> list[str]:
    criteria = cast(list[dict[str, object]], report["criteria"])
    if not criteria:
        return []
    rows = []
    for item in criteria:
        if item["citation_covered"]:
            status = palette.ok("covered")
        elif item["covered"]:
            status = palette.warn("citation gap")
        else:
            status = palette.warn("missing")
        rows.append(f"- {status}  {_short(str(item['criterion']), width)}")
    return rows


def _retractions(report: dict[str, object], width: int) -> list[str]:
    rows = []
    for item in cast(list[dict[str, str]], report["retracted"]):
        rows.append(
            f"- {item['result']}: {_short(item['contrast_test'], width)} "
            f"({item['commitment']})"
        )
    return rows


def _violations(report: dict[str, object], palette: Palette) -> list[str]:
    violations = cast(list[str], report["violations"])
    if not violations:
        return [palette.ok("No consistency violations.")]
    return [palette.warn(", ".join(violations))]


def _evidence(supports: Sequence[Support], width: int, palette: Palette) -> list[str]:
    rows = []
    for support in supports:
        if support.content.startswith("Judge response:\n"):
            continue
        rows.append(f"- {palette.link(support.node_id)}  {_short(support.content, width)}")
    return rows or [palette.muted("No evidence supports.")]


def _visible(case: dict[str, object], flagged_only: bool) -> bool:
    return not flagged_only or bool(_flags(case))


def _flags(case: dict[str, object]) -> list[str]:
    return cast(list[str], _report(case)["human_review_flags"])


def _report(case: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], case["report"])


def _flag_text(flags: Sequence[str], palette: Palette) -> str:
    if not flags:
        return palette.ok("none")
    return ", ".join(_paint_flag(flag, palette) for flag in flags)


def _paint_flag(flag: str, palette: Palette) -> str:
    if flag in HARD_FLAGS:
        return palette.bad(flag)
    if flag in SOFT_FLAGS:
        return palette.warn(flag)
    return palette.warn(flag)


def _hard_flags(flags: Sequence[str]) -> list[str]:
    return [flag for flag in flags if flag in HARD_FLAGS]


def _soft_flags(flags: Sequence[str]) -> list[str]:
    return [flag for flag in flags if flag in SOFT_FLAGS]


def _verdict_text(verdict: str, palette: Palette) -> str:
    normalized = verdict.casefold()
    if normalized in {"pass", "correct", "yes", "approve"}:
        return palette.ok(verdict)
    if normalized in {"fail", "incorrect", "no", "reject"}:
        return palette.bad(verdict)
    return palette.warn(verdict)


def _rule(width: int, fill: str = "=") -> str:
    return fill * _safe_width(width)


def _safe_width(width: int) -> int:
    return min(140, max(60, width))


def _wrap_block(text: str, width: int) -> str:
    lines = [_short(line, width) for line in text.splitlines()]
    return "\n".join(f"  {line}" for line in lines if line)


def _short(text: str, width: int) -> str:
    compact = " ".join(text.split())
    return shorten(compact, width=max(20, width), placeholder="...")


def _expand_paths(raw_paths: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        matches = sorted(glob.glob(raw))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(raw))
    return [path for path in paths if path.exists()]


def _color_enabled(mode: str, stream: TextIO) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return stream.isatty() and "NO_COLOR" not in os.environ


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmg-view",
        description="Render Claim Memory Graph JSONL logs as a terminal audit view.",
    )
    parser.add_argument("paths", nargs="+", help="JSONL graph paths or glob patterns")
    parser.add_argument("--flagged-only", action="store_true", help="show only cases with flags")
    parser.add_argument("--show-evidence", action="store_true", help="include evidence supports")
    parser.add_argument("--summary", action="store_true", help="show a compact run summary")
    parser.add_argument("--json", action="store_true", help="print machine-readable reports")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="color output policy",
    )
    parser.add_argument("--width", type=int, default=110, help="text truncation width")
    return parser


__all__ = ["main"]
