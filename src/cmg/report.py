from __future__ import annotations

import re

from cmg.graph import ClaimGraph
from cmg.judge import active_claims
from cmg.nodes import INVALIDATION_ACCEPTED, Commitment, Decision, Invalidation, Support

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"and", "the", "with", "for", "that", "this", "must", "should"})
_CRITERION_ALIASES = {
    "instruction following": (
        "instruction",
        "instructions",
        "follows",
        "following",
        "followed",
        "meets the user request",
        "satisfies the user request",
        "addresses the user request",
        "fulfills the user request",
        "as requested",
        "requested",
        "prompt",
        "requirement",
        "requirements",
        "word limit",
        "format",
    ),
    "helpfulness": (
        "helpful",
        "useful",
        "actionable",
        "practical",
        "addresses",
        "provides",
        "recommendations",
        "guidance",
    ),
    "completeness": (
        "complete",
        "completeness",
        "covers",
        "covering",
        "addresses all",
        "all required",
        "without omission",
        "omits",
        "incomplete",
        "missing",
    ),
    "clarity": (
        "clear",
        "clarity",
        "coherent",
        "structured",
        "well organized",
        "well structured",
        "easy to read",
        "easy to follow",
        "concise",
        "readable",
    ),
    "correctness": (
        "correct",
        "accurate",
        "accuracy",
        "matches",
        "identifies",
        "valid",
        "right",
    ),
    "factual accuracy": (
        "factual",
        "accurate",
        "accuracy",
        "correct",
        "factually",
    ),
    "explanation quality": (
        "explanation",
        "explains",
        "explain",
        "reasoning",
        "rationale",
        "cause",
        "quality",
    ),
    "concision": (
        "concise",
        "brief",
        "short",
        "to the point",
        "compact",
    ),
}


def judge_report(graph: ClaimGraph) -> dict[str, object]:
    """Machine-readable judge audit report derived only from graph state."""
    supports = _support_index(graph)
    claims = active_claims(graph)
    decision = graph.last_decision()
    criteria = _criteria_coverage(supports, claims)
    return {
        "verdict": decision.content if decision else None,
        "verdict_refs": list(decision.refs) if decision else [],
        "claims": [
            {
                "id": c.node_id,
                "claim": c.content,
                "evidence": [r for r in c.refs if r in supports],
            }
            for c in claims
        ],
        "criteria": criteria,
        "judge_responses": _support_contents(supports, "Judge response:\n"),
        "verdict_errors": _support_contents(supports, "Judge verdict error:\n"),
        "retracted": [
            {
                "commitment": inv.previous_commitment,
                "new_information": inv.new_information,
                "contrast_test": inv.contrast_test,
                "result": inv.result,
            }
            for inv in _accepted_invalidations(graph)
        ],
        "human_review_flags": _human_review_flags(graph, supports, claims, decision, criteria),
        "violations": sorted({v.code for v in graph.violations()}),
    }


def to_markdown(graph: ClaimGraph, *, confidence: dict[str, int] | None = None) -> str:
    """Render a compact Markdown audit report for a human reviewer."""
    supports = _support_index(graph)
    claims = active_claims(graph)
    decision = graph.last_decision()
    criteria = _criteria_coverage(supports, claims)
    flags = _human_review_flags(graph, supports, claims, decision, criteria)

    out: list[str] = ["# Judge audit report", ""]
    out.append(f"**Verdict:** {decision.content if decision else '(none)'}")
    out.append("")
    out.append(f"## Claims ({len(claims)})")
    if not claims:
        out.append("_No supported claims._")
    for c in claims:
        tag = f" - confidence {confidence.get(c.node_id, 0)}" if confidence else ""
        out.append(f"- **{c.content}**{tag}")
        for ref in c.refs:
            if ref in supports:
                head = (supports[ref].content.splitlines() or [""])[0]
                out.append(f"    - evidence `{ref}`: {head}")

    if criteria:
        out += ["", "## Criteria coverage"]
        for item in criteria:
            if item["citation_covered"]:
                status = "covered"
            elif item["covered"]:
                status = "citation gap"
            else:
                status = "missing"
            out.append(f"- {item['criterion']} - {status}")

    invalids = _accepted_invalidations(graph)
    if invalids:
        out += ["", "## Retractions"]
        out += [f"- {inv.previous_commitment}: {inv.contrast_test} -> {inv.result}" for inv in invalids]

    if flags:
        out += ["", "## Human review flags", ", ".join(flags)]

    codes = sorted({v.code for v in graph.violations()})
    if codes:
        out += ["", "## Consistency ledger", ", ".join(codes)]

    return "\n".join(out) + "\n"


def _support_index(graph: ClaimGraph) -> dict[str, Support]:
    return {node.node_id: node for node in graph.nodes() if isinstance(node, Support)}


def _invalidations(graph: ClaimGraph) -> list[Invalidation]:
    return [node for node in graph.nodes() if isinstance(node, Invalidation)]


def _accepted_invalidations(graph: ClaimGraph) -> list[Invalidation]:
    return [node for node in _invalidations(graph) if node.result == INVALIDATION_ACCEPTED]


def _support_contents(supports: dict[str, Support], prefix: str) -> list[dict[str, str]]:
    return [
        {
            "id": support.node_id,
            "content": support.content.removeprefix(prefix),
        }
        for support in supports.values()
        if support.content.startswith(prefix)
    ]


def _criteria_coverage(
    supports: dict[str, Support],
    claims: tuple[Commitment, ...],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for support in supports.values():
        prefix = "Criterion:\n"
        if not support.content.startswith(prefix):
            continue
        criterion = support.content.removeprefix(prefix).strip()
        citation_covered = any(support.node_id in claim.refs for claim in claims)
        out.append(
            {
                "support_id": support.node_id,
                "criterion": criterion,
                "citation_covered": citation_covered,
                "covered": citation_covered or _criterion_discussed(criterion, claims),
            }
        )
    return out


def _human_review_flags(
    graph: ClaimGraph,
    supports: dict[str, Support],
    claims: tuple[Commitment, ...],
    decision: Decision | None,
    criteria: list[dict[str, object]],
) -> list[str]:
    flags = {v.code for v in graph.violations()}
    if decision is None:
        flags.add("missing_verdict")
    elif not decision.refs:
        flags.add("uncited_verdict")
    if _verdict_error(supports, "invalid_verdict"):
        flags.add("invalid_verdict")
    if _verdict_error(supports, "missing_verdict"):
        flags.add("missing_verdict")
    if not claims:
        flags.add("no_supported_claims")
    if any(not item["citation_covered"] for item in criteria):
        flags.add("criterion_citation_gap")
    if any(not item["covered"] for item in criteria):
        flags.add("rubric_coverage_gap")
    if _reference_ignored(supports, claims):
        flags.add("reference_ignored")
    return sorted(flags)


def _criterion_discussed(criterion: str, claims: tuple[Commitment, ...]) -> bool:
    criterion_text = _normalize(criterion)
    aliases = _CRITERION_ALIASES.get(criterion_text, ())
    terms = _terms(criterion_text)
    for claim in claims:
        claim_text = _normalize(claim.content)
        if aliases and any(_contains_term(claim_text, alias) for alias in aliases):
            return True
        if terms and all(term in claim_text.split() for term in terms):
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def _terms(text: str) -> tuple[str, ...]:
    return tuple(word for word in text.split() if len(word) > 2 and word not in _STOPWORDS)


def _contains_term(text: str, term: str) -> bool:
    term_text = _normalize(term)
    if " " in term_text:
        return term_text in text
    return term_text in text.split()


def _verdict_error(supports: dict[str, Support], prefix: str) -> bool:
    return any(
        support.content.removeprefix("Judge verdict error:\n").startswith(prefix)
        for support in supports.values()
        if support.content.startswith("Judge verdict error:\n")
    )


def _reference_ignored(supports: dict[str, Support], claims: tuple[Commitment, ...]) -> bool:
    reference_ids = {
        support.node_id for support in supports.values()
        if support.content.startswith("Reference answer:\n")
    }
    if not reference_ids:
        return False
    cited = {ref for claim in claims for ref in claim.refs}
    return reference_ids.isdisjoint(cited)


__all__ = ["judge_report", "to_markdown"]
