from __future__ import annotations

from pathlib import Path

import pytest

from cmg import ClaimGraph, JsonlStorage
from cmg.cli import _amain, _parser


async def test_cmg_view_renders_audit_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "case.jsonl"
    async with ClaimGraph(JsonlStorage(path)) as graph:
        support = (await graph.add_support("Rubric:\nanswer must be correct")).node
        claim = (await graph.add_commitment("Candidate is correct", refs=(support.node_id,))).node
        await graph.add_support("Judge response:\nVERDICT: pass\nThe answer is correct.")
        await graph.add_decision("pass", refs=(claim.node_id,))

    args = _parser().parse_args([str(path), "--color", "never", "--show-evidence"])
    code = await _amain(args)
    out = capsys.readouterr().out

    assert code == 0
    assert "CMG Judge Audit" in out
    assert "(o.o)" in out
    assert "(> <)" in out
    assert "case" in out
    assert "verdict: pass" in out
    assert "Candidate is correct" in out
    assert "No consistency violations" in out
    assert "Rubric: answer must be correct" in out


async def test_cmg_view_flagged_only_filters_clean_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean = tmp_path / "clean.jsonl"
    async with ClaimGraph(JsonlStorage(clean)) as graph:
        support = (await graph.add_support("Rubric:\nanswer must be correct")).node
        claim = (await graph.add_commitment("Candidate is correct", refs=(support.node_id,))).node
        await graph.add_decision("pass", refs=(claim.node_id,))

    flagged = tmp_path / "flagged.jsonl"
    async with ClaimGraph(JsonlStorage(flagged)) as graph:
        await graph.add_support("Rubric:\nanswer must be correct")
        await graph.add_decision("fail", refs=())

    args = _parser().parse_args([str(tmp_path / "*.jsonl"), "--flagged-only", "--color", "never"])
    code = await _amain(args)
    out = capsys.readouterr().out

    assert code == 0
    assert "1 flagged" in out
    assert "flagged" in out
    assert "clean" not in out


async def test_cmg_view_summary_renders_run_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean = tmp_path / "clean.jsonl"
    async with ClaimGraph(JsonlStorage(clean)) as graph:
        support = (await graph.add_support("Rubric:\nanswer must be correct")).node
        claim = (await graph.add_commitment("Candidate is correct", refs=(support.node_id,))).node
        await graph.add_decision("pass", refs=(claim.node_id,))

    flagged = tmp_path / "flagged.jsonl"
    async with ClaimGraph(JsonlStorage(flagged)) as graph:
        await graph.add_support("Rubric:\nanswer must be correct")
        await graph.add_decision("fail", refs=())

    soft = tmp_path / "soft.jsonl"
    async with ClaimGraph(JsonlStorage(soft)) as graph:
        candidate = (await graph.add_support("Candidate output:\nA clear answer")).node
        await graph.add_support("Criterion:\nClarity")
        claim = (await graph.add_commitment(
            "The answer is clear and easy to follow.",
            refs=(candidate.node_id,),
        )).node
        await graph.add_decision("pass", refs=(claim.node_id,))

    args = _parser().parse_args([str(tmp_path / "*.jsonl"), "--summary", "--color", "never"])
    code = await _amain(args)
    out = capsys.readouterr().out

    assert code == 0
    assert "Summary" in out
    assert "cases: 3 loaded" in out
    assert "fail" in out
    assert "pass" in out
    assert "uncited_verdict" in out
    assert "criterion_citation_gap" in out
    assert "[Hard flags]" in out
    assert "[Soft flags]" in out
    assert "-- Judge response --" not in out
