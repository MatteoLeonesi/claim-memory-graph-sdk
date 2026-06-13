from __future__ import annotations

import json
import re
from pathlib import Path

from cmg import (
    INVALIDATION_ACCEPTED,
    INVALIDATION_REJECTED,
    ClaimGraph,
    Decision,
    JsonlStorage,
    Message,
    active_claims,
    arun_judge,
    judge_report,
    to_markdown,
)
from cmg.judge import extract_verdict

_SUPPORT_RE = re.compile(r"\[(s-[0-9a-f]+)\] ([^\n]+)")


def _cmg(ops: list[dict[str, object]]) -> str:
    return "```cmg\n" + json.dumps({"ops": ops}) + "\n```"


def _support_ids(messages: list[Message]) -> dict[str, str]:
    text = "\n\n".join(message["content"] for message in messages)
    return {label: sid for sid, label in _SUPPORT_RE.findall(text)}


async def test_active_claims_gate_dedup_and_retire(tmp_path: Path) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        support = (await graph.add_support("Rubric:\nanswer must name Paris")).node
        good1 = (await graph.add_commitment("Candidate names Paris", refs=(support.node_id,))).node
        good2 = (await graph.add_commitment("Candidate names Paris", refs=(support.node_id,))).node
        await graph.add_commitment("uncited claim", refs=())

        assert active_claims(graph) == (good2,)

        await graph.add_invalidation(
            previous_commitment=good1.node_id,
            previous_support=(support.node_id,),
            new_information="The claim was superseded by a stricter read.",
            contrast_test="Check the active duplicate claim.",
            result=INVALIDATION_ACCEPTED,
        )

        assert active_claims(graph) == ()


async def test_active_claims_keeps_rejected_invalidation_active(tmp_path: Path) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        support = (await graph.add_support("Rubric:\nanswer must name Paris")).node
        claim = (await graph.add_commitment("Candidate names Paris", refs=(support.node_id,))).node

        await graph.add_invalidation(
            previous_commitment=claim.node_id,
            previous_support=(support.node_id,),
            new_information="Counterargument was checked and did not hold.",
            contrast_test="Compare the answer with the reference.",
            result=INVALIDATION_REJECTED,
        )

        assert graph.active_commitments() == (claim,)
        assert active_claims(graph) == (claim,)
        assert "no_supported_claims" not in judge_report(graph)["human_review_flags"]


async def test_arun_judge_records_generic_verdict_and_claims(tmp_path: Path) -> None:
    captured: list[list[Message]] = []

    async def llm(messages: list[Message]) -> str:
        captured.append(messages)
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["rubric"], ids["criterion:1"]]
        return (
            "VERDICT: pass\n"
            "The candidate gives the expected answer.\n"
            + _cmg([
                {
                    "op": "commitment",
                    "content": "Candidate identifies Paris as the capital.",
                    "refs": refs,
                }
            ])
        )

    async with ClaimGraph(JsonlStorage(tmp_path / "judge.jsonl")) as graph:
        result = await arun_judge(
            graph,
            llm,
            prompt="What is the capital of France?",
            candidate_output="Paris",
            reference_answer="Paris is the capital of France.",
            rubric="The answer must identify Paris.",
            criteria=("Correctness",),
        )

    assert result.decision.content == "pass"
    assert len(result.commitments) == 1
    assert result.decision.refs == (result.commitments[0].node_id,)
    assert result.violations == ()
    assert result.support_ids["judge_response"].startswith("s-")
    prompt_text = "\n".join(message["content"] for message in captured[0])
    assert "evidence-grounded LLM judge" in prompt_text
    assert "candidate_output" in prompt_text


async def test_arun_judge_flags_missing_verdict_without_decision(tmp_path: Path) -> None:
    async def llm(messages: list[Message]) -> str:
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["rubric"]]
        return (
            "The candidate gives the expected answer.\n"
            + _cmg([
                {
                    "op": "commitment",
                    "content": "Candidate identifies Paris as the capital.",
                    "refs": refs,
                }
            ])
        )

    async with ClaimGraph(JsonlStorage(tmp_path / "judge.jsonl")) as graph:
        result = await arun_judge(
            graph,
            llm,
            prompt="What is the capital of France?",
            candidate_output="Paris",
            rubric="The answer must identify Paris.",
        )
        report = judge_report(graph)

    assert result.decision is None
    assert graph.last_decision() is None
    assert "missing_verdict" in report["human_review_flags"]
    assert report["verdict_errors"]
    assert report["verdict_errors"][0]["content"] == "missing_verdict"


async def test_arun_judge_flags_invalid_verdict_without_decision(tmp_path: Path) -> None:
    async def llm(messages: list[Message]) -> str:
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["rubric"]]
        return (
            "VERDICT: maybe\n"
            "The candidate may be correct.\n"
            + _cmg([
                {
                    "op": "commitment",
                    "content": "Candidate identifies Paris as the capital.",
                    "refs": refs,
                }
            ])
        )

    async with ClaimGraph(JsonlStorage(tmp_path / "judge.jsonl")) as graph:
        result = await arun_judge(
            graph,
            llm,
            prompt="What is the capital of France?",
            candidate_output="Paris",
            rubric="The answer must identify Paris.",
            verdicts=("pass", "fail"),
        )
        report = judge_report(graph)

    assert result.decision is None
    assert graph.last_decision() is None
    assert "invalid_verdict" in report["human_review_flags"]
    assert any(warning.startswith("invalid_verdict: maybe") for warning in result.parse_warnings)


async def test_arun_judge_ignores_model_decision_ops(tmp_path: Path) -> None:
    async def llm(messages: list[Message]) -> str:
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["rubric"]]
        return (
            "VERDICT: pass\n"
            "The candidate gives the expected answer.\n"
            + _cmg([
                {
                    "op": "decision",
                    "content": "fail",
                    "refs": [],
                },
                {
                    "op": "commitment",
                    "content": "Candidate identifies Paris as the capital.",
                    "refs": refs,
                },
            ])
        )

    async with ClaimGraph(JsonlStorage(tmp_path / "judge.jsonl")) as graph:
        result = await arun_judge(
            graph,
            llm,
            prompt="What is the capital of France?",
            candidate_output="Paris",
            rubric="The answer must identify Paris.",
        )
        report = judge_report(graph)

    decisions = [node for node in graph.nodes() if isinstance(node, Decision)]
    assert len(decisions) == 1
    assert decisions[0] == result.decision
    assert decisions[0].content == "pass"
    assert "ignored disallowed op kind: 'decision'" in result.parse_warnings
    assert "verdict_flip_without_invalidation" not in report["human_review_flags"]


async def test_judge_report_flags_missing_criterion_and_reference(tmp_path: Path) -> None:
    async def llm(messages: list[Message]) -> str:
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["rubric"], ids["criterion:1"]]
        return (
            "VERDICT: fail\n"
            "The answer is incomplete.\n"
            + _cmg([
                {
                    "op": "commitment",
                    "content": "Candidate omits the requested explanation.",
                    "refs": refs,
                }
            ])
        )

    async with ClaimGraph(JsonlStorage(tmp_path / "judge.jsonl")) as graph:
        await arun_judge(
            graph,
            llm,
            prompt="Explain why the sky appears blue.",
            candidate_output="Because it is blue.",
            reference_answer="Rayleigh scattering explains the color.",
            rubric="Reward physical correctness and explanation quality.",
            criteria=("Correctness", "Explanation quality"),
        )
        report = judge_report(graph)
        markdown = to_markdown(graph)

    assert report["verdict"] == "fail"
    assert "criterion_citation_gap" in report["human_review_flags"]
    assert "rubric_coverage_gap" not in report["human_review_flags"]
    assert "reference_ignored" in report["human_review_flags"]
    assert "# Judge audit report" in markdown


async def test_judge_report_exact_criterion_citation_has_no_coverage_flags(
    tmp_path: Path,
) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        candidate = (await graph.add_support("Candidate output:\nParis")).node
        criterion = (await graph.add_support("Criterion:\nCorrectness")).node
        claim = (await graph.add_commitment(
            "Candidate correctly identifies Paris.",
            refs=(candidate.node_id, criterion.node_id),
        )).node
        await graph.add_decision("pass", refs=(claim.node_id,))
        report = judge_report(graph)

    assert report["criteria"][0]["citation_covered"] is True
    assert report["criteria"][0]["covered"] is True
    assert "criterion_citation_gap" not in report["human_review_flags"]
    assert "rubric_coverage_gap" not in report["human_review_flags"]


async def test_judge_report_discussed_criterion_without_id_is_citation_gap_only(
    tmp_path: Path,
) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        candidate = (await graph.add_support("Candidate output:\nA structured answer")).node
        await graph.add_support("Criterion:\nClarity")
        claim = (await graph.add_commitment(
            "The answer is clear and easy to follow.",
            refs=(candidate.node_id,),
        )).node
        await graph.add_decision("pass", refs=(claim.node_id,))
        report = judge_report(graph)

    assert report["criteria"][0]["citation_covered"] is False
    assert report["criteria"][0]["covered"] is True
    assert "criterion_citation_gap" in report["human_review_flags"]
    assert "rubric_coverage_gap" not in report["human_review_flags"]


async def test_judge_report_undiscussed_criterion_has_citation_and_coverage_gap(
    tmp_path: Path,
) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        candidate = (await graph.add_support("Candidate output:\nA short answer")).node
        await graph.add_support("Criterion:\nFactual accuracy")
        claim = (await graph.add_commitment(
            "The answer is concise.",
            refs=(candidate.node_id,),
        )).node
        await graph.add_decision("pass", refs=(claim.node_id,))
        report = judge_report(graph)

    assert report["criteria"][0]["citation_covered"] is False
    assert report["criteria"][0]["covered"] is False
    assert "criterion_citation_gap" in report["human_review_flags"]
    assert "rubric_coverage_gap" in report["human_review_flags"]


async def test_judge_report_retractions_only_include_accepted_invalidations(
    tmp_path: Path,
) -> None:
    async with ClaimGraph(JsonlStorage(tmp_path / "log.jsonl")) as graph:
        support = (await graph.add_support("Rubric:\nanswer must name Paris")).node
        rejected = (await graph.add_commitment(
            "Candidate names Paris",
            refs=(support.node_id,),
        )).node
        accepted = (await graph.add_commitment(
            "Candidate gives a short answer",
            refs=(support.node_id,),
        )).node

        await graph.add_invalidation(
            previous_commitment=rejected.node_id,
            previous_support=(support.node_id,),
            new_information="The challenge was checked and rejected.",
            contrast_test="Compare the candidate with the rubric.",
            result=INVALIDATION_REJECTED,
        )
        await graph.add_invalidation(
            previous_commitment=accepted.node_id,
            previous_support=(support.node_id,),
            new_information="The answer also needs an explanation.",
            contrast_test="Check whether brevity is enough.",
            result=INVALIDATION_ACCEPTED,
        )
        report = judge_report(graph)
        markdown = to_markdown(graph)

    assert report["retracted"] == [
        {
            "commitment": accepted.node_id,
            "new_information": "The answer also needs an explanation.",
            "contrast_test": "Check whether brevity is enough.",
            "result": INVALIDATION_ACCEPTED,
        }
    ]
    assert rejected.node_id not in markdown
    assert accepted.node_id in markdown


async def test_judge_report_replays_from_jsonl(tmp_path: Path) -> None:
    async def llm(messages: list[Message]) -> str:
        ids = _support_ids(messages)
        refs = [ids["candidate_output"], ids["reference_answer"], ids["rubric"]]
        return (
            "VERDICT: pass\n"
            "The answer matches the reference.\n"
            + _cmg([
                {
                    "op": "commitment",
                    "content": "Candidate matches the reference answer.",
                    "refs": refs,
                }
            ])
        )

    path = tmp_path / "judge.jsonl"
    async with ClaimGraph(JsonlStorage(path)) as graph:
        await arun_judge(
            graph,
            llm,
            prompt="2 + 2?",
            candidate_output="4",
            reference_answer="4",
            rubric="The answer should be exactly 4.",
        )
        original = judge_report(graph)

    replayed = await ClaimGraph.aload(JsonlStorage(path))
    assert judge_report(replayed) == original


def test_extract_verdict() -> None:
    assert extract_verdict("VERDICT: PASS\nok") == "pass"
    assert extract_verdict("No verdict") == "unknown"
