from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path

from cmg import AsyncLLMFn, ClaimGraph, JsonlStorage, Message, arun_judge, judge_report

SUPPORT_RE = re.compile(r"\[(s-[0-9a-f]+)\] ([^\n]+)")


def fake_judge(
    *,
    verdict: str,
    claim: str,
    cited_labels: Sequence[str],
) -> AsyncLLMFn:
    async def judge(messages: list[Message]) -> str:
        text = "\n\n".join(message["content"] for message in messages)
        support_ids = {label: sid for sid, label in SUPPORT_RE.findall(text)}
        refs = [support_ids[label] for label in cited_labels if label in support_ids]
        ops = {"ops": [{"op": "commitment", "content": claim, "refs": refs}]}
        return (
            f"VERDICT: {verdict}\n"
            f"{claim}\n"
            "```cmg\n"
            f"{json.dumps(ops)}\n"
            "```"
        )

    return judge


async def run_case(
    *,
    path: Path,
    prompt: str,
    candidate_output: str,
    reference_answer: str,
    rubric: str,
    criteria: Sequence[str],
    verdict: str,
    claim: str,
    cited_labels: Sequence[str],
) -> None:
    await asyncio.to_thread(path.unlink, missing_ok=True)
    async with ClaimGraph(JsonlStorage(path)) as graph:
        await arun_judge(
            graph,
            fake_judge(verdict=verdict, claim=claim, cited_labels=cited_labels),
            prompt=prompt,
            candidate_output=candidate_output,
            reference_answer=reference_answer,
            rubric=rubric,
            criteria=criteria,
        )
        report = judge_report(graph)
    print(f"{path}: verdict={report['verdict']} flags={report['human_review_flags']}")


async def main() -> None:
    output_dir = Path("cmg-runs")
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

    await run_case(
        path=output_dir / "capital-pass.cmg.jsonl",
        prompt="What is the capital of France?",
        candidate_output="Paris",
        reference_answer="Paris is the capital of France.",
        rubric="Pass only if the answer identifies Paris.",
        criteria=("Correctness",),
        verdict="pass",
        claim="Candidate identifies Paris as the capital of France.",
        cited_labels=("candidate_output", "reference_answer", "rubric", "criterion:1"),
    )
    await run_case(
        path=output_dir / "sky-review.cmg.jsonl",
        prompt="Explain why the sky appears blue.",
        candidate_output="Because it is blue.",
        reference_answer="Rayleigh scattering explains the color.",
        rubric="Pass only if the answer is physically correct and explanatory.",
        criteria=("Correctness", "Explanation quality"),
        verdict="fail",
        claim="Candidate does not explain the physical cause.",
        cited_labels=("candidate_output", "rubric", "criterion:1"),
    )

    print("\nView the audit logs:")
    print("cmg-view cmg-runs/*.cmg.jsonl --show-evidence")
    print("cmg-view cmg-runs/*.cmg.jsonl --flagged-only")


if __name__ == "__main__":
    asyncio.run(main())
