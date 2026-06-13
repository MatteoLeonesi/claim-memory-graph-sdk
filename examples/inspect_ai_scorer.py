from __future__ import annotations

import asyncio
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from cmg import ClaimGraph, JsonlStorage, arun_judge, judge_report
from cmg.providers import make_openai_llm_fn


@scorer(metrics=[accuracy()])
def cmg_judge_scorer(output_dir: str = "cmg-inspect"):
    judge_fn = make_openai_llm_fn("gpt-4o-mini")

    async def score(state: TaskState, target: Target) -> Score:
        output_path = Path(output_dir)
        await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)
        graph_path = output_path / f"{state.sample_id}.cmg.jsonl"
        await asyncio.to_thread(graph_path.unlink, missing_ok=True)
        candidate_output = state.output.completion if state.output else ""

        async with ClaimGraph(JsonlStorage(graph_path)) as graph:
            result = await arun_judge(
                graph,
                judge_fn,
                prompt=state.input_text,
                candidate_output=candidate_output,
                reference_answer=target.text,
                rubric="Return pass only when the answer satisfies the target.",
                criteria=("Correctness",),
            )
            report = judge_report(graph)

        verdict = result.decision.content if result.decision else "invalid"
        return Score(
            value=CORRECT if verdict == "pass" else INCORRECT,
            answer=verdict,
            explanation=result.visible_text,
            metadata={
                "cmg_graph_path": str(graph_path),
                "cmg_human_review_flags": report["human_review_flags"],
                "cmg_claims": report["claims"],
            },
        )

    return score


@task
def judged_capitals() -> Task:
    return Task(
        dataset=[
            Sample(
                input="What is the capital of France?",
                target="The answer should identify Paris.",
            )
        ],
        solver=generate(),
        scorer=cmg_judge_scorer(),
    )
