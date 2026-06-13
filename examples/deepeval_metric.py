from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from cmg import ClaimGraph, JsonlStorage, arun_judge, judge_report
from cmg.providers import make_openai_llm_fn


class CmgJudgeMetric(BaseMetric):
    def __init__(
        self,
        *,
        output_dir: Path = Path("cmg-deepeval"),
        threshold: float = 1.0,
    ) -> None:
        self.judge_fn = make_openai_llm_fn("gpt-4o-mini")
        self.output_dir = output_dir
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = None
        self.error = None
        self.async_mode = True
        self.strict_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase) -> float:
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        item_id = test_case.name or hashlib.sha1(test_case.input.encode()).hexdigest()[:12]
        await asyncio.to_thread(self.output_dir.mkdir, parents=True, exist_ok=True)
        graph_path = self.output_dir / f"{item_id}.cmg.jsonl"
        await asyncio.to_thread(graph_path.unlink, missing_ok=True)

        async with ClaimGraph(JsonlStorage(graph_path)) as graph:
            result = await arun_judge(
                graph,
                self.judge_fn,
                prompt=test_case.input,
                candidate_output=test_case.actual_output or "",
                reference_answer=test_case.expected_output or "",
                rubric="Return pass only when the actual output satisfies the expected output.",
                criteria=("Correctness",),
            )
            report = judge_report(graph)

        verdict = result.decision.content if result.decision else "invalid"
        self.score = 1.0 if verdict == "pass" else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"verdict={verdict}; cmg_graph_path={graph_path}; "
            f"cmg_human_review_flags={report['human_review_flags']}"
        )
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "CMG Judge"


test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="The answer should identify Paris.",
        name="capital-france",
    )
]

evaluate(test_cases=test_cases, metrics=[CmgJudgeMetric()])
