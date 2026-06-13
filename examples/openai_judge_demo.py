from __future__ import annotations

import asyncio
from pathlib import Path

from cmg import ClaimGraph, JsonlStorage, arun_judge, judge_report
from cmg.providers import make_openai_llm_fn


async def main() -> None:
    judge_fn = make_openai_llm_fn("gpt-4o-mini")
    graph_path = Path("cmg-runs/openai-demo.cmg.jsonl")
    await asyncio.to_thread(graph_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(graph_path.unlink, missing_ok=True)

    async with ClaimGraph(JsonlStorage(graph_path)) as graph:
        result = await arun_judge(
            graph,
            judge_fn,
            prompt="What is the capital of France?",
            candidate_output="Paris",
            reference_answer="Paris is the capital of France.",
            rubric="The answer must identify Paris and avoid unrelated facts.",
            criteria=("Correctness", "Concision"),
        )
        print(result.visible_text)
        print(f"Verdict: {result.decision.content if result.decision else '(invalid)'}")
        print(judge_report(graph))
        print(f"\nAudit log: {graph_path}")
        print(f"View it with: cmg-view {graph_path}")


if __name__ == "__main__":
    asyncio.run(main())
