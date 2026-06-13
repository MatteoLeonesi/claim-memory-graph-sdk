from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from openai import AsyncOpenAI

from cmg import ClaimGraph, JsonlStorage, Message, arun_judge, judge_report

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LIMIT = int(os.getenv("CMG_LIMIT", "20"))
OUT = Path(os.getenv("CMG_OUT", "cmg-runs/deepseek-mt-bench"))
URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


async def load_questions() -> list[dict[str, Any]]:
    def fetch() -> list[dict[str, Any]]:
        with urlopen(URL, timeout=30) as response:
            lines = response.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines[:LIMIT]]

    return await asyncio.to_thread(fetch)


async def chat(client: AsyncOpenAI, messages: list[Message], max_tokens: int) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("set DEEPSEEK_API_KEY first")

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    await asyncio.to_thread(OUT.mkdir, parents=True, exist_ok=True)
    rows = []

    for index, question in enumerate(await load_questions(), start=1):
        prompt = str(question["turns"][0])
        candidate = await chat(
            client,
            [
                {"role": "system", "content": "Answer clearly and concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )

        graph_path = OUT / f"{index:02d}-{question['question_id']}.cmg.jsonl"
        await asyncio.to_thread(graph_path.unlink, missing_ok=True)

        async def judge_fn(messages: list[Message]) -> str:
            return await chat(client, messages, max_tokens=900)

        async with ClaimGraph(JsonlStorage(graph_path)) as graph:
            result = await arun_judge(
                graph,
                judge_fn,
                prompt=prompt,
                candidate_output=candidate,
                rubric=(
                    "Score the candidate answer from 1 to 10 for the user request. "
                    "Use 10 for excellent, complete, clear, and instruction-following answers. "
                    "Use 1 for useless or unsafe answers."
                ),
                criteria=("Instruction following", "Helpfulness", "Completeness", "Clarity"),
                verdicts=tuple(str(n) for n in range(1, 11)),
            )
            report = judge_report(graph)

        row = {
            "question_id": question["question_id"],
            "category": question["category"],
            "score": result.decision.content if result.decision else None,
            "flags": report["human_review_flags"],
            "graph": str(graph_path),
        }
        rows.append(row)
        print(f"{index:02d} q={row['question_id']} score={row['score']} flags={row['flags']}")

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flagged = sum(bool(row["flags"]) for row in rows)
    print(f"\nsummary={summary}")
    print(f"flagged={flagged}/{len(rows)}")
    print(f"view=cmg-view {OUT}/*.cmg.jsonl --flagged-only --show-evidence")


if __name__ == "__main__":
    asyncio.run(main())
