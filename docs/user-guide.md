# CMG user guide

CMG helps you inspect one LLM judge case at a time. For each case it stores the input, the candidate answer, the rubric, the judge's claims, and the final verdict. All of it goes to a JSONL audit log that you can read back with the CLI.

## 1. Install

```bash
pip install claim-memory-graph
```

Optional helpers:

```bash
pip install 'claim-memory-graph[openai]'
pip install 'claim-memory-graph[anthropic]'
```

The import name is `cmg`.

## 2. Try it locally

The demo runs without calling an external model.

```bash
python examples/local_judge_demo.py
```

From there you can view the cases a few different ways. Show every case with its evidence.

```bash
cmg-view cmg-runs/*.cmg.jsonl --show-evidence
```

Get a compact run summary.

```bash
cmg-view cmg-runs/*.cmg.jsonl --summary
```

See only the cases that need human review.

```bash
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

Export the report as JSON.

```bash
cmg-view cmg-runs/*.cmg.jsonl --json
```

## 3. How CMG fits your judge

Your app still owns the goal, the main prompt, and the model call. CMG does not replace any of that. What it adds is a way to tell the judge what to check, using the fields you pass in.

| Field | What it means |
|---|---|
| `prompt` | The task given to the candidate model. |
| `candidate_output` | The answer you want to judge. |
| `reference_answer` | The expected answer, if you have one. |
| `rubric` | The rule the judge should follow. |
| `criteria` | The exact checks the judge should cover. |
| `extra_supports` | Extra facts, logs, tool output, policy text, or source material. |

CMG stores those fields as evidence and then asks the judge to cite them. That citation step is what makes CMG useful, because the judge can no longer just answer `pass`. It has to make claims, and each of those claims has to cite the evidence ids.

## 4. Add CMG to your judge

Create one graph per evaluated item, and use a fresh file for each run. Do not append repeat runs to the same case file.

```python
from pathlib import Path

from cmg import ClaimGraph, JsonlStorage, arun_judge, judge_report


async def judge_fn(messages: list[dict[str, str]]) -> str:
    return await call_your_judge_model(messages)


graph_path = Path("cmg-runs/case-1.cmg.jsonl")

async with ClaimGraph(JsonlStorage(graph_path)) as graph:
    result = await arun_judge(
        graph,
        judge_fn,
        prompt="The task shown to the candidate model.",
        candidate_output="The candidate model answer.",
        reference_answer="The gold answer, if you have one.",
        rubric="The rule used by the judge.",
        criteria=("Correctness", "Completeness"),
        extra_supports={
            "policy": "Any answer with unsupported medical advice should fail.",
        },
        verdicts=("pass", "fail"),
    )

    report = judge_report(graph)
```

Then carry the report into your eval row metadata.

```python
metadata = {
    "cmg_graph_path": str(graph_path),
    "cmg_human_review_flags": report["human_review_flags"],
    "cmg_claims": report["claims"],
}
```

## 5. What your judge sees

CMG sends your judge an ordinary chat message list, with the support ids embedded in it.

```text
[s-abc123] candidate_output
Candidate output:
The answer text...
```

Your `judge_fn` receives that list and passes it to whatever model you already use. CMG does not care which provider that is.

```python
async def judge_fn(messages: list[dict[str, str]]) -> str:
    response = await your_model_client.chat(messages)
    return response.text
```

## 6. What your judge must return

The visible answer has to start with a verdict line.

```text
VERDICT: pass
```

The verdict has to be one of the labels in the `verdicts` list. If it is missing or not allowed, `result.decision` is `None`, and the report includes `missing_verdict` or `invalid_verdict`.

The judge should also add a hidden CMG block.

````text
```cmg
{"ops": [{"op": "commitment", "content": "The answer matches the reference.", "refs": ["s-..."]}]}
```
````

Each `commitment` is one claim, and each claim should cite support ids of the form `s-...`. CMG creates those ids for the prompt, candidate output, reference answer, rubric, and criteria, and your model receives them in the judge prompt.

## 7. What CMG records

CMG uses four node types.

| Node | Meaning |
|---|---|
| `Support` | Evidence. This includes the prompt, answer, reference, rubric, and criteria. |
| `Commitment` | A claim made by the judge. It should cite support ids. |
| `Decision` | The final verdict. CMG records this, not the model. |
| `Invalidation` | A retraction of a prior claim. |

The graph is append-only, so the whole audit trail can be replayed later.

## 8. Read the report

`judge_report(graph)` returns a dict. These are the fields that matter most.

| Field | Meaning |
|---|---|
| `verdict` | The final verdict, or `None`. |
| `claims` | Active claims with evidence ids. |
| `criteria` | Rubric criteria and coverage. |
| `judge_responses` | The visible judge text. |
| `verdict_errors` | Missing or invalid verdict details. |
| `retracted` | Accepted retractions only. |
| `human_review_flags` | Deterministic flags for review. |
| `violations` | Raw graph consistency codes. |

The flags split into two groups. Hard flags are real audit failures. Soft flags are just signals to review. Here are the common ones.

| Flag | Meaning |
|---|---|
| `missing_verdict` | The judge did not return `VERDICT: ...`. |
| `invalid_verdict` | The judge returned a label outside `verdicts`. |
| `uncited_verdict` | The verdict has no active cited claims. |
| `no_supported_claims` | No active claim has valid evidence. |
| `criterion_citation_gap` | A criterion may be covered, but no active claim cited that exact criterion id. |
| `rubric_coverage_gap` | At least one criterion does not appear in any active claim text. |
| `reference_ignored` | A reference answer was not cited. |
| `verdict_flip_without_invalidation` | The verdict changed without a retraction. |
| `silent_commitment_drop` | A later verdict dropped an active claim. |

A flag does not prove the judge is wrong. It tells you where a human should look first.

## 9. Use the CLI

`cmg-view` is the local dashboard for now. Show all logs.

```bash
cmg-view cmg-runs/*.cmg.jsonl
```

Show a summary.

```bash
cmg-view cmg-runs/*.cmg.jsonl --summary
```

Include the evidence.

```bash
cmg-view cmg-runs/*.cmg.jsonl --show-evidence
```

Show only the risky cases.

```bash
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

Export JSON.

```bash
cmg-view cmg-runs/*.cmg.jsonl --json
```

## 10. Production notes

Use a fresh JSONL file for each evaluated item, and a fresh output directory for each full eval run. Keep dataset ids, model ids, and run ids in your eval framework's metadata, and store the CMG path alongside each score. Review the flagged cases like this.

```bash
cmg-view path/to/run/*.cmg.jsonl --flagged-only --show-evidence
```

## 11. What CMG does not do

CMG does not decide whether the judge is correct, and it does not replace human review or your eval harness. What it gives you is a clear audit trail for each judge decision.
