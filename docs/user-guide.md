# CMG user guide

CMG helps you inspect one LLM judge case at a time.

It stores the input, the candidate answer, the rubric, the judge claims, and the
final verdict.

It writes a JSONL audit log.

You can read that log with the CLI.

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

## 2. Try It Locally

Run the demo.

It does not call an external model.

```bash
python examples/local_judge_demo.py
```

View all cases:

```bash
cmg-view cmg-runs/*.cmg.jsonl --show-evidence
```

View a compact run summary:

```bash
cmg-view cmg-runs/*.cmg.jsonl --summary
```

View only cases that need human review:

```bash
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

Export report JSON:

```bash
cmg-view cmg-runs/*.cmg.jsonl --json
```

## 3. How CMG Fits Your Judge

Your app still owns the goal.

Your app still owns the main prompt.

Your app still owns the model call.

CMG does not replace those parts.

CMG helps you tell the judge what to verify.

You pass these fields:

| Field | What it means |
|---|---|
| `prompt` | The task given to the candidate model. |
| `candidate_output` | The answer you want to judge. |
| `reference_answer` | The expected answer, if you have one. |
| `rubric` | The rule the judge should follow. |
| `criteria` | The exact checks the judge should cover. |
| `extra_supports` | Extra facts, logs, tool output, policy text, or source material. |

CMG stores those fields as evidence.

Then it asks the judge to cite that evidence.

That is the main value.

The judge cannot just say `pass`.

It must make claims.

Those claims must cite the evidence ids.

## 4. Add CMG To Your Judge

Create one graph per evaluated item.

Use a fresh file for each item run.

Do not append repeat runs to the same case file.

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

Use the report in your eval row metadata.

```python
metadata = {
    "cmg_graph_path": str(graph_path),
    "cmg_human_review_flags": report["human_review_flags"],
    "cmg_claims": report["claims"],
}
```

## 5. What Your Judge Sees

CMG sends your judge a normal chat message list.

The message contains support ids.

They look like this:

```text
[s-abc123] candidate_output
Candidate output:
The answer text...
```

Your `judge_fn` receives that message list.

You pass it to the model you already use.

CMG does not care which provider you use.

```python
async def judge_fn(messages: list[dict[str, str]]) -> str:
    response = await your_model_client.chat(messages)
    return response.text
```

## 6. What Your Judge Must Return

The visible answer must start with a verdict line.

```text
VERDICT: pass
```

The verdict must be in the `verdicts` list.

If it is missing, `result.decision` is `None`.

If it is not allowed, `result.decision` is also `None`.

The report will include `missing_verdict` or `invalid_verdict`.

The judge should also emit a hidden CMG block.

````text
```cmg
{"ops": [{"op": "commitment", "content": "The answer matches the reference.", "refs": ["s-..."]}]}
```
````

Each `commitment` is one claim.

Each claim should cite support ids.

Support ids look like `s-...`.

CMG creates them for the prompt, candidate output, reference answer, rubric, and
criteria.

Your model receives those ids in the judge prompt.

## 7. What CMG Records

CMG uses four node types.

| Node | Meaning |
|---|---|
| `Support` | Evidence. This includes the prompt, answer, reference, rubric, and criteria. |
| `Commitment` | A claim made by the judge. It should cite support ids. |
| `Decision` | The final verdict. CMG records this, not the model. |
| `Invalidation` | A retraction of a prior claim. |

The graph is append-only.

That means the audit trail can be replayed later.

## 8. Read The Report

`judge_report(graph)` returns a dict.

Important fields:

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

Flags are split into two groups.

Hard flags are structural audit failures.

Soft flags are review signals.

Common flags:

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

Flags do not prove the judge is wrong.

They tell you where a human should look first.

## 9. Use The CLI

`cmg-view` is the local dashboard for now.

Show all logs:

```bash
cmg-view cmg-runs/*.cmg.jsonl
```

Show a summary:

```bash
cmg-view cmg-runs/*.cmg.jsonl --summary
```

Show the evidence too:

```bash
cmg-view cmg-runs/*.cmg.jsonl --show-evidence
```

Show only risky cases:

```bash
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

Export JSON:

```bash
cmg-view cmg-runs/*.cmg.jsonl --json
```

## 10. Production Notes

Use a fresh JSONL file for each evaluated item.

Use a fresh output directory for each full eval run.

Keep dataset ids, model ids, and run ids in your eval framework metadata.

Store the CMG path with each score.

Then review flagged cases with:

```bash
cmg-view path/to/run/*.cmg.jsonl --flagged-only --show-evidence
```

## 11. What CMG Does Not Do

CMG does not decide whether the judge is correct.

CMG does not replace human review.

CMG does not replace your eval harness.

It gives you a clear audit trail for each judge decision.
