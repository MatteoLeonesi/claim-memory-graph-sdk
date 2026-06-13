[![PyPI version](https://img.shields.io/pypi/v/claim-memory-graph.svg)](https://pypi.org/project/claim-memory-graph/)
[![Downloads](https://static.pepy.tech/badge/claim-memory-graph)](https://pepy.tech/project/claim-memory-graph)

# CMG - Claim Memory Graph

<p align="center">
  <img src="docs/assets/banner.png" alt="Claim Memory Graph banner" width="100%">
</p>

<p align="center">
  <strong>A small audit layer for LLM-as-a-judge systems.</strong>
</p>

CMG records what an LLM judge saw.

It records the claims the judge made.

It records the verdict it gave.

It also records when a claim was later challenged or retracted.

The goal is simple: make judge output easier for a human to check.

## Why

LLM judges are useful, but they are not neutral.

Research has shown common failure modes:

- Zheng et al. report position bias, verbosity bias, self-enhancement bias, and
  limited reasoning.
- Li et al. show scoring bias from rubric order, score ids, and reference
  answer scoring.
- Feng et al. show that explicit rubrics and criteria can help judge
  consistency, but do not solve it.
- Wang et al. show weak evidence verification in research-agent judging.
- Chen et al. show reliability gaps for long-form outputs, even when rubrics or
  references are present.

CMG does not remove these problems.

It makes them easier to see.

It helps you tell the judge what to verify.

You pass the task, answer, reference, rubric, and criteria.

CMG turns them into cited evidence.

It then asks the judge to make claims against that evidence.

It forces each valid verdict to point to active claims.

Each claim must point to evidence.

The viewer then shows missing evidence, ignored references, uncovered rubric
items, invalid verdicts, and unsafe verdict changes.

Today the local viewer is the dashboard:

```bash
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

A web dashboard can use the same report data later.

## Install

```bash
pip install claim-memory-graph
```

Optional provider helpers:

```bash
pip install 'claim-memory-graph[openai]'
pip install 'claim-memory-graph[anthropic]'
```

The package name is `claim-memory-graph`.

The import name is `cmg`.

The core package has no runtime dependencies.

## Quickstart

Run the local demo first.

It needs no API key.

```bash
python examples/local_judge_demo.py
cmg-view cmg-runs/*.cmg.jsonl --summary
cmg-view cmg-runs/*.cmg.jsonl --show-evidence
cmg-view cmg-runs/*.cmg.jsonl --flagged-only
```

The CLI starts with this header:

```text
  ()_()  CMG Judge Audit
  (o.o)  claim memory graph
  (> <)
```

Then add CMG to your own judge.

You own the main task.

You own the rubric.

CMG only adds the audit layer.

```python
from pathlib import Path

from cmg import ClaimGraph, JsonlStorage, arun_judge, judge_report


async def judge_fn(messages):
    return await call_your_judge_model(messages)


async with ClaimGraph(JsonlStorage(Path("cmg-runs/case-1.cmg.jsonl"))) as graph:
    result = await arun_judge(
        graph,
        judge_fn,
        prompt="Question shown to the candidate model.",
        candidate_output="Candidate model answer.",
        reference_answer="Optional gold answer.",
        rubric="How the judge should decide.",
        criteria=("Correctness", "Completeness"),
        verdicts=("pass", "fail"),
    )

    report = judge_report(graph)

if result.decision is None:
    print("The judge returned a missing or invalid verdict.")
else:
    print(result.decision.content)

print(report["human_review_flags"])
```

## What The Judge Must Return

The judge must start with a verdict line:

```text
VERDICT: pass
```

The judge should also add a hidden CMG block with claims:

````text
```cmg
{"ops": [{"op": "commitment", "content": "The answer matches the reference.", "refs": ["s-..."]}]}
```
````

CMG records the final `Decision` itself.

If the model emits a `decision` op, `arun_judge` ignores it.

If the model returns `maybe` when only `pass` and `fail` are allowed, CMG does
not record a decision.

The report marks the case for human review.

## What You Get

`judge_report(graph)` returns:

- `verdict`
- `claims`
- `criteria`
- `judge_responses`
- `verdict_errors`
- `retracted`
- `human_review_flags`
- `violations`

Flags are split into two groups.

Hard flags are structural audit failures.

Soft flags are review signals.

Useful flags:

| Flag | Meaning |
|---|---|
| `missing_verdict` | The judge did not return a valid verdict line. |
| `invalid_verdict` | The verdict was not in the allowed list. |
| `uncited_verdict` | A verdict has no active cited claims. |
| `no_supported_claims` | No active claim has valid evidence. |
| `criterion_citation_gap` | A criterion was discussed or may be covered, but no active claim cited that exact criterion id. |
| `rubric_coverage_gap` | A criterion does not appear to be covered by any active claim text. |
| `reference_ignored` | A reference answer exists, but no active claim cites it. |
| `verdict_flip_without_invalidation` | A verdict changed without retracting old claims first. |
| `silent_commitment_drop` | A later decision dropped an active claim without a retraction. |

## Integrations

CMG does not replace your eval framework.

It sits inside it.

Use your framework for datasets, model calls, scores, and aggregation.

Use CMG for per-case audit logs.

Examples:

- `examples/openai_judge_demo.py`
- `examples/inspect_ai_scorer.py`
- `examples/deepeval_metric.py`

Use a fresh output file for each case run.

Do not append many runs of the same case to the same JSONL file.

## Docs

| Topic | Link |
|---|---|
| User guide | [docs/user-guide.md](docs/user-guide.md) |
| Developer guide | [docs/dev-guide.md](docs/dev-guide.md) |
| Release checklist | [docs/release.md](docs/release.md) |

## Sources

- [Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Li et al., Evaluating Scoring Bias in LLM-as-a-Judge](https://arxiv.org/abs/2506.22316)
- [Feng et al., Are We on the Right Way to Assessing LLM-as-a-Judge?](https://arxiv.org/abs/2512.16041)
- [Wang et al., Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents?](https://arxiv.org/abs/2605.19196)
- [Chen et al., Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation](https://arxiv.org/abs/2606.01629)

## License

Apache-2.0.
