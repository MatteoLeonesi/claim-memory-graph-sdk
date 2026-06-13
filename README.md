[![PyPI version](https://img.shields.io/pypi/v/claim-memory-graph.svg)](https://pypi.org/project/claim-memory-graph/)
[![Downloads](https://static.pepy.tech/badge/claim-memory-graph)](https://pepy.tech/project/claim-memory-graph)


# CMG — Claim Memory Graph

A small memory layer for judge and reviewer workflows. CMG tracks what a model reads (evidence), what it concludes (claims), what it decides (decisions), and what it walks back (retractions). When a verdict changes, you can see why.

## Why I Built This

I built CMG while experimenting with LLM code reviewers. I didn't just need the final decision. I needed to know how it was reached. CMG keeps a record of every piece of evidence, every claim, every
decision, and every retraction behind a review. When a verdict flips, you can trace exactly why.

## Install

```bash
pip install claim-memory-graph
```

Optional helpers:

```bash
pip install 'claim-memory-graph[openai]'
pip install 'claim-memory-graph[anthropic]'
```

The package installs as `claim-memory-graph` and imports as `cmg`.

## Quickstart

```python
import asyncio
from pathlib import Path
from cmg import ClaimGraph, JsonlStorage

async def main() -> None:
    async with ClaimGraph(JsonlStorage(Path("review.cmg.jsonl"))) as graph:
        evidence = (await graph.add_support(
            "Unit test test_total fails after the patch"
        )).node
        claim = (await graph.add_commitment(
            "The patch breaks the total calculation",
            refs=(evidence.node_id,),
        )).node
        await graph.add_decision("request_changes", refs=(claim.node_id,))
        print(graph.last_decision())
        print([v.code for v in graph.violations()])

asyncio.run(main())
```

## Integrations

CMG fits into existing evaluation frameworks without touching how they
score or report results.

With **DeepEval**, wrap CMG in a custom metric to log the evidence,
claims, and decisions behind each judge output. See the
[DeepEval adapter guide](docs/user-guide.md#deepeval-adapter).

With **Inspect AI**, add CMG to a custom scorer to capture decision
traces alongside evaluation results. See the
[Inspect AI scorer guide](docs/user-guide.md#inspect-ai-scorer).

## What It Catches
CMG flags verdict flips without retractions, dropped active claims,
unknown references, wrong reference types, and references to already-
invalidated claims. These are signals, not blockers. Your application
decides what to do with them.

## Where It Helps
CMG is useful anywhere an LLM makes decisions that need to be
traceable: code review agents, judge systems, evaluation pipelines,
research review workflows, multi-agent reviewers.

## License

Apache-2.0
