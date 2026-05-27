# cmg SDK user guide

This guide is for application developers integrating `cmg` as a lightweight
memory and observability layer for LLM judges, reviewers, and agents that make
verdicts over evidence.

Use it when you want to inspect what a model claimed, which evidence it cited,
which verdict it selected, and whether a later verdict changed without an
explicit retraction.

## Install

```bash
pip install claim-memory-graph

# Optional provider helpers:
pip install 'claim-memory-graph[openai]'
pip install 'claim-memory-graph[anthropic]'
pip install 'claim-memory-graph[all]'
```

The PyPI distribution is `claim-memory-graph`; the Python import package is `cmg`.
The core package has no runtime dependencies beyond Python 3.10+.

## Mental model

`cmg` records an append-only graph of what the model claimed, decided, and
retracted. It observes problems by returning `Violation` records; it does not
block or rewrite the model response.

The four graph node types are:

| Node | Created by | Refs should point to |
|---|---|---|
| `Support` | `graph.add_support(...)` or a model `support` op | Optional external refs |
| `Commitment` | `graph.add_commitment(...)` or a model `commitment` op | Existing `s-...` support IDs |
| `Decision` | `graph.add_decision(...)` or a model `decision` op | Existing active `k-...` commitment IDs |
| `Invalidation` | `graph.add_invalidation(...)` or a model `invalidation` op | An active `k-...` commitment and its cited `s-...` support IDs |

Refs must already exist when an op is applied. The SDK mints IDs during ingest,
so a model cannot create a support and cite that freshly minted support ID later
in the same annotation block unless your application already knows that ID.
Pre-seed supports from application evidence, or let the model create support in
one turn and cite it in a later turn after your app has shown the ID.

For judge and reviewer systems, the usual shape is:

1. Store rubric items, candidate answers, diffs, logs, or tool results as
   `Support` nodes.
2. Ask the model for explicit `Commitment` nodes tied to those support IDs.
3. Record a `Decision` such as `pass`, `fail`, `approve`, `request_changes`, or
   a domain-specific verdict.
4. Watch for `Violation` records when a later verdict no longer lines up with
   the active commitments.

## Basic async integration

```python
from pathlib import Path
from cmg import ClaimGraph, JsonlStorage, arun_turn, build_annotation_system_prompt

async with ClaimGraph(JsonlStorage(Path("conversation.jsonl"))) as graph:
    support = (await graph.add_support("log line 42 shows timeout")).node

    async def llm(messages):
        ...

    result = await arun_turn(
        graph,
        llm,
        [
            {"role": "system", "content": build_annotation_system_prompt()},
            {"role": "user", "content": f"Evidence {support.node_id}: {support.content}"},
        ],
    )

    visible_text = result.visible_text
    violations = result.violations()
```

`result.visible_text` is the model response with `cmg` annotation blocks removed.
`result.raw_text` keeps the original response for debugging or audit trails.
`result.parse_warnings` contains malformed annotation warnings.

## Use CMG inside an eval harness

In an eval, `cmg` should sit around the judge, not around the model being
evaluated. Your benchmark, gold labels, scoring rules, and aggregate metrics stay
where they are. `cmg` adds an audit trail for the judge's reasoning surface:
which evidence it cited, which commitments it made, which verdict it recorded,
and whether a later pass changed verdict without an explicit invalidation.

The common setup is one graph per evaluated item:

- `Support`: prompt, candidate output, reference answer, rubric, tool results, or
  any evidence the judge may cite.
- `Commitment`: short claims the judge makes while explaining its verdict.
- `Decision`: the verdict your harness records for the item.
- `Violation`: extra telemetry attached to the eval row.

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from cmg import ClaimGraph, JsonlStorage, arun_turn, build_annotation_system_prompt

JudgeFn = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass(frozen=True)
class EvalCase:
    item_id: str
    prompt: str
    candidate_model: str
    candidate_output: str
    reference_answer: str
    rubric: str


def extract_verdict(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            return line.removeprefix("VERDICT:").strip().casefold()
    return "unknown"


async def evaluate_case(case: EvalCase, judge_fn: JudgeFn, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"{case.item_id}.cmg.jsonl"

    async with ClaimGraph(JsonlStorage(graph_path)) as graph:
        prompt = (await graph.add_support(f"Prompt:\n{case.prompt}")).node
        candidate = (await graph.add_support(
            f"Candidate output from {case.candidate_model}:\n{case.candidate_output}"
        )).node
        reference = (await graph.add_support(
            f"Reference answer:\n{case.reference_answer}"
        )).node
        rubric = (await graph.add_support(f"Rubric:\n{case.rubric}")).node

        observed_violations = []

        judge_result = await arun_turn(
            graph,
            judge_fn,
            [
                {"role": "system", "content": build_annotation_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        "Evaluate the candidate output.\n"
                        "Return one line formatted as `VERDICT: pass` or `VERDICT: fail`.\n"
                        "Then explain briefly.\n"
                        "Add cmg commitment ops for the claims that justify your verdict.\n"
                        "Each commitment must cite the relevant support IDs.\n\n"
                        f"{prompt.node_id}: {prompt.content}\n"
                        f"{candidate.node_id}: {candidate.content}\n"
                        f"{reference.node_id}: {reference.content}\n"
                        f"{rubric.node_id}: {rubric.content}"
                    ),
                },
            ],
            on_violation=observed_violations.append,
        )

        commitment_ids = [
            result.node.node_id
            for result in judge_result.applied
            if result.node.kind == "commitment"
        ]
        verdict = extract_verdict(judge_result.visible_text)

        decision = await graph.add_decision(verdict, refs=commitment_ids)
        observed_violations.extend(decision.violations)

        return {
            "item_id": case.item_id,
            "candidate_model": case.candidate_model,
            "judge_verdict": verdict,
            "judge_visible_text": judge_result.visible_text,
            "cmg_graph_path": str(graph_path),
            "cmg_commitment_ids": commitment_ids,
            "cmg_violation_codes": [v.code for v in observed_violations],
            "cmg_parse_warnings": list(judge_result.parse_warnings),
        }


async def run_eval(cases: list[EvalCase], judge_fn: JudgeFn) -> list[dict[str, object]]:
    return [
        await evaluate_case(case, judge_fn, Path("cmg-eval-run"))
        for case in cases
    ]
```

The returned dict can be merged into the same JSONL/CSV row where your eval
already stores `score`, `gold_label`, latency, token counts, and model metadata.
Treat these fields as judge diagnostics, not as ground-truth correctness labels.
Typical CMG columns are:

| Column | Use |
|---|---|
| `cmg_graph_path` | Link to the per-item audit log. |
| `cmg_commitment_ids` | Claims the verdict depended on. |
| `cmg_violation_codes` | Consistency signals to aggregate or filter on. |
| `cmg_parse_warnings` | Annotation problems from the judge output. |
| `judge_visible_text` | Judge explanation without hidden `cmg` annotation blocks. |

For most eval harnesses, app-owned decisions are the simplest path: parse the
verdict from your normal judge output, then call `graph.add_decision(...)` with
the commitment IDs that were just applied. If your judge emits `decision` ops
itself, do not add a duplicate decision from the app.

To test whether a judge is stable under reviewer pushback, run a second judge
turn against the same graph after the initial `Decision`. Leave
`inject_state=True` so the judge sees active commitments and the last verdict. If
it flips from `pass` to `fail` without emitting an accepted `invalidation`, the
eval row will include `verdict_flip_without_invalidation`.

## DeepEval adapter

In DeepEval, put `cmg` inside a custom metric. DeepEval still owns test cases,
test runs, and pass/fail reporting; `cmg` records what the judge cited and adds
diagnostic metadata through the metric reason.

```python
import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from cmg import ClaimGraph, JsonlStorage, arun_turn, build_annotation_system_prompt

JudgeFn = Callable[[list[dict[str, str]]], Awaitable[str]]


def extract_verdict(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            return line.removeprefix("VERDICT:").strip().casefold()
    return "unknown"


class CmgJudgeMetric(BaseMetric):
    def __init__(
        self,
        judge_fn: JudgeFn,
        *,
        output_dir: Path = Path("cmg-deepeval"),
        threshold: float = 1.0,
    ) -> None:
        self.judge_fn = judge_fn
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph_path = self.output_dir / f"{item_id}.cmg.jsonl"

        async with ClaimGraph(JsonlStorage(graph_path)) as graph:
            input_node = (await graph.add_support(f"Input:\n{test_case.input}")).node
            actual_node = (await graph.add_support(
                f"Actual output:\n{test_case.actual_output or ''}"
            )).node
            expected_node = (await graph.add_support(
                f"Expected output / rubric:\n{test_case.expected_output or ''}"
            )).node

            result = await arun_turn(
                graph,
                self.judge_fn,
                [
                    {"role": "system", "content": build_annotation_system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Judge the actual output against the input and rubric.\n"
                            "Return `VERDICT: pass` or `VERDICT: fail`, then explain.\n"
                            "Emit cmg commitment ops for the claims behind the verdict.\n\n"
                            f"{input_node.node_id}: {input_node.content}\n"
                            f"{actual_node.node_id}: {actual_node.content}\n"
                            f"{expected_node.node_id}: {expected_node.content}"
                        ),
                    },
                ],
            )

            commitment_ids = [
                applied.node.node_id
                for applied in result.applied
                if applied.node.kind == "commitment"
            ]
            verdict = extract_verdict(result.visible_text)
            decision = await graph.add_decision(verdict, refs=commitment_ids)
            violation_codes = [
                *(v.code for v in result.violations()),
                *(v.code for v in decision.violations),
            ]

        self.score = 1.0 if verdict == "pass" else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"verdict={verdict}; cmg_graph_path={graph_path}; "
            f"cmg_violation_codes={violation_codes}"
        )
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "CMG Judge"


async def judge_fn(messages: list[dict[str, str]]) -> str:
    # Replace this with your actual judge model call.
    raise NotImplementedError("connect your judge model here")


test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="The answer should identify Paris.",
        name="capital-france",
    )
]

evaluate(test_cases=test_cases, metrics=[CmgJudgeMetric(judge_fn)])
```

The DeepEval score stays simple: `1.0` for `pass`, `0.0` for `fail`. The CMG
value is in the metric reason and the per-item graph path.

## Inspect AI scorer

In Inspect AI, put `cmg` in a custom scorer. The solver still evaluates the
candidate model. The scorer runs the judge, records CMG telemetry, and returns an
Inspect `Score` with CMG metadata stored in the eval log.

```python
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from cmg import ClaimGraph, JsonlStorage, arun_turn, build_annotation_system_prompt


def extract_verdict(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            return line.removeprefix("VERDICT:").strip().casefold()
    return "unknown"


async def judge_fn(messages: list[dict[str, str]]) -> str:
    # Replace this with your actual judge model call.
    raise NotImplementedError("connect your judge model here")


@scorer(metrics=[accuracy()])
def cmg_judge_scorer(output_dir: str = "cmg-inspect"):
    async def score(state: TaskState, target: Target) -> Score:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        graph_path = output_path / f"{state.sample_id}.cmg.jsonl"
        candidate_output = state.output.completion if state.output else ""

        async with ClaimGraph(JsonlStorage(graph_path)) as graph:
            prompt = (await graph.add_support(f"Prompt:\n{state.input_text}")).node
            candidate = (await graph.add_support(
                f"Candidate output:\n{candidate_output}"
            )).node
            rubric = (await graph.add_support(f"Target / rubric:\n{target.text}")).node

            result = await arun_turn(
                graph,
                judge_fn,
                [
                    {"role": "system", "content": build_annotation_system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Judge the candidate output.\n"
                            "Return `VERDICT: pass` or `VERDICT: fail`, then explain.\n"
                            "Emit cmg commitment ops for the claims behind the verdict.\n\n"
                            f"{prompt.node_id}: {prompt.content}\n"
                            f"{candidate.node_id}: {candidate.content}\n"
                            f"{rubric.node_id}: {rubric.content}"
                        ),
                    },
                ],
            )

            commitment_ids = [
                applied.node.node_id
                for applied in result.applied
                if applied.node.kind == "commitment"
            ]
            verdict = extract_verdict(result.visible_text)
            decision = await graph.add_decision(verdict, refs=commitment_ids)
            violation_codes = [
                *(v.code for v in result.violations()),
                *(v.code for v in decision.violations),
            ]

        return Score(
            value=CORRECT if verdict == "pass" else INCORRECT,
            answer=verdict,
            explanation=result.visible_text,
            metadata={
                "cmg_graph_path": str(graph_path),
                "cmg_commitment_ids": commitment_ids,
                "cmg_violation_codes": violation_codes,
                "cmg_parse_warnings": list(result.parse_warnings),
            },
        )

    return score


@task
def judged_capitals():
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
```

Run it with Inspect as usual:

```bash
inspect eval judged_capitals.py --model openai/gpt-4o-mini
```

Inspect keeps the normal score and log viewer workflow. CMG adds per-sample
metadata and JSONL graph files that explain how the judge selected the verdict.

## Configuration knobs

These are the main pieces an application can change without modifying the SDK:

| What you want to control | Use |
|---|---|
| Rubric, facts, diffs, test output, retrieved docs | Pre-seeded `Support` nodes |
| The actual model call | Any `async def llm(messages) -> str`, or provider helpers |
| Whether CMG state is injected into the prompt | `inject_state=True/False` |
| How verdicts are recorded | Model `decision` ops or app-owned `graph.add_decision(...)` |
| What happens when a check fires | `on_violation` or `result.violations()` |
| Persistence backend | `JsonlStorage` or a custom `Storage` implementation |
| Secret and PII handling | `redact_fn` |
| Streaming UI behavior | `astream_turn` |
| Notebook or script usage | `ClaimGraphSync` |

### `inject_state`

`arun_turn(..., inject_state=True)` and `astream_turn(..., inject_state=True)`
prepend a system message with active commitments and the last decision.

Use the default when you want the model to see its current claim state. Set it to
`False` when your application already builds the complete prompt or when you need
full control over system-message ordering.

### `on_violation`

Pass `on_violation` to handle violations as they are observed:

```python
def log_violation(v):
    logger.warning("cmg_violation", extra={"code": v.code, "detail": v.detail})

result = await arun_turn(graph, llm, messages, on_violation=log_violation)
```

The operation is still appended. Violations are observations, not exceptions.

### `redact_fn`

Use `redact_fn` to scrub stored graph text:

```python
graph = ClaimGraph(storage, redact_fn=lambda text: text.replace(api_key, "[REDACTED]"))
```

Redaction applies to support, commitment, decision, and invalidation text fields
before persistence.

### Storage

`JsonlStorage(path)` is the default backend. For production systems, implement
the `Storage` protocol:

```python
class MyStorage:
    async def append_node(self, node): ...
    async def append_violation(self, violation): ...
    def iter_records(self): ...
    async def aclose(self): ...
```

Use `ClaimGraph.aload(storage)` to replay an existing log into memory.

### Sync wrapper

For scripts and notebooks without a running event loop:

```python
from cmg.sync import ClaimGraphSync

with ClaimGraphSync(JsonlStorage(Path("conversation.jsonl"))) as graph:
    support = graph.add_support("evidence").node
```

Do not use `ClaimGraphSync` inside an already running event loop; use
`ClaimGraph` directly.

### Streaming

Use `astream_turn` when your provider streams text deltas. Visible prose streams
immediately, while `cmg` ops are applied after the full response is available.

```python
async for chunk in astream_turn(graph, stream_fn, messages):
    if chunk.is_final:
        result = chunk.result
    else:
        send_to_user(chunk.visible_text_delta)
```

### Provider helpers

Provider helpers are optional. You can always pass your own async function with
the shape `async def llm(messages: list[dict[str, str]]) -> str`.

OpenAI:

```python
from cmg.providers import make_openai_llm_fn

llm = make_openai_llm_fn("gpt-4o-mini", api_key="...")
```

Anthropic:

```python
from cmg.providers import make_anthropic_llm_fn

llm = make_anthropic_llm_fn("claude-sonnet-4-6", max_tokens=2048, api_key="...")
```

The provider adapters accept normal client keyword arguments and lazy-import
their SDKs, so the core package remains dependency-free.

## Recommended integration pattern

1. Create a graph per conversation, review, arbitration case, or evaluation item.
2. Pre-seed `Support` nodes for application evidence the model may cite.
3. Include `build_annotation_system_prompt()` and the relevant support IDs in
   your prompt.
4. Ask for commitments tied to support IDs before recording a verdict.
5. Record the verdict as a `Decision`, either from a model op or from your app.
6. Show `result.visible_text` to the user.
7. Send `result.violations()` or `on_violation` events to your observability
   pipeline.
8. Close storage with `async with`, `with`, `await graph.aclose()`, or `graph.close()`.

## Common violation signals

| Code | Meaning |
|---|---|
| `verdict_flip_without_invalidation` | The model changed verdict while prior commitments remained active. |
| `silent_commitment_drop` | The verdict stayed the same but stopped citing an active prior commitment. |
| `unknown_ref` | An op cited an ID that is not in the graph. |
| `wrong_ref_kind` | A commitment cited a non-support, or a decision cited a non-commitment. |
| `ref_not_active` | A decision cited an inactive commitment. |
| `empty_refs` | A commitment, decision, or invalidation omitted required refs. |

## Production checklist

- Persist logs somewhere durable, not a temporary directory.
- Use `redact_fn` if graph text may contain secrets or PII.
- Treat violations as telemetry and decide in your app whether to alert, retry,
  ask the model for an explicit invalidation, or simply record the event.
- Add provider timeouts, retries, and rate-limit handling around your LLM call.
- Keep `result.raw_text` only if your retention policy permits storing model
  annotations and hidden audit metadata.
