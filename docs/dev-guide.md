# Developer guide

This guide explains the internal shape of CMG. Keep changes small: the package
is intentionally a thin audit layer around judge decisions.

## Architecture

```
Data        nodes.py  errors.py  _schema.py
Logic       checks.py  graph.py  parser.py  storage/
Glue        integration.py  judge.py  report.py  cli.py  sync.py  providers/
```

- The data layer defines immutable dataclasses and constants.
- The logic layer owns validation, append-only state, JSONL replay, and parsing.
- The glue layer turns LLM text into graph operations and report artifacts.

The core graph does not call an LLM. LLM calls live in `integration.py`,
`judge.py`, or user code.

## Data Model

| Kind | Prefix | Purpose |
|---|---|---|
| `Support` | `s-` | Evidence the judge may cite. |
| `Commitment` | `k-` | A concrete claim tied to support ids. |
| `Decision` | `d-` | A verdict tied to active commitments. |
| `Invalidation` | `inv-` | A structured retraction of a commitment. |

Every mutator returns `AppendResult(node, violations)`. A `Violation` is an
observation, not an exception. The graph raises only for system-integrity
errors such as duplicate ids or malformed logs.

## Append Path

All writes go through `ClaimGraph._commit`:

1. Mint a node id.
2. Build a `GraphState` snapshot.
3. Run `check_node`.
4. Append the node to storage.
5. Append any violations to storage.
6. Update in-memory indexes.

Replay uses the same checks and compares recomputed violation codes with the
stored codes. A mismatch becomes `replay_mismatch`.

## Checks

`checks.py` validates relationships that should hold in a claim graph:

- Commitments should cite existing support ids.
- Decisions should cite active commitment ids.
- Invalidations should target active commitments.
- Verdict flips should invalidate prior active commitments first.
- Same-verdict decisions should not silently drop prior active commitments.

Checks must stay pure and deterministic. Add new checks in `checks.py` and test
both the clean and violating path.

## LLM Integration

`integration.py` provides:

- `arun_turn(graph, llm_fn, messages, inject_state=True)`
- `astream_turn(graph, stream_fn, messages, inject_state=True)`
- `build_annotation_system_prompt()`

The model can emit hidden `cmg` annotation blocks. CMG strips those from
visible text, applies the ops, and returns parse warnings for malformed blocks.

`judge.py` builds on this with `arun_judge`, a generic LLM-as-a-judge helper
that:

1. Pre-seeds support nodes for prompt, candidate output, reference answer,
   rubric, criteria, and extra evidence.
2. Calls the judge model once and ignores model-emitted `decision` ops.
3. Extracts `VERDICT: ...` from the visible text.
4. Validates the verdict against the allowed labels.
5. Records an app-owned decision over the active commitments emitted in that
   turn, but only when the verdict is valid.

When the verdict is missing or invalid, `JudgeResult.decision` is `None`.
`report.py` then marks the case with `missing_verdict` or `invalid_verdict`.

Do not put domain-specific prompts in `judge.py`. Domain-specific prompts belong
in user code or examples.

## Reports

`report.py` is pure read-side code. `judge_report(graph)` and
`to_markdown(graph)` must derive their output only from graph state. They should
not call a model, mutate the graph, or read external files.

Human review flags are intentionally simple. Prefer deterministic signals such
as missing cited claims, uncovered criteria, ignored references, or graph
violations.

`cli.py` builds the `cmg-view` terminal experience from one or more JSONL logs.
Keep it dependency-free, deterministic, and useful in plain terminals. It may
replay JSONL logs, but it should not call a model.

## Storage

`JsonlStorage` writes one schema-versioned JSON object per line. It is suitable
for local eval runs and debugging. Production users can implement the `Storage`
protocol without changing `ClaimGraph`.

If the persisted node shape changes, bump `SCHEMA_VERSION` and implement a
migration in `_schema.py`.

## Provider Helpers

Provider helpers must lazy-import third-party SDKs inside factory functions.
The core package must remain dependency-free unless users install extras.

## Change Map

| Change | Files |
|---|---|
| New graph consistency check | `checks.py`, `tests/test_checks.py` |
| New node kind | `nodes.py`, `checks.py`, `graph.py`, `integration.py`, `_schema.py` |
| Judge prompt or output contract | `judge.py`, `tests/test_judge.py`, docs |
| Report shape | `report.py`, `tests/test_judge.py`, docs |
| CLI output shape | `cli.py`, `tests/test_cli.py`, docs |
| Provider helper | `providers/`, `pyproject.toml`, provider tests |
| Parser format | `parser.py`, `integration.py`, parser/integration tests |

## Local Verification

Before handing off a change, run:

```bash
uv run python -m pytest
uv run ruff check .
uv run python -m mypy
```
