# Developer guide

## 1. Overview

cmg watches the claims an LLM makes during a multi-turn conversation. Each
claim, verdict and retraction becomes a typed node in an append-only graph.
Eleven deterministic checks run on every operation. When a check sees a
deviation (e.g. the model flipped its verdict without invalidating the
prior commitment), it writes a `Violation` record. The model's prose is
returned to the caller untouched. The graph is persisted as JSONL.

---

## 2. The big picture — three layers

```
┌──────────────────────────────────────────────────────────────┐
│  Glue layer        integration.py  sync.py  providers/*      │  ← talk to an LLM
├──────────────────────────────────────────────────────────────┤
│  Logic layer       graph.py  checks.py  parser.py  storage/  │  ← do the work
├──────────────────────────────────────────────────────────────┤
│  Data layer        nodes.py  errors.py  _schema.py           │  ← pure types
└──────────────────────────────────────────────────────────────┘
```

- The **data layer** has no I/O, no async, no state. Pure dataclasses and
  constants. Safe to import anywhere.
- The **logic layer** is where the work happens. It owns the asyncio lock,
  the in-memory indexes, the JSONL writer.
- The **glue layer** turns a string from an LLM into operations on the
  logic layer. You can replace it with your own code.

The data layer never depends on the logic layer. The logic layer never
depends on the glue layer. You can use cmg without integration.py if you
want to drive the graph yourself.

---

## 3. The data shapes

Four node kinds, each with a fixed ID prefix:

| Kind          | Prefix  | Purpose                                              |
|---------------|---------|------------------------------------------------------|
| Support       | `s-`    | A piece of evidence.                                 |
| Commitment    | `k-`    | A substantive claim the model makes.                 |
| Decision      | `d-`    | A verdict that closes a deliberation step.           |
| Invalidation  | `inv-`  | A structured retraction of a prior commitment.       |

Two more shapes you will see everywhere:

- **`Violation(node_id, code, detail, created_at)`** — an observation about
  the node that just got appended. Never raises; always returned.
- **`AppendResult(node, violations)`** — returned by every `add_*` method.

All shapes are `@dataclass(frozen=True, slots=True)`. Immutable.

---

## 4. The lifecycle of one turn (end-to-end)

This is what really happens when you do:

```python
result = await arun_turn(graph, llm_fn, [{"role": "user", "content": "review"}])
```

```
USER
  │
  ▼
arun_turn(graph, llm_fn, messages)
  │
  ├─[1]─ _with_state(graph, messages)
  │        └─ _state_message(graph)
  │             reads graph.active_commitments() and graph.last_decision()
  │             builds {"role": "system", "content": "Active commitments: ..."}
  │
  ├─[2]─ response = await llm_fn(payload)          ← your LLM is called here
  │
  ├─[3]─ _ingest_text(graph, response, on_violation)
  │        └─ parse_turn(response)
  │             ├─ _FENCE_RE.finditer  (find ```cmg ... ``` blocks)
  │             ├─ _TAG_RE.finditer    (find <cmg ops='...'/> tags)
  │             ├─ for each match: _extract_ops(body)
  │             └─ _strip_spans  → visible_text without the annotations
  │
  ├─[4]─ for op in parsed.ops:
  │        _apply_op(graph, op)
  │          └─ match op["op"]:
  │             ├─ "support"      → graph.add_support(...)
  │             ├─ "commitment"   → graph.add_commitment(...)
  │             ├─ "decision"     → graph.add_decision(...)
  │             └─ "invalidation" → graph.add_invalidation(...)
  │                                    │
  │                                    └─ async with graph._lock:
  │                                         node = X(node_id=mint_id(), ...)
  │                                         await graph._commit(node)
  │                                           ├─ raise if id duplicate
  │                                           ├─ violations = check_node(node, state)
  │                                           ├─ await storage.append_node(node)
  │                                           ├─ for v in violations:
  │                                           │     await storage.append_violation(v)
  │                                           └─ graph._apply(node, violations)
  │                                                ├─ update _nodes, _by_id
  │                                                ├─ update _active set
  │                                                └─ update _last_decision
  │
  └─[5]─ return TurnResult(visible_text, raw_text, applied, parse_warnings)

USER  ← shows result.visible_text to the human
       ← inspects result.violations() for logging / alerting
```

Streaming flow (`astream_turn`) is the same with two changes:

- It iterates `stream_fn(payload)` and pushes each delta through a
  `_StreamingAnnotationStripper`. The stripper holds back partial
  annotation blocks and emits "safe" text as soon as it knows the text is
  not part of an annotation.
- Ops are applied **only at the end**, on the full accumulated text. The
  final `StreamChunk` carries the `TurnResult`. We never apply half a
  JSON object.

---

## 5. Module by module

### `src/cmg/nodes.py` — the data shapes

**Purpose.** Define what a claim looks like and how to make a new id.

**Constants.**

- `PREFIX_SUPPORT = "s-"`, `PREFIX_COMMITMENT = "k-"`, `PREFIX_DECISION = "d-"`,
  `PREFIX_INVALIDATION = "inv-"`. These are the id prefixes used everywhere.
- `KIND_TO_PREFIX`: maps `"support" -> "s-"` and so on. Used by checks
  and by `mint_id`.
- `NODE_CLASSES`: maps `"support" -> Support`, etc. Used by `_replay` to
  rebuild typed nodes from a dict.
- `INVALIDATION_ACCEPTED = "invalidation_accepted"` and
  `INVALIDATION_REJECTED`. The only two allowed values for
  `Invalidation.result`.

**Functions.**

- `mint_id(kind: str) -> str` — make a new id like `s-abc123...`. The id is
  the prefix for the kind plus the full hex value from `uuid4().hex`. Called
  by every `add_*` method on the graph.
- `now_iso() -> str` — current UTC time as an ISO 8601 string. Called
  wherever we need a `created_at`.

**Classes.** All are `@dataclass(frozen=True, slots=True)`.

- `Support(node_id, content, created_at, refs=(), source_turn=0, kind="support")`
- `Commitment(node_id, content, created_at, refs=(), kind="commitment")`
- `Decision(node_id, content, created_at, refs=(), kind="decision")`
- `Invalidation(node_id, created_at, previous_commitment, previous_support,
   new_information, contrast_test, result, content="", evidence_source_turn=0,
   kind="invalidation")`
- `Node` — type alias `Support | Commitment | Decision | Invalidation`.
- `Violation(node_id, code, detail, created_at)` — an observation. Never
  raised; always returned in an `AppendResult`.
- `AppendResult(node, violations)` — what every `add_*` returns.

---

### `src/cmg/errors.py` — only system-integrity errors

**Purpose.** Define the two exceptions cmg ever raises.

**Classes.**

- `CmgError(Exception)` — base class. Catch this in user code if you want.
- `DuplicateNodeIdError(CmgError)` — the in-memory id index already has
  this id. Should never happen unless the JSONL log was edited by hand,
  or a bug in `mint_id` produces collisions.
- `MalformedLogLineError(CmgError)` — a JSONL line is not valid JSON, or
  is missing a required field, or has the wrong type for a field.

There is **no** exception class for "the model contradicted itself".
That is a `Violation`.

---

### `src/cmg/_schema.py` — on-disk format version

**Purpose.** Tag every persisted record with a version so we can change
the format later without breaking old logs.

**Constants.**

- `SCHEMA_VERSION = 1`.
- `RECORD_NODE = "node"`, `RECORD_VIOLATION = "violation"`. The
  discriminator that says which shape a JSONL record has.

**Functions.**

- `migrate_record(record, from_version) -> dict` — today it returns the
  record unchanged if the version matches, otherwise raises
  `SchemaMigrationError`. When you bump `SCHEMA_VERSION` you must edit
  this function to convert old records to the new shape.

---

### `src/cmg/checks.py` — the rules engine

**Purpose.** Pure functions that look at a node + graph state and return
the list of `Violation` records that apply. **They never raise.**

**Classes.**

- `GraphState(nodes_by_id, active_commitment_ids, last_decision)` —
  frozen dataclass. The snapshot the checks see. Built by
  `graph._snapshot_state()` just before each commit.

**Private helpers.**

- `_v(node_id, code, **detail) -> Violation` — short-hand to build a
  `Violation` with `now_iso()` as `created_at`.
- `_id_kind_check(node) -> tuple[Violation, ...]` — checks that the
  node's id prefix matches its kind. Only fires when replaying an
  externally-edited log; the SDK mints ids itself.
- `_check_refs(node_id, refs, *, expected_prefix, nodes_by_id,
   active=None) -> list[Violation]` — single point that runs three
  ref-level checks: `wrong_ref_kind` (prefix mismatch), `unknown_ref`
  (ref id not in the graph) and, when `active` is given, `ref_not_active`
  (ref is no longer in the active commitment set). Used by
  `check_commitment` (no `active`) and `check_decision` (with
  `active=state.active_commitment_ids`).

**Public checks.** All have the signature `(node, state) -> tuple[Violation, ...]`.

- `check_support(node, state)` — id, content not empty. Support refs are
  free-form so we do not validate them.
- `check_commitment(node, state)` — id, content not empty, refs not
  empty, then delegates to `_check_refs` with `PREFIX_SUPPORT`.
- `check_decision(node, state)` — id, content not empty, refs not empty,
  then delegates to `_check_refs` with `PREFIX_COMMITMENT` and the active
  set. Then it calls `_decision_transition_violations`.
- `_decision_transition_violations(node, state)` — the headline
  anti-capitulation check. Two cases:
  - Verdict text changed: if any prior-active commitment from the
    previous decision was not invalidated, emit
    `verdict_flip_without_invalidation`.
  - Verdict text did not change: if the new decision drops a commitment
    that the previous decision cited and that is still active, emit
    `silent_commitment_drop`.
- `check_invalidation(node, state)` — id, `new_information` and
  `contrast_test` not empty, `result` is one of the two allowed values,
  `previous_support` is non-empty, `previous_commitment` starts with
  `k-` and is currently active, and every `previous_support` id is one
  of the target commitment's own `refs` (you cannot challenge support
  the commitment never cited).
- `check_node(node, state)` — `match`/`case` dispatcher. Calls one of
  the four checks above.

---

### `src/cmg/storage/protocol.py` — pluggable interface

**Purpose.** Define what a storage backend must look like.

**Class.**

- `Storage(Protocol)` (runtime checkable) — has four methods:
  - `async append_node(node) -> None`
  - `async append_violation(violation) -> None`
  - `iter_records() -> Iterator[dict]` (sync; replay reads the whole
    file before any await)
  - `async aclose() -> None`

If you write your own class with those four methods, the graph will
accept it as `storage=...`. No changes to `graph.py` needed.

---

### `src/cmg/storage/jsonl.py` — the default backend

**Purpose.** Write nodes and violations as one JSON record per line.

**Class.**

`JsonlStorage(path: Path)` — single-process append-only writer.

**Internals.**

- `self.path: Path` — the JSONL file.
- `self._writer: IO[str] | None` — lazily opened on first write.

**Methods.**

- `_open_writer() -> IO[str]` — opens the file in `"a"` mode the first
  time it is needed. Idempotent.
- `async append_node(node)` — `asdict(node)`, add `"record": "node"` and
  `"cmg_schema_version": 1`, write a single JSON line, flush.
- `async append_violation(violation)` — same shape with
  `"record": "violation"`.
- `_write_line(payload)` — `json.dumps + write + flush`. Sync inside an
  async method on purpose: a few hundred bytes flush in microseconds; we
  don't pull in `aiofiles`.
- `iter_records() -> Iterator[dict]` — yields the records one by one.
  Raises `MalformedLogLineError` if a line is not valid JSON or is not
  an object. Used by replay.
- `async aclose()` — closes the file handle if it was opened.

**Note.** `JsonlStorage` is **single-process**. If two processes write to
the same file at the same time you can get interleaved lines. For
multi-process, write a different backend (sqlite, postgres) using the
Protocol.

---

### `src/cmg/graph.py` — the orchestrator

**Purpose.** The only place that owns mutable state and the lock.

**Class.**

`ClaimGraph(storage, *, redact_fn=None)`.

**Constructor.** Sets up empty `_nodes` list, `_by_id` dict, `_active`
set, `_violations` list, `_last_decision` reference, an `asyncio.Lock`
and the redact function (default is identity).

**Classmethod `aload(storage, *, redact_fn=None) -> ClaimGraph`** — builds
the graph then calls `_replay()`. Use this when you want to resume from
disk.

**Async context manager.** `async with ClaimGraph(...) as graph:` returns
the graph and calls `aclose()` on exit.

**Async mutators.** Each one:

1. Takes `self._lock`.
2. Mints a fresh id with `mint_id(kind)`.
3. Builds the node, applying `_redact` to free-text fields.
4. Calls `await self._commit(node)`.

There are four:

- `add_support(content, refs=(), *, source_turn=0)`
- `add_commitment(content, refs)`
- `add_decision(content, refs)`
- `add_invalidation(*, previous_commitment, previous_support,
   new_information, contrast_test, result, content="",
   evidence_source_turn=0)`

**`async _commit(node) -> AppendResult`** — the core path. In order:

1. If `node.node_id` is already in `_by_id`, raise `DuplicateNodeIdError`.
2. Build a `GraphState` snapshot with `_snapshot_state()`.
3. Get `violations = check_node(node, state)`.
4. `await self._storage.append_node(node)`.
5. For each `v` in violations: `await self._storage.append_violation(v)`.
6. Call `_apply(node, violations)` to update in-memory state.
7. Return `AppendResult(node, violations)`.

**`_snapshot_state() -> GraphState`** — packs the current in-memory state
into a frozen `GraphState`. Called by `_commit`.

**`_apply(node, violations)`** — updates the in-memory indexes:

- Append the node to `_nodes` and `_by_id`.
- Extend `_violations`.
- If `Commitment`: add its id to `_active`.
- If `Invalidation` with `result == "invalidation_accepted"`: remove
  `previous_commitment` from `_active`.
- If `Decision`: set as `_last_decision`.

**`async _replay()`** — used by `aload`. Iterates
`storage.iter_records()`. For each record:

- Reads `record["record"]` and `record["cmg_schema_version"]`.
- Calls `migrate_record` if the version is not current.
- If `"node"`: deserialize with `_node_from_record` and queue it.
- If `"violation"`: store the on-disk code under its `node_id`.
- Else: raise `MalformedLogLineError`.

Then it re-applies every node in order, recomputing checks. If the
recomputed violation codes do not match the on-disk codes, it appends a
`replay_mismatch` violation. **It does not raise on mismatch** — the
philosophy is observability.

**Read accessors (all sync).**

- `nodes() -> tuple[Node, ...]`
- `get(node_id) -> Node | None`
- `active_commitments() -> tuple[Commitment, ...]`
- `active_commitment_ids() -> frozenset[str]`
- `last_decision() -> Decision | None`
- `violations() -> tuple[Violation, ...]`
- `violations_for(node_id) -> tuple[Violation, ...]`

They are sync because they read from in-memory state that is maintained
incrementally; nothing to await.

**`async aclose()`** — delegates to `self._storage.aclose()`.

**Helpers.**

- `_node_from_record(record) -> Node` — used by `_replay`. Validates the
  `kind`, then builds the right typed dataclass with explicit field
  reads. `match`/`case` for type-safety; mypy `--strict` would not
  accept `NODE_CLASSES[kind](**record)`.
- `_required_str(record, key, *, default=None)` — read a string field or
  raise `MalformedLogLineError`. If `default` is given and the key is
  missing, returns the default.
- `_required_int(record, key, *, default)` — read an int field. Excludes
  `bool` (bool is an int subclass in Python and would slip through
  `isinstance`).
- `_str_tuple(record, key)` — read a list-of-strings field as a tuple.

---

### `src/cmg/parser.py` — text in, ops out

**Purpose.** Pull `cmg` op blocks out of a plain-text model response.
Never raises.

**Constants.**

- `_FENCE_RE` — matches ```` ```cmg ... ``` ```` fenced blocks (multiline).
- `_TAG_RE` — matches ` <cmg ops='...'/> ` self-closing tags.
- `_BLANKLINES_RE` — collapses 3+ blank lines down to 2.
- `MAX_PARSE_INPUT_BYTES = 1 MiB` — defensive cap on input size.

**Public class.**

- `ParsedTurn(visible_text, raw_text, ops, parse_warnings)` — frozen
  dataclass returned by `parse_turn`.

**Public function.**

- `parse_turn(text) -> ParsedTurn`:
  1. If `text` is bigger than `MAX_PARSE_INPUT_BYTES`, truncate it and
     add `"input truncated"` to `parse_warnings`. This guards against a
     model that floods the parser with megabytes of partial fences.
  2. Run `_FENCE_RE.finditer` and `_TAG_RE.finditer` and collect every
     match as `(start, end, source_kind, body_text)`.
  3. Sort the matches by `start` so ops come out in document order.
  4. For each match, call `_extract_ops`. If parsing fails, append a
     warning string and skip. Never raise.
  5. Build `visible_text` with `_strip_spans` so the user does not see
     the annotation blocks.
  6. Return `ParsedTurn`.

**Helpers.**

- `_extract_ops(body, source, offset) -> (ops_list, warning_or_none)` —
  `json.loads` the body, accept either a top-level list or a
  `{"ops": [...]}` object, keep only entries that are dicts.
- `_strip_spans(text, spans) -> str` — remove the spans from the text in
  order, then collapse leftover blank lines and strip outer whitespace.

---

### `src/cmg/integration.py` — talk to a real LLM

**Purpose.** Bridge between the graph and an LLM caller. BYOM: the
caller supplies an async function that takes messages and returns text.

**Type aliases.**

- `Message = dict[str, str]` — `{"role": "system"|"user"|"assistant", "content": ...}`.
- `AsyncLLMFn = Callable[[list[Message]], Awaitable[str]]`.
- `AsyncLLMStreamFn = Callable[[list[Message]], AsyncIterator[str]]`.

**Public dataclasses.**

- `TurnResult(visible_text, raw_text, applied, parse_warnings)`:
  - `applied` is a tuple of `AppendResult` — one per op the model
    emitted, in order.
  - `violations()` is a method that flattens every violation across
    every `AppendResult`.
- `StreamChunk(visible_text_delta, is_final, result)`:
  - `is_final` is `True` only for the last chunk.
  - `result` is the `TurnResult`, populated only on the final chunk.

**Public functions.**

- `async arun_turn(graph, llm_fn, messages, *, inject_state=True,
   on_violation=None) -> TurnResult`:
  1. If `inject_state`, prepend a `system` message built by
     `_state_message`. Otherwise use `messages` as-is.
  2. `response = await llm_fn(payload)`. Single call. No retries.
  3. `return await _ingest_text(graph, response, on_violation)`.

- `async astream_turn(graph, stream_fn, messages, *, inject_state=True,
   on_violation=None) -> AsyncIterator[StreamChunk]`:
  1. Same state injection.
  2. Iterate the stream. For each delta:
     - Append to `raw_parts`.
     - Push through `_StreamingAnnotationStripper.update`.
     - If the stripper has safe text, `yield StreamChunk(delta_text,
       is_final=False, result=None)`.
  3. After the stream ends, call `stripper.finalize()` for the tail.
  4. Parse and apply ops on the full `raw_text`. Yield one final chunk
     `StreamChunk("", is_final=True, result=TurnResult(...))`.

- `build_annotation_system_prompt() -> str` — returns a short
  instruction the caller can put in the system message to tell the
  model how to annotate its output. Optional. The model can also just
  emit plain prose with no annotations.

**Private helpers.**

- `_ingest_text(graph, text, on_violation) -> TurnResult` — runs
  `parse_turn`, then for each op calls `_apply_op` and collects the
  results. If a violation occurs and `on_violation` is given, calls it
  once per violation.
- `_apply_op(graph, op) -> (AppendResult | None, warning | None)` —
  dispatches on `op["op"]` to the right `graph.add_*` method.
  Validates fields with `_str / _seq / _int`. Catches `KeyError` and
  `TypeError` (raised by the field validators) and turns them into a
  warning string. Never raises.
- `_str(op, key, *, default=None) -> str` — read a string field. Raises
  if missing without a default, or if not a string.
- `_seq(op, key) -> tuple[str, ...]` — read a list field. Coerces every
  element to `str`.
- `_int(op, key, default) -> int` — read an int field. Excludes `bool`.
- `_with_state(graph, messages) -> list[Message]` — prepend the
  `_state_message` and return a new list.
- `_state_message(graph) -> Message` — build the system message that
  shows the model its active commitments and its last decision. If the
  graph is empty, content is `"CMG state: empty graph."`.

**Class `_StreamingAnnotationStripper`.**

Holds a private `_buf: str`. On each `update(delta)`:

1. Append the delta to the buffer.
2. If the buffer is over `MAX_STREAM_BUFFER_BYTES` (defensive cap, the
   model probably never closed an annotation), flush and reset.
3. Otherwise call `_drain(final=False)` and return whatever is safe to
   show.

On `finalize()`, call `_drain(final=True)` and return the rest.

`_drain(final)` loop:

- Look for the first start of an annotation in the buffer
  (`_earliest_start`).
- If no start: flush the buffer except a possible partial-prefix suffix
  (`_max_partial_suffix`). On `final`, flush everything.
- If a start: flush the prose before it, then try to match a complete
  annotation block at the start (`_consume_complete_block`). If matched,
  drop it and loop. If not matched yet, hold the buffer (model is still
  in the middle of an annotation). On `final`, flush whatever is left so
  the consumer is never stuck.

`_max_partial_suffix(buf) -> int` — how many trailing chars might be the
beginning of an annotation pattern. Held back so a later delta can
complete the pattern.

---

### `src/cmg/sync.py` — synchronous wrapper

**Purpose.** Make the async API usable from a script or a Jupyter cell
without `asyncio.run`.

**Class `ClaimGraphSync`.**

- `__init__(storage, *, redact_fn=None)` — creates a private
  `asyncio.new_event_loop()` and a `ClaimGraph`. Does no I/O.
- `load(storage, *, redact_fn=None)` classmethod — builds a graph and
  replays the storage on the private loop. Sync equivalent of `aload`.
- `_run(coro)` — runs a coroutine with `self._loop.run_until_complete`.
- `__enter__` / `__exit__` — context-manager support; `__exit__`
  delegates to `close()`.
- `add_support` / `add_commitment` / `add_decision` /
  `add_invalidation` — sync wrappers, each does
  `self._run(self._async.add_*(...))`.
- `run_turn(llm_fn, messages, ...)` — sync wrapper around `arun_turn`.
  The `llm_fn` is still an `AsyncLLMFn`; the wrapper drives it on the
  private loop.
- Read accessors (`nodes`, `get`, `active_commitments`,
  `active_commitment_ids`, `last_decision`, `violations`,
  `violations_for`) — delegate directly to `self._async`.
- `close()` — closes the storage and then the event loop. Idempotent.

**Caveat.** Do not use `ClaimGraphSync` from inside a running event
loop. `loop.run_until_complete` would raise.

---

### `src/cmg/providers/openai_provider.py` — optional helper

**Public functions.**

- `make_openai_llm_fn(model, **client_kwargs) -> AsyncLLMFn`:
  1. `from openai import AsyncOpenAI` (lazy, inside the body).
  2. If the import fails, raise `ImportError("install claim-memory-graph[openai]...")`.
  3. Build an `AsyncOpenAI` client.
  4. Return an async function that calls
     `client.chat.completions.create(model, messages)` and returns
     `resp.choices[0].message.content or ""`.

- `make_openai_astream_fn(model, **client_kwargs) -> AsyncLLMStreamFn`:
  Same shape with `stream=True`, yielding `chunk.choices[0].delta.content`.

### `src/cmg/providers/anthropic_provider.py` — optional helper

Same shape as openai_provider, with two extra details:

- `_split_system(messages)` separates `system` messages from the rest
  because Anthropic's API takes `system` as a top-level parameter.
- Both factories take `max_tokens` (default 1024).

### `src/cmg/providers/__init__.py` — provider convenience exports

Exports the optional provider factories from the provider package, so users can
write `from cmg.providers import make_openai_llm_fn`. The imported provider
modules must still keep third-party SDK imports inside factory functions, not at
module top level.

---

## 6. Where to change what

| You want to...                                 | Touch these files                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------------|
| Add a new check / Violation code               | `checks.py` (logic), `tests/test_checks.py` (legal + illegal + still-appended)    |
| Add a new node kind                            | `nodes.py`, `checks.py`, `graph.py` (`add_X` + `_node_from_record`), `integration.py` (`_apply_op`), bump `_schema.py` |
| Add a new storage backend                      | New file implementing `Storage` Protocol. Do NOT touch `graph.py`.                |
| Support a new LLM provider                     | New file under `providers/` with lazy import, add extras to `pyproject.toml`, export helper from `providers/__init__.py` |
| Change the annotation wire format              | `parser.py` (regex + `_extract_ops`), maybe `build_annotation_system_prompt`      |
| Forward violations to Datadog / Sentry         | Pass `on_violation` to `arun_turn` / `astream_turn`. No SDK change needed.        |
| Redact PII before persistence                  | Pass `redact_fn` to `ClaimGraph(storage, redact_fn=...)`. No SDK change needed.   |
| Bump the on-disk format                        | Edit `_schema.py`: bump `SCHEMA_VERSION`, implement `migrate_record`.             |

---

## 7. Reading order for new contributors

- **10 minutes:** `nodes.py`, `errors.py`, `checks.py`. You will understand
  the data model and the rules.
- **30 minutes:** add `graph.py`, `parser.py`, `storage/jsonl.py`. You will
  understand the full append-and-replay path.
- **1 hour:** add `integration.py` (especially
  `_StreamingAnnotationStripper`) and `sync.py`. You will know how the
  glue layer talks to the world.

Everything else (`providers/*`, docs, tests) is application code.

---

## 8. Things to keep in mind when changing code

- **Never raise for semantic deviations.** Use a `Violation`. Reserve
  exceptions for system-integrity bugs.
- **Read accessors are sync. Mutators are async.** Do not change this.
  Sync reads use frozen snapshots from indexes maintained incrementally.
- **The lock is held during validate-then-append.** Do not move I/O out
  of `_commit` without thinking carefully about the failure modes
  documented above.
- **Replay must re-run the checks.** Never trust on-disk violation codes
  blindly; recompute and compare. That is how we detect tampering.
- **Provider modules import lazily.** Never import `openai` or
  `anthropic` at module top level. Doing so breaks zero-deps install.
- **Schema versioning is required on every persisted record.** If you
  change the persisted shape, bump `SCHEMA_VERSION` and write the
  migration.
