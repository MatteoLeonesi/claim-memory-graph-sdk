from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cmg._schema import SCHEMA_VERSION
from cmg.errors import CmgError, DuplicateNodeIdError, MalformedLogLineError
from cmg.graph import ClaimGraph
from cmg.integration import (
    AsyncLLMFn,
    AsyncLLMStreamFn,
    Message,
    StreamChunk,
    TurnResult,
    arun_turn,
    astream_turn,
    build_annotation_system_prompt,
)
from cmg.nodes import (
    INVALIDATION_ACCEPTED,
    INVALIDATION_REJECTED,
    AppendResult,
    Commitment,
    Decision,
    Invalidation,
    Node,
    Support,
    Violation,
)
from cmg.parser import ParsedTurn, parse_turn
from cmg.storage import JsonlStorage, Storage

try:
    __version__ = version("claim-memory-graph")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "INVALIDATION_ACCEPTED",
    "INVALIDATION_REJECTED",
    "SCHEMA_VERSION",
    "AppendResult",
    "AsyncLLMFn",
    "AsyncLLMStreamFn",
    "ClaimGraph",
    "CmgError",
    "Commitment",
    "Decision",
    "DuplicateNodeIdError",
    "Invalidation",
    "JsonlStorage",
    "MalformedLogLineError",
    "Message",
    "Node",
    "ParsedTurn",
    "Storage",
    "StreamChunk",
    "Support",
    "TurnResult",
    "Violation",
    "__version__",
    "arun_turn",
    "astream_turn",
    "build_annotation_system_prompt",
    "parse_turn",
]
