from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

PREFIX_SUPPORT = "s-"
PREFIX_COMMITMENT = "k-"
PREFIX_DECISION = "d-"
PREFIX_INVALIDATION = "inv-"

KIND_TO_PREFIX: dict[str, str] = {
    "support": PREFIX_SUPPORT,
    "commitment": PREFIX_COMMITMENT,
    "decision": PREFIX_DECISION,
    "invalidation": PREFIX_INVALIDATION,
}

INVALIDATION_ACCEPTED = "invalidation_accepted"
INVALIDATION_REJECTED = "invalidation_rejected"
INVALIDATION_RESULTS = frozenset({INVALIDATION_ACCEPTED, INVALIDATION_REJECTED})


def mint_id(kind: str) -> str:
    return f"{KIND_TO_PREFIX[kind]}{uuid.uuid4().hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class Support:
    node_id: str
    content: str
    created_at: str
    refs: tuple[str, ...] = ()
    source_turn: int = 0
    kind: Literal["support"] = "support"


@dataclass(frozen=True, slots=True)
class Commitment:
    node_id: str
    content: str
    created_at: str
    refs: tuple[str, ...] = ()
    kind: Literal["commitment"] = "commitment"


@dataclass(frozen=True, slots=True)
class Decision:
    node_id: str
    content: str
    created_at: str
    refs: tuple[str, ...] = ()
    kind: Literal["decision"] = "decision"


@dataclass(frozen=True, slots=True)
class Invalidation:
    node_id: str
    created_at: str
    previous_commitment: str
    previous_support: tuple[str, ...]
    new_information: str
    contrast_test: str
    result: str
    content: str = ""
    evidence_source_turn: int = 0
    kind: Literal["invalidation"] = "invalidation"


Node = Support | Commitment | Decision | Invalidation


@dataclass(frozen=True, slots=True)
class Violation:
    node_id: str
    code: str
    detail: dict[str, object]
    created_at: str


@dataclass(frozen=True, slots=True)
class AppendResult:
    node: Node
    violations: tuple[Violation, ...]


NODE_CLASSES: dict[str, type[Node]] = {
    "support": Support,
    "commitment": Commitment,
    "decision": Decision,
    "invalidation": Invalidation,
}


__all__ = [
    "INVALIDATION_ACCEPTED",
    "INVALIDATION_REJECTED",
    "INVALIDATION_RESULTS",
    "KIND_TO_PREFIX",
    "NODE_CLASSES",
    "PREFIX_COMMITMENT",
    "PREFIX_SUPPORT",
    "AppendResult",
    "Commitment",
    "Decision",
    "Invalidation",
    "Node",
    "Support",
    "Violation",
    "mint_id",
    "now_iso",
]
