from __future__ import annotations


class CmgError(Exception):
    """Base class for cmg system-integrity errors. Semantic deviations use Violation, not exceptions."""


class DuplicateNodeIdError(CmgError):
    """A node id already exists in the in-memory graph or replayed log."""


class MalformedLogLineError(CmgError):
    """A persisted JSONL record is not valid cmg data."""


__all__ = ["CmgError", "DuplicateNodeIdError", "MalformedLogLineError"]
