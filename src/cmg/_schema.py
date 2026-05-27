from __future__ import annotations

SCHEMA_VERSION = 1

RECORD_NODE = "node"
RECORD_VIOLATION = "violation"


class SchemaMigrationError(Exception):
    pass


def migrate_record(record: dict[str, object], from_version: int) -> dict[str, object]:
    if from_version == SCHEMA_VERSION:
        return record
    raise SchemaMigrationError(
        f"no migration from schema {from_version} to {SCHEMA_VERSION}"
    )


__all__ = [
    "RECORD_NODE",
    "RECORD_VIOLATION",
    "SCHEMA_VERSION",
    "SchemaMigrationError",
    "migrate_record",
]
