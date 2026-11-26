"""Pick a repository connector by name, so engines never import a connector directly."""

from __future__ import annotations

from typing import Any

from .base import RecordRepository, RepositoryError, VectorRepository

VECTOR_KINDS = ("milvus_lite", "qdrant")
RECORD_KINDS = ("sqlite", "remote_http")


def get_vector_repository(kind: str, **cfg: Any) -> VectorRepository:
    """`milvus_lite` (path, dimension, collection) or `qdrant` (dimension, collection, url, ...)."""
    if kind == "milvus_lite":
        from .milvus_lite import MilvusLiteRepository

        return MilvusLiteRepository(**cfg)
    if kind == "qdrant":
        from .qdrant import QdrantRepository

        return QdrantRepository(**cfg)
    raise RepositoryError(f"unknown vector repository {kind!r}; use one of {VECTOR_KINDS}")


def get_record_repository(kind: str, **cfg: Any) -> RecordRepository:
    """`sqlite` (path) or `remote_http` (url, api_key)."""
    if kind == "sqlite":
        from .sqlite import SQLiteRecordRepository

        return SQLiteRecordRepository(**cfg)
    if kind == "remote_http":
        from .remote_http import RemoteRecordRepository

        return RemoteRecordRepository(**cfg)
    raise RepositoryError(f"unknown record repository {kind!r}; use one of {RECORD_KINDS}")
