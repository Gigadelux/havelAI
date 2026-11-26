"""Data-write layer: one abstract `Repository`, one connector per backend. See `base.py`."""

from .base import (
    RecordRepository,
    Repository,
    RepositoryError,
    VectorHit,
    VectorRecord,
    VectorRepository,
)
from .factory import get_record_repository, get_vector_repository

__all__ = [
    "RecordRepository",
    "Repository",
    "RepositoryError",
    "VectorHit",
    "VectorRecord",
    "VectorRepository",
    "get_record_repository",
    "get_vector_repository",
]
