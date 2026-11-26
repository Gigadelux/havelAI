"""Hosted analytics store: send reports to an HTTP ingest endpoint (`httpx`, `hosted` extra).

Contract (JSON, `Authorization: Bearer <key>`):

    POST {url}/detections          body: DetectionReport.to_dict()   -> 2xx
    GET  {url}/detections?limit=N  -> {"detections": [report, ...]}   newest first
    GET  {url}/detections/count    -> {"count": N}

A server implementing it can store the reports with SCHEMA.sql (e.g. on PostgreSQL). URL and key
default to `HAVEL_ANALYTICS_URL` / `HAVEL_ANALYTICS_API_KEY` in the environment or `.env`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..services.common.env import get_env
from .base import RecordRepository, RepositoryError


class RemoteRecordRepository(RecordRepository):
    kind = "remote_http"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Any = None,
    ) -> None:
        self.url = (url or get_env("HAVEL_ANALYTICS_URL") or "").rstrip("/")
        if not self.url:
            raise RepositoryError("no analytics URL: pass url=... or set HAVEL_ANALYTICS_URL")
        self._api_key = api_key or get_env("HAVEL_ANALYTICS_API_KEY")
        self.timeout = timeout
        self._client = client

    def __repr__(self) -> str:
        return f"RemoteRecordRepository({self.url!r})"

    def connect(self) -> None:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise RepositoryError(
                    "remote analytics needs httpx: pip install havelai[hosted]"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        self.connect()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            resp = self._client.request(method, f"{self.url}{path}", headers=headers, **kw)
        except Exception as exc:
            raise RepositoryError(f"could not reach {self.url}: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise RepositoryError(f"{method} {self.url}{path} returned HTTP {resp.status_code}")
        return resp.json() if resp.content else None

    def count(self) -> int:
        return int(self._request("GET", "/detections/count")["count"])

    def write_report(self, report: Mapping[str, Any]) -> str:
        self._request("POST", "/detections", json=dict(report))
        return str(report["id"])

    def reports(self, limit: int | None = None) -> list[dict[str, Any]]:
        params = {"limit": int(limit)} if limit is not None else None
        return list(self._request("GET", "/detections", params=params)["detections"])
