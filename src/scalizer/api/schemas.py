"""Response contracts exported in /openapi.json for the React frontend."""
from typing import Literal

from pydantic import BaseModel, Field

from .jobs import BuildRequest


class DatasetRef(BaseModel):
    service: str
    version: str


class InputCreated(BaseModel):
    input_id: str
    rows: int
    services: list[str]


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


class ServiceResult(DatasetRef):
    status: Literal["calibrated", "clean_only", "version_conflict", "failed"]
    quality_counts: dict[str, int] = Field(default_factory=dict)
    fit: dict[str, float | None] | None = None
    data_version: str | None = None
    error: str | None = None


class JobReport(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "completed_with_errors", "failed"]
    stage: str
    owner: str
    created_at: str
    request: BuildRequest
    results: list[ServiceResult]
    error: str | None = None


class SeriesPoint(BaseModel):
    minute: int
    day: int
    hour_of_day: int
    load: float | None
    containers: float | None
    load_per_instance: float | None
    cpu: float | None
    mem: float | None
    rt_ms: float | None
    quality: Literal["ok", "containers_fixed", "spike_fixed", "interpolated", "missing"]
    rt_p95_ms: float | None = None
    error_rate: float | None = None


class SeriesPage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[SeriesPoint]


class DatasetPage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[DatasetRef]

