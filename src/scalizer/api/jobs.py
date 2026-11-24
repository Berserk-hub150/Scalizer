"""A bounded, single-worker MVP queue with persisted job status and archive cache."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, RLock
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scalizer.workload import AlibabaTraceSource, FileRepository, WorkloadAnalyzer
from scalizer.workload.alibaba import parse_when
from scalizer.workload.calibrate import add_features
from scalizer.workload.clean import clean_series
from scalizer.workload.contracts import Calibration, CleaningParams, SourceDescription
from scalizer.workload.interfaces import RAW_REQUIRED_COLUMNS, validate_raw_series

log = logging.getLogger(__name__)
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]
TraceTime = Annotated[str, Field(pattern=r"^(?:[0-9]|1[0-3])d(?:[0-9]|1[0-9]|2[0-3])$")]


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    services: list[Identifier] = Field(min_length=1, max_length=32)
    start: TraceTime
    end: TraceTime
    version: Identifier = "v1"
    source: Literal["alibaba", "csv"] = "alibaba"
    input_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")] | None = None

    @model_validator(mode="after")
    def check_period(self):
        if parse_when(self.end) <= parse_when(self.start):
            raise ValueError("end must be after start; end is exclusive")
        if len(set(self.services)) != len(self.services):
            raise ValueError("services must not contain duplicates")
        if (self.source == "csv") != (self.input_id is not None):
            raise ValueError("CSV jobs need input_id; Alibaba jobs must omit it")
        return self


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def json_values(value):
    """Convert numpy scalars and non-finite values into strict JSON."""
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_csv(path):
    raw = pd.read_csv(path)
    if "service" not in raw and "msname" in raw:
        raw = raw.rename(columns={"msname": "service"})
    raw = raw.drop(columns="load_per_instance", errors="ignore")
    if any(column not in raw for column in RAW_REQUIRED_COLUMNS):
        raise ValueError("CSV is missing required canonical columns")
    if raw.empty:
        raise ValueError("CSV must contain at least one data row")
    if not raw["service"].map(lambda x: isinstance(x, str) and
                              bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", x))).all():
        raise ValueError("Invalid service identifier")
    for name in [*RAW_REQUIRED_COLUMNS[2:], "rt_p95_ms", "error_rate"]:
        if name in raw:
            raw[name] = pd.to_numeric(raw[name], errors="raise")
            if np.isinf(raw[name].to_numpy(dtype=float)).any():
                raise ValueError(f"{name} must not contain infinity")
    if not raw["minute"].between(0, 13 * 1440 - 1).all():
        raise ValueError("minute must be in the supported 13-day period")
    return validate_raw_series(raw)


@dataclass
class CsvSource:
    description: SourceDescription

    def describe(self):
        return self.description

    def default_cleaning(self):
        return CleaningParams(startup_minutes=self.description.model_extra.get("startup_minutes", 0))


class Jobs:
    def __init__(self, root: Path, alibaba_factory=AlibabaTraceSource):
        self.root, self.alibaba_factory = root, alibaba_factory
        self.lock = RLock()
        self.capacity = BoundedSemaphore(4)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scalizer")
        for folder in ("jobs", "inputs", "cache", "datasets"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        self.repository = FileRepository(root / "datasets")
        self.records = {}
        for path in (root / "jobs").glob("*/job.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] in {"queued", "running"}:
                record.update(status="failed", stage="interrupted",
                              error="Server restarted; submit the request again to reuse archive cache")
                write_json(path, record)
            self.records[record["id"]] = record

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.records[job_id])

    def update(self, job_id, **changes):
        with self.lock:
            self.records[job_id].update(changes)
            write_json(self.root / "jobs" / job_id / "job.json", self.records[job_id])

    def submit(self, request: BuildRequest, owner: str):
        if not self.capacity.acquire(blocking=False):
            raise OverflowError("The processing queue is full; try again later")
        job_id = uuid4().hex
        folder = self.root / "jobs" / job_id
        record = dict(id=job_id, status="queued", stage="queued", owner=owner,
                      created_at=datetime.now(timezone.utc).isoformat(),
                      request=request.model_dump(), results=[], error=None)
        try:
            folder.mkdir()
            with self.lock:
                self.records[job_id] = record
                write_json(folder / "job.json", record)
            self.executor.submit(self.run, job_id, request)
        except BaseException:
            self.capacity.release()
            raise
        return copy.deepcopy(record)

    def publish(self, dataset, staged):
        target = self.repository.root / f"{dataset.service}_{dataset.version}"
        with self.lock:
            if target.exists():
                existing = self.repository.load(dataset.service, dataset.version)
                if existing.calibration != dataset.calibration:
                    raise FileExistsError("Version already contains different data; choose a new version")
            else:
                staged.replace(target)  # Readers only see a complete dataset.

    def run(self, job_id, request):
        folder = self.root / "jobs" / job_id
        results = []
        try:
            self.update(job_id, status="running", stage="ingest")
            start, end = parse_when(request.start), parse_when(request.end)
            if request.source == "csv":
                input_folder = self.root / "inputs" / request.input_id
                raw = read_csv(input_folder / "series.csv")
                source = CsvSource(SourceDescription.model_validate_json(
                    (input_folder / "source.json").read_text(encoding="utf-8")))
            else:
                source = self.alibaba_factory()
                key = hashlib.sha256(json.dumps(
                    [sorted(request.services), start, end]).encode()).hexdigest()[:20]
                raw = source.fetch(request.services, start, end, self.root / "cache" / key)
                raw.to_csv(folder / "raw_series.csv.gz", index=False, compression="gzip")
            self.update(job_id, stage="build")
            (folder / "clean").mkdir()
            for service in request.services:
                data = raw[(raw["service"] == service) & raw["minute"].between(start, end - 1)]
                result = dict(service=service, version=request.version)
                try:
                    if data.empty:
                        raise ValueError("No data for this service and period")
                    # Preserve an inspectable clean series even when fitting fails.
                    cleaned = add_features(clean_series(data, start, end, source.default_cleaning()))
                    cleaned.to_csv(folder / "clean" / f"{service}.csv.gz", index=False,
                                   compression="gzip")
                    result["quality_counts"] = cleaned["quality"].value_counts().to_dict()
                    with TemporaryDirectory(dir=folder) as temporary:
                        repository = FileRepository(temporary)
                        analyzer = WorkloadAnalyzer(source, repository)
                        dataset = analyzer.build_from_raw(data, service, start, end, request.version)
                        Calibration.model_validate(dataset.calibration)
                        json.dumps(dataset.calibration, allow_nan=False)
                        self.publish(dataset, Path(dataset.location))
                    result.update(status="calibrated", data_version=dataset.data_version,
                                  fit=json_values(dataset.fit))
                except FileExistsError as error:
                    result.update(status="version_conflict", error=str(error))
                except (ValueError, np.linalg.LinAlgError) as error:
                    result.update(status="clean_only" if "quality_counts" in result else "failed",
                                  error=str(error))
                results.append(json_values(result))
                self.update(job_id, results=results)
            calibrated = sum(x["status"] == "calibrated" for x in results)
            self.update(job_id, status="completed" if calibrated == len(results)
                        else "completed_with_errors", stage="done")
        except Exception:
            log.exception("Workload job %s failed", job_id)
            self.update(job_id, status="failed", stage="failed",
                        error="Processing failed; inspect the server log for this job ID")
        finally:
            self.capacity.release()

