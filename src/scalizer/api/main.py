"""Run: python -m uvicorn scalizer.api.main:create_app --factory --host 127.0.0.1."""
from contextlib import asynccontextmanager
import json
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from scalizer.workload import DatasetReader
from scalizer.workload.contracts import Calibration, CpuBasis, SourceDescription

from .auth import AccessTokens, Settings, authenticated
from .jobs import BuildRequest, Identifier, Jobs, read_csv, write_json
from .schemas import DatasetPage, InputCreated, JobAccepted, JobReport, SeriesPage


def series_json(series, offset, limit):
    page = series.iloc[offset:offset + limit]
    return dict(total=len(series), offset=offset, limit=limit,
                rows=json.loads(page.to_json(orient="records", double_precision=15)))


def create_app(settings: Settings | None = None, *, alibaba_factory=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        kwargs = {"alibaba_factory": alibaba_factory} if alibaba_factory else {}
        app.state.jobs = Jobs(settings.root, **kwargs)
        try:
            yield
        finally:
            app.state.jobs.executor.shutdown(wait=True)

    app = FastAPI(title="Scalizer Workload API", version="0.1.0", lifespan=lifespan,
                  description="Prepare workloads and read per-minute series and calibrated models.")
    app.state.settings = settings
    app.state.tokens = AccessTokens(settings)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins),
                       allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type"])

    def reader():
        return DatasetReader(app.state.jobs.repository)

    def job(job_id):
        try:
            return app.state.jobs.get(job_id.hex)
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/workloads/inputs", status_code=201, response_model=InputCreated)
    def upload_csv(file: UploadFile = File(...), load_unit: str = Form(..., min_length=1),
                   cpu_basis: CpuBasis = Form(...), provider: str = Form("uploaded CSV"),
                   startup_minutes: int = Form(0, ge=0, le=18720),
                   user=Depends(authenticated)):
        """Upload an already adapted canonical CSV; raw Alibaba archives use jobs instead."""
        input_id = uuid4().hex
        folder = settings.root / "inputs" / input_id
        folder.mkdir()
        path = folder / "series.csv"
        size = 0
        try:
            with path.open("wb") as output:
                while block := file.file.read(1024 * 1024):
                    size += len(block)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, "CSV exceeds the 50 MiB upload limit")
                    output.write(block)
            try:
                raw = read_csv(path)
            except (ValueError, UnicodeError, OverflowError):
                raise HTTPException(422, "CSV must use the canonical service-minute format") from None
            description = SourceDescription(kind="canonical-csv", provider=provider,
                                            load_unit=load_unit, cpu_basis=cpu_basis,
                                            startup_minutes=startup_minutes)
            write_json(folder / "source.json", description.model_dump(mode="json"))
            return dict(input_id=input_id, rows=len(raw),
                        services=sorted(raw["service"].unique().tolist()))
        except BaseException:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            file.file.close()

    @app.post("/api/workloads/jobs", status_code=202, response_model=JobAccepted)
    def build(request: BuildRequest, user=Depends(authenticated)):
        """Fetch/ingest, clean and build in the worker; returns immediately with a job ID."""
        if request.input_id and not (settings.root / "inputs" / request.input_id / "source.json").is_file():
            raise HTTPException(404, "CSV input not found")
        try:
            record = app.state.jobs.submit(request, user["sub"])
        except OverflowError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "10"}) from None
        return dict(job_id=record["id"], status=record["status"],
                    status_url=f"/api/jobs/{record['id']}")

    @app.get("/api/jobs/{job_id}", response_model=JobReport)
    def get_job(job_id: UUID, user=Depends(authenticated)):
        return job(job_id)

    @app.get("/api/jobs/{job_id}/services/{service}/series", response_model=SeriesPage,
             response_model_exclude_unset=True)
    def get_job_series(job_id: UUID, service: Identifier,
                       offset: int = Query(0, ge=0), limit: int = Query(1440, ge=1, le=2000),
                       user=Depends(authenticated)):
        record = job(job_id)
        if record["status"] in {"queued", "running"}:
            raise HTTPException(409, "The job has not finished")
        path = settings.root / "jobs" / job_id.hex / "clean" / f"{service}.csv.gz"
        if not path.is_file():
            raise HTTPException(404, "Clean series not found")
        import pandas as pd
        return series_json(pd.read_csv(path), offset, limit)

    @app.get("/api/datasets", response_model=DatasetPage)
    def datasets(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
                 user=Depends(authenticated)):
        items = reader().list_datasets()
        return dict(total=len(items), offset=offset, limit=limit,
                    items=[dict(service=s, version=v) for s, v in items[offset:offset + limit]])

    @app.get("/api/datasets/{service}/{version}/series", response_model=SeriesPage,
             response_model_exclude_unset=True)
    def get_series(service: Identifier, version: Identifier,
                   start: int | None = Query(None, ge=0), end: int | None = Query(None, ge=0),
                   offset: int = Query(0, ge=0), limit: int = Query(1440, ge=1, le=2000),
                   user=Depends(authenticated)):
        if start is not None and end is not None and end <= start:
            raise HTTPException(422, "end must be after start; end is exclusive")
        try:
            return series_json(reader().get_series(service, version, start, end), offset, limit)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found") from None

    @app.get("/api/datasets/{service}/{version}/calibration", response_model=Calibration)
    def calibration(service: Identifier, version: Identifier, user=Depends(authenticated)):
        try:
            return reader().get_calibration(service, version).model_dump(mode="json")
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found") from None

    return app
