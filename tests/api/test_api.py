"""HTTP integration tests against the real Workload Analyzer, without Alibaba downloads."""
from datetime import datetime, timedelta, timezone
import json
import time
from types import SimpleNamespace

from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import jwt
import numpy as np
import pandas as pd
import pytest

from scalizer.api.auth import AccessTokens, Settings
from scalizer.api.main import create_app
from scalizer.workload.contracts import CleaningParams, SourceDescription


def measurements(service="MS_TEST", n=300):
    minute = np.arange(n)
    load = 100 + minute * 0.5
    cpu = 0.015 * (load / 10) + 0.05
    return pd.DataFrame(dict(service=service, minute=minute, load=load,
                             containers=10, cpu=cpu, mem=0.2,
                             rt_ms=20 + 3 * cpu / (1 - cpu)))


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(tmp_path, local_demo=True))
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        yield http


def upload(client, raw):
    response = client.post("/api/workloads/inputs", files={
        "file": ("../../ignored.csv", raw.to_csv(index=False), "text/csv")}, data={
        "load_unit": "requests per second", "cpu_basis": "fraction_of_request"})
    assert response.status_code == 201, response.text
    return response.json()["input_id"]


def build(client, input_id, services=None, version="v1"):
    response = client.post("/api/workloads/jobs", json=dict(
        services=services or ["MS_TEST"], start="0d0", end="0d6", version=version,
        source="csv", input_id=input_id))
    assert response.status_code == 202, response.text
    return response.json()


def finished(client, accepted):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(accepted["status_url"])
        assert response.status_code == 200
        record = response.json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(0.01)
    pytest.fail("Job did not complete")


def test_csv_to_real_calibration_and_chart_json(client):
    raw = measurements().drop(index=[100, 101, 102])
    record = finished(client, build(client, upload(client, raw)))
    assert record["status"] == "completed", record
    result = record["results"][0]
    assert result["status"] == "calibrated"
    assert result["fit"]["cpu_r2"] > 0.999
    assert result["quality_counts"]["interpolated"] == 3
    assert result["quality_counts"]["missing"] == 60

    listing = client.get("/api/datasets").json()
    assert listing["items"] == [{"service": "MS_TEST", "version": "v1"}]
    calibration = client.get("/api/datasets/MS_TEST/v1/calibration").json()
    assert calibration["load_unit"] == "requests per second"
    assert calibration["cpu_basis"] == "fraction_of_request"
    assert calibration["cpu_model"]["a"] == pytest.approx(0.015)
    assert "a_b_correlation" in calibration["cpu_model"]

    page = client.get("/api/datasets/MS_TEST/v1/series?start=298&end=303&limit=4").json()
    assert page["total"] == 5
    assert [p["minute"] for p in page["rows"]] == [298, 299, 300, 301]
    assert page["rows"][-1]["load"] is None
    assert page["rows"][-1]["quality"] == "missing"
    preview = client.get(f"/api/jobs/{record['id']}/services/MS_TEST/series?limit=2").json()
    assert len(preview["rows"]) == 2
    assert preview["rows"][0]["quality"] == "ok"  # company data has no Alibaba startup gap


def test_uncalibratable_service_preserves_clean_series(client):
    record = finished(client, build(client, upload(client, measurements(n=20))))
    assert record["status"] == "completed_with_errors"
    assert record["results"][0]["status"] == "clean_only"
    assert "at least 60" in record["results"][0]["error"]
    assert client.get("/api/datasets").json()["total"] == 0
    preview = client.get(f"/api/jobs/{record['id']}/services/MS_TEST/series").json()
    assert preview["total"] == 360


def test_version_is_immutable(client):
    first = upload(client, measurements())
    original = finished(client, build(client, first))
    before = client.get("/api/datasets/MS_TEST/v1/calibration").json()
    changed = measurements()
    changed["load"] *= 2
    second = upload(client, changed)
    conflict = finished(client, build(client, second))
    assert conflict["results"][0]["status"] == "version_conflict"
    assert client.get("/api/datasets/MS_TEST/v1/calibration").json() == before
    assert finished(client, build(client, second, version="v2"))["status"] == "completed"
    assert client.get(f"/api/jobs/{original['id']}").status_code == 200


@pytest.mark.parametrize("change", [
    {"end": "0d0"}, {"end": "0d24"}, {"start": "13d1", "end": "13d2"},
    {"services": ["../escape"]}, {"version": "../../escape"},
    {"services": ["MS_TEST", "MS_TEST"]}, {"source": "csv"},
    {"source": "alibaba", "input_id": "a" * 32}, {"unknown": "typo"},
])
def test_invalid_build_inputs(client, change):
    request = dict(services=["MS_TEST"], start="0d0", end="1d0", version="v1")
    request.update(change)
    assert client.post("/api/workloads/jobs", json=request).status_code == 422


def test_unknown_input_dataset_and_limits(client):
    request = dict(services=["MS_TEST"], start="0d0", end="1d0", version="v1",
                   source="csv", input_id="a" * 32)
    assert client.post("/api/workloads/jobs", json=request).status_code == 404
    assert client.get("/api/datasets/MS_UNKNOWN/v1/series").status_code == 404
    assert client.get("/api/datasets/MS_UNKNOWN/v1/calibration").status_code == 404
    assert client.get("/api/datasets?limit=501").status_code == 422
    assert client.get("/api/datasets/MS_TEST/v1/series?limit=99999").status_code == 422
    assert client.get("/api/jobs/" + "a" * 32).status_code == 404


@pytest.mark.parametrize("variant", ["missing_column", "duplicates", "infinite", "bad_minute", "bad_error_rate"])
def test_csv_validation(client, variant):
    raw = measurements()
    if variant == "missing_column":
        raw = raw.drop(columns="cpu")
    elif variant == "duplicates":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif variant == "infinite":
        raw.loc[0, "cpu"] = np.inf
    elif variant == "bad_minute":
        raw.loc[0, "minute"] = -1
    else:
        raw["error_rate"] = "not a number"
    response = client.post("/api/workloads/inputs", files={
        "file": ("data.csv", raw.to_csv(index=False), "text/csv")}, data={
        "load_unit": "requests per second", "cpu_basis": "fraction_of_request"})
    assert response.status_code == 422
    assert not list((client.app.state.settings.root / "inputs").iterdir())


def test_authentication_required_and_openapi_documents_it(tmp_path):
    with TestClient(create_app(Settings(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/datasets").status_code == 401
        assert client.get("/api/datasets", headers={"Authorization": "Bearer invalid"}).status_code == 401
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/api/datasets"]["get"]["security"]
        assert "SeriesPoint" in schema["components"]["schemas"]


def test_local_demo_rejects_remote_clients(tmp_path):
    with TestClient(create_app(Settings(tmp_path, local_demo=True)),
                    client=("203.0.113.4", 50000)) as client:
        assert client.get("/api/datasets").status_code == 403


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.mark.parametrize("change", [
    {"aud": "another-app"}, {"iss": "https://wrong.example/realms/scalizer"},
    {"exp": 1}, {"typ": "ID"}, {"sub": None},
])
def test_keycloak_claim_validation(tmp_path, rsa_key, change):
    settings = Settings(tmp_path)
    tokens = AccessTokens(settings)
    tokens.jwks.get_signing_key_from_jwt = lambda token: SimpleNamespace(key=rsa_key.public_key())
    now = datetime.now(timezone.utc)
    claims = dict(sub="student", iss=settings.issuer, aud="scalizer", typ="Bearer",
                  iat=now, exp=now + timedelta(minutes=5))
    token = jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": "test"})
    assert tokens.verify(token)["sub"] == "student"
    claims.update(change)
    with pytest.raises(jwt.PyJWTError):
        tokens.verify(jwt.encode(claims, rsa_key, algorithm="RS256"))


def test_completed_jobs_are_persisted_and_interrupted_jobs_reported(tmp_path):
    app = create_app(Settings(tmp_path, local_demo=True))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        record = finished(client, build(client, upload(client, measurements())))
    path = tmp_path / "jobs" / record["id"] / "job.json"
    with TestClient(create_app(Settings(tmp_path, local_demo=True)),
                    client=("127.0.0.1", 50000)) as client:
        assert client.get(f"/api/jobs/{record['id']}").json()["status"] == "completed"
        assert client.get("/api/datasets").json()["total"] == 1
    stale = json.loads(path.read_text())
    stale["status"] = "running"
    path.write_text(json.dumps(stale))
    with TestClient(create_app(Settings(tmp_path, local_demo=True)),
                    client=("127.0.0.1", 50000)) as client:
        interrupted = client.get(f"/api/jobs/{record['id']}").json()
        assert interrupted["status"] == "failed"
        assert interrupted["stage"] == "interrupted"


def test_alibaba_ingest_uses_worker_without_network(tmp_path):
    class Source:
        def describe(self):
            return SourceDescription(kind="test", provider="test", load_unit="requests/s",
                                     cpu_basis="fraction_of_request")

        def default_cleaning(self):
            return CleaningParams()

        def fetch(self, services, start, end, workdir):
            return measurements()

    app = create_app(Settings(tmp_path, local_demo=True), alibaba_factory=Source)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/workloads/jobs", json=dict(
            services=["MS_TEST"], start="0d0", end="0d6", version="v1"))
        assert response.status_code == 202
        assert finished(client, response.json())["status"] == "completed"


def test_queue_backpressure_does_not_block_status_requests(tmp_path):
    from threading import Event

    entered, release = Event(), Event()

    class SlowSource:
        def fetch(self, *args):
            entered.set()
            release.wait(5)
            raise OSError("Test source deliberately unavailable")

    app = create_app(Settings(tmp_path, local_demo=True), alibaba_factory=SlowSource)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        body = dict(services=["MS_TEST"], start="0d0", end="1d0", version="v1")
        try:
            accepted = [client.post("/api/workloads/jobs", json=body) for _ in range(4)]
            assert all(r.status_code == 202 for r in accepted)
            assert entered.wait(1)
            assert client.post("/api/workloads/jobs", json=body).status_code == 429
            assert client.get(accepted[0].json()["status_url"]).status_code == 200
        finally:
            release.set()
        assert finished(client, accepted[0].json())["status"] == "failed"


def test_upload_limit_and_legacy_header(tmp_path):
    small = create_app(Settings(tmp_path / "small", local_demo=True, max_upload_bytes=10))
    with TestClient(small, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/workloads/inputs", files={
            "file": ("x.csv", "x" * 11, "text/csv")}, data={
            "load_unit": "requests/s", "cpu_basis": "fraction_of_request"})
        assert response.status_code == 413
    app = create_app(Settings(tmp_path / "legacy", local_demo=True))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        raw = measurements().rename(columns={"service": "msname"})
        assert finished(client, build(client, upload(client, raw)))["status"] == "completed"


def test_bad_signature_is_rejected(tmp_path, rsa_key):
    tokens = AccessTokens(Settings(tmp_path))
    tokens.jwks.get_signing_key_from_jwt = lambda token: SimpleNamespace(key=rsa_key.public_key())
    forged_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    claims = dict(sub="student", iss=tokens.settings.issuer, aud="scalizer", typ="Bearer",
                  iat=now, exp=now + timedelta(minutes=5))
    with pytest.raises(jwt.InvalidSignatureError):
        tokens.verify(jwt.encode(claims, forged_key, algorithm="RS256"))


def test_csv_startup_setting_and_unit_immutability(client):
    data = measurements().to_csv(index=False)
    form = dict(load_unit="requests/s", cpu_basis="fraction_of_request", startup_minutes="15")
    response = client.post("/api/workloads/inputs", files={
        "file": ("x.csv", data, "text/csv")}, data=form)
    assert response.status_code == 201
    record = finished(client, build(client, response.json()["input_id"]))
    assert record["results"][0]["quality_counts"]["missing"] == 75
    series = client.get(f"/api/jobs/{record['id']}/services/MS_TEST/series?limit=1").json()
    assert series["rows"][0]["quality"] == "missing"
    form["load_unit"] = "different units"
    response = client.post("/api/workloads/inputs", files={
        "file": ("x.csv", data, "text/csv")}, data=form)
    record = finished(client, build(client, response.json()["input_id"]))
    assert record["results"][0]["status"] == "version_conflict"
