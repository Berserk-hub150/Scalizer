# Workload API

This is the HTTP adapter around the existing Python Workload Analyzer. The API
returns JSON; React draws the charts. Cleaning, calibration and the model
formulas remain in `scalizer.workload`.

## Start on Windows

Open PowerShell in the repository root:

```powershell
py -m pip install -r requirements-api.txt
$env:PYTHONPATH = "$PWD\src"
py -m uvicorn scalizer.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for Swagger and
<http://127.0.0.1:8000/openapi.json> for the machine-readable contract. Run one
API process. This MVP has one worker and at most four queued/running jobs.

For a local demonstration before connecting Keycloak, set
`$env:SCALIZER_LOCAL_DEMO = "true"` **before** starting the server. This mode
accepts API calls from loopback clients only. Remove that environment variable
when connecting the authenticated React application.

To try the whole flow without downloading Alibaba data, run
`py examples/create_api_demo.py`, upload the resulting
`data/api-demo-synthetic.csv` in Swagger, and use its returned `input_id` to
submit a CSV job for `MS_EXAMPLE`, `start=0d0`, `end=0d5`, `version=v1`.
Declare `load_unit=requests per second` and `cpu_basis=fraction_of_request`.
This demonstration is explicitly synthetic and is not evidence from the Alibaba trace.

## Keycloak and React

Authentication is enabled by default. The API validates access-token signatures
using the realm's JWKS, plus issuer, audience, expiry, issued-at and subject. It
accepts RS256 access tokens with `typ=Bearer`; user passwords stay with Keycloak.

```powershell
$env:SCALIZER_KEYCLOAK_ISSUER = "http://localhost:8080/realms/scalizer"
$env:SCALIZER_KEYCLOAK_AUDIENCE = "scalizer"
$env:SCALIZER_CORS_ORIGINS = "http://localhost:5173"
Remove-Item Env:SCALIZER_LOCAL_DEMO -ErrorAction SilentlyContinue
```

The issuer must match the token's `iss` exactly, including HTTP/HTTPS and port.
Use a trusted certificate for HTTPS. Configure CORS with the actual React origin
when Aspire assigns a different port.

Keycloak does not necessarily include the frontend client ID in access-token
audiences. In the `scalizer` client's dedicated client scope, add an **Audience**
mapper with **Included Client Audience = scalizer** and **Add to access token = On**.
Obtain a new access token after the change. Alternatively configure a dedicated
API audience and set `SCALIZER_KEYCLOAK_AUDIENCE` to that audience.

React sends `Authorization: Bearer <keycloak.token>` on every `/api/` request,
refreshing the token through Keycloak when needed. The MVP uses a shared project
catalogue: authenticated members can inspect its jobs and datasets. Health,
Swagger and the OpenAPI schema are public.

## Start an Alibaba workload

`POST /api/workloads/jobs`:

```json
{
  "services": ["MS_70785"],
  "start": "0d0",
  "end": "1d0",
  "version": "v1"
}
```

The four agreed inputs are preserved. `services` is a JSON array. `start` and
`end` use the CLI's relative `<day>d<hour>` format; **end is exclusive**. Thus
`0d10` to `0d11` selects relative minutes 600–659. These are trace-relative
hours, not a confirmed local timezone. A request selects 1–32 distinct services
inside the supported 13-day period. `13d0` is a valid exclusive end.

Response: HTTP **202**, for example:

```json
{
  "job_id": "7aba17b3202d4879a14c7b3b609f2300",
  "status": "queued",
  "status_url": "/api/jobs/7aba17b3202d4879a14c7b3b609f2300"
}
```

The worker fetches/aggregates the trace, then calls
`WorkloadAnalyzer.build_from_raw`, which already cleans and calibrates. A separate
mandatory `clean` HTTP call is unnecessary. The server controls storage paths
and the Alibaba source URL. The browser never submits shell commands or paths.
Repeated Alibaba jobs reuse the archive cache for the same selection and period.
An initial ingestion can download several GB: the API does not replace the
underlying archive ingestion with a small remote query.

## Upload an already adapted CSV

`POST /api/workloads/inputs` takes multipart form fields:

| Field | Meaning |
| --- | --- |
| `file` | Plain UTF-8 CSV, up to 50 MiB; not a raw Alibaba archive or a gzip file |
| `load_unit` | Actual unit of `load`, e.g. `requests per second` |
| `cpu_basis` | `fraction_of_request`, `fraction_of_limit`, `fraction_of_core`, `fraction_of_vm` or `normalised_unknown_scale` |
| `provider` | Optional source name |
| `startup_minutes` | Optional absolute minutes excluded at the time origin; default 0; use 15 when reproducing the Alibaba CLI defaults |

Canonical columns: `service,minute,load,containers,cpu,mem,rt_ms`. `minute` is an
integer index; each service/minute pair must be unique. Missing `mem` and `rt_ms`
values are allowed, but their columns must exist. Optional `rt_p95_ms` and
`error_rate` follow the Workload contract. The legacy `msname` header is accepted.
Company exports must be adapted to this schema and their units declared correctly;
the API does not infer arbitrary formats or convert unknown units. Timestamps
must already be converted into trace-relative minutes.

Example file:

```csv
service,minute,load,containers,cpu,mem,rt_ms
MS_EXAMPLE,0,100,10,0.20,0.25,20.75
MS_EXAMPLE,1,110,10,0.215,0.25,20.82
```

This two-row example illustrates the format; it is too short to calibrate a model.
Upload returns `input_id`, `rows` and `services`. Then submit a job with the same
four parameters and two additional fields:

```json
{
  "services": ["MS_EXAMPLE"],
  "start": "0d0",
  "end": "1d0",
  "version": "v1",
  "source": "csv",
  "input_id": "7aba17b3202d4879a14c7b3b609f2300"
}
```

## Poll and display results

Poll `GET /api/jobs/{job_id}` until the status is terminal:

| Status | Meaning |
| --- | --- |
| `queued`, `running` | Work in progress; `stage` is `ingest` or `build` |
| `completed` | Every requested service was calibrated |
| `completed_with_errors` | Inspect per-service results: some fits failed or versions conflicted |
| `failed` | The job failed; server logs contain technical details |

Each service result has `status`, `quality_counts` and, on success, `fit` and
`data_version`. A `clean_only` result retains its clean series and reports why
calibration failed. A `version_conflict` requires selecting another version.
Calibration success alone does not establish model accuracy; display the fit
errors alongside the result. `cpu_test_error` and `latency_test_error` are
fractions: **0.025 means 2.5%**.

| Endpoint | Result |
| --- | --- |
| `GET /api/jobs/{id}/services/{service}/series` | Clean data from a finished job, including services that could not be calibrated |
| `GET /api/datasets?offset=0&limit=100` | Catalogue of calibrated service/version pairs |
| `GET /api/datasets/{service}/{version}/series?start=0&end=1440&offset=0&limit=1440` | Stored minute-level series, suitable for charts and tables |
| `GET /api/datasets/{service}/{version}/calibration` | Existing typed `Calibration` JSON contract |

Series responses contain `{total, offset, limit, rows}`. Every row includes
`minute`, `load`, `containers`, `load_per_instance`, `cpu`, `mem`, `rt_ms`, time
features and `quality`. Unavailable/non-finite chart values become JSON `null`.
Preserve these gaps when plotting. Row pages are limited to 2,000 points and
catalogue pages to 500 items, so React never downloads the whole day/all services
in one response. `start` and `end` in GET queries are integer minutes and describe
a half-open interval. `total` counts the filtered rows before pagination.

Example React request after login:

```ts
await keycloak.updateToken(30);
const response = await fetch("http://127.0.0.1:8000/api/workloads/jobs", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${keycloak.token}`,
  },
  body: JSON.stringify({services: ["MS_70785"], start: "0d0", end: "1d0", version: "v1"}),
});
if (!response.ok) throw new Error(await response.text());
const {job_id, status_url} = await response.json();
// Poll status_url with the same Authorization header, then fetch a series page.
```

## Existing data and MVP operation

Storage defaults to `data/api/{inputs,jobs,cache,datasets}`. Set
`SCALIZER_API_DATA` to a directory containing an existing `datasets/` folder to
inspect those Workload outputs. The repository is filesystem-backed through the
existing `FileRepository`; PostgreSQL integration and simulation APIs are separate
increments. Versioned datasets are published atomically and conflicting data or
units cannot overwrite an existing version.

Job state survives restarts. Jobs that were queued/running are marked interrupted
on startup: resubmit them to reuse archive cache. This is an in-process MVP queue,
not a durable distributed execution service. Use one Uvicorn worker, avoid
`--reload` during processing, and allow jobs to finish during graceful shutdown.
For a shared/public deployment, also apply request-body limits at the reverse
proxy and replace the local queue with a durable worker as needed.

Tests: `py -m pip install -e ".[api,dev]"`, then `py -m pytest -q`.

Reference implementations used: [FastAPI](https://fastapi.tiangolo.com/),
[PyJWT](https://pyjwt.readthedocs.io/en/stable/usage.html), and
[Keycloak audience mappers](https://www.keycloak.org/admin-api/protocol-mappers).
