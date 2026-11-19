"""The Alibaba cluster trace, microservices v2022: the only module that knows its format.

AlibabaTraceSource (interfaces.WorkloadSource) turns two tables of the trace
into the raw series, one row per service and minute:

    MSResource  CPU and memory of each container      one archive per 30 minutes
    MSRTMCR     calls received and their latency      one archive per 3 minutes

The trace is about 4 GB per hour, so each archive is streamed, filtered to the
selected services and cached (an interrupted run resumes):

    1. read_shard            keep the rows of the services and minutes asked for
    2. per_container_minute  drop duplicates, average repeated readings
    3. aggregate_resource,   one row per service and minute
       aggregate_traffic
    4. join_tables           join the two tables on service + minute
"""
from __future__ import annotations

import io
import json
import tarfile
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contracts import CleaningParams, CpuBasis, SourceDescription
from .interfaces import RAW_REQUIRED_COLUMNS, validate_raw_series

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"
MINUTE_MS = 60_000
TRACE_MINUTES = 13 * 24 * 60   # the trace covers 13 days


@dataclass(frozen=True)
class Table:
    name: str            # name used in Scalizer (and in the cache)
    remote: str          # path prefix on the Alibaba server
    local_prefix: str    # file prefix used by Alibaba's fetchData.sh
    shard_minutes: int   # minutes covered by one archive
    columns: tuple       # column order in the CSV


MSRESOURCE = Table("msresource", "MSMetricsUpdate/MSMetricsUpdate", "MSMetrics", 30, (
    "timestamp", "msname", "msinstanceid", "nodeid",
    "cpu_utilization", "memory_utilization"))

MSRTMCR = Table("msrtmcr", "MCRRTUpdate/MCRRTUpdate", "MSRTMCR", 3, (
    "timestamp", "msname", "msinstanceid", "nodeid",
    "providerrpc_rt", "providerrpc_mcr", "consumerrpc_rt", "consumerrpc_mcr",
    "writemc_rt", "writemc_mcr", "readmc_rt", "readmc_mcr",
    "writedb_rt", "writedb_mcr", "readdb_rt", "readdb_mcr",
    "consumermq_rt", "consumermq_mcr", "providermq_rt", "providermq_mcr",
    "http_mcr", "http_rt"))

# Calls the service RECEIVES: its demand. Outbound calls (consumerrpc, database,
# Memcached, consumermq) are not part of the load.
INBOUND = ("providerrpc", "http", "providermq")

ID_COLUMNS = ["timestamp", "msname", "msinstanceid"]
KEEP = {MSRESOURCE.name: ID_COLUMNS + ["cpu_utilization", "memory_utilization"],
        MSRTMCR.name: ID_COLUMNS + [c for t in INBOUND for c in (f"{t}_mcr", f"{t}_rt")]}


def parse_when(value: str) -> int:
    """'<day>d<hour>' (the format of fetchData.sh) to a trace minute: 1d6 -> 1800."""
    try:
        day, hour = value.lower().split("d")
        minute = int(day) * 1440 + int(hour) * 60
    except ValueError as exc:
        raise ValueError(f"Invalid time {value!r}: use <day>d<hour>, e.g. 0d10") from exc
    if not 0 <= minute <= TRACE_MINUTES:
        raise ValueError(f"Time {value!r} is outside the 13 days of the trace")
    return minute


def shard_indices(table: Table, start: int, end: int) -> list[int]:
    """Archives of `table` that cover the minutes [start, end)."""
    if end <= start:
        return []
    return list(range(start // table.shard_minutes, (end - 1) // table.shard_minutes + 1))


# ===================================================================== the source

class AlibabaTraceSource:
    def __init__(self, location: str = BASE_URL, log=print):
        self.location = location   # the Alibaba server, or a local folder of archives
        self.log = log

    def describe(self) -> SourceDescription:
        return SourceDescription(
            kind="alibaba-microservices-v2022",
            provider="Alibaba cluster trace, microservices v2022 (MSResource, MSRTMCR)",
            location=self.location,
            load_unit="sum of inbound call rates (" + " + ".join(INBOUND)
                      + "), min-max normalised by Alibaba",
            cpu_basis=CpuBasis.NORMALISED_UNKNOWN_SCALE,
            notes="No wall-clock time: minute 0 is the start of the trace. Mean response "
                  "time only (no percentiles) and no error data: p95 and errors are "
                  "assumptions.")

    def default_cleaning(self) -> CleaningParams:
        # Container counts are unstable in the first minutes of the trace
        # (MS_70785: 226 -> 170 in 12 minutes).
        return CleaningParams(startup_minutes=15)

    def fetch(self, services: list[str], start: int | None, end: int | None,
              workdir: Path) -> pd.DataFrame:
        if start is None or end is None:
            raise ValueError("The Alibaba trace is 2 TB: give an explicit period "
                             "(start and end minutes)")
        return validate_raw_series(ingest(services, start, end, workdir,
                                          source=self.location, log=self.log))


# =================================================================== reading archives

class _Readable(io.RawIOBase):
    """Expose a file inside a streamed tar archive to pandas."""

    def __init__(self, stream):
        self.stream = stream

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _open_url(url: str, retries: int = 5):
    """Open a URL, retrying with a growing pause on network errors."""
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(url, timeout=120)
        except OSError as exc:
            if attempt == retries:
                raise OSError(f"Cannot download {url}: {exc}") from exc
            time.sleep(5 * attempt)


def _find_local(folder: Path, table: Table, idx: int) -> Path:
    """An archive or extracted CSV, named as on the server (MCRRTUpdate_200) or
    as by fetchData.sh (MSRTMCR_200)."""
    stems = {f"{Path(table.remote).name}_{idx}", f"{table.local_prefix}_{idx}"}
    for path in folder.rglob("*"):
        for suffix in (".tar.gz", ".csv.gz", ".csv"):
            if (path.name.endswith(suffix) and path.name[: -len(suffix)] in stems
                    and path.is_file()):
                return path
    raise FileNotFoundError(f"No file for {table.name} archive {idx} under {folder}")


def _csv_streams(source: str, table: Table, idx: int):
    """The binary CSV streams of one archive, remote or local."""
    if source.startswith(("http://", "https://")):
        # "r|gz": decompressed while it downloads, never written to disk.
        archive = _open_url(f"{source.rstrip('/')}/{table.remote}_{idx}.tar.gz")
    else:
        path = _find_local(Path(source), table, idx)
        if not path.name.endswith(".tar.gz"):
            with open(path, "rb") as f:
                yield f
            return
        archive = open(path, "rb")
    with archive, tarfile.open(fileobj=archive, mode="r|gz") as tar:
        for member in tar:
            if member.isfile():
                yield io.BufferedReader(_Readable(tar.extractfile(member)), 1 << 20)


def read_shard(source: str, table: Table, idx: int, services: set[str],
               start: int, end: int, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Rows of one archive for `services` in minutes [start, end)."""
    keep = KEEP[table.name]
    parts = []
    for stream in _csv_streams(source, table, idx):
        # Read as text in blocks: a header row or a bad value does not break parsing.
        for chunk in pd.read_csv(stream, header=None, names=list(table.columns),
                                 usecols=keep, dtype=str, chunksize=chunksize,
                                 low_memory=False):
            ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
            valid = ts.notna() & chunk["msname"].notna() & chunk["msinstanceid"].notna()
            minute = ts // MINUTE_MS
            mask = valid & chunk["msname"].isin(services) & minute.between(start, end - 1)
            if mask.any():
                kept = chunk.loc[mask].copy()
                kept["timestamp"] = ts[mask].astype("int64")
                for col in keep[3:]:
                    kept[col] = pd.to_numeric(kept[col], errors="coerce")
                parts.append(kept)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)


# ======================================================================= aggregation

def per_container_minute(data: pd.DataFrame, table: Table) -> pd.DataFrame:
    """One row per container and minute: identical rows count once, several
    readings in the same minute are averaged."""
    metrics = KEEP[table.name][3:]
    data = data.drop_duplicates(keep="first")
    data = data.assign(minute=data["timestamp"] // MINUTE_MS)
    return data.groupby(["msname", "msinstanceid", "minute"], as_index=False)[metrics].mean()


def aggregate_resource(rows: pd.DataFrame) -> pd.DataFrame:
    """Containers counted, CPU and memory averaged."""
    g = rows.groupby(["msname", "minute"])
    return pd.DataFrame({"containers": g["msinstanceid"].nunique(),
                         "cpu": g["cpu_utilization"].mean(),
                         "mem": g["memory_utilization"].mean()}).reset_index()


def aggregate_traffic(rows: pd.DataFrame) -> pd.DataFrame:
    """Load = sum of the inbound call rates over the containers; latency weighted
    by calls (a container serving 100 calls weighs more than one serving 1)."""
    rows = rows.copy()
    rows["load"] = sum(rows[f"{t}_mcr"].fillna(0) for t in INBOUND)
    rows["rt_x_mcr"] = sum(rows[f"{t}_rt"].fillna(0) * rows[f"{t}_mcr"].fillna(0)
                           for t in INBOUND)
    g = rows.groupby(["msname", "minute"])
    out = pd.DataFrame({"load": g["load"].sum(), "rt_x_mcr": g["rt_x_mcr"].sum()}).reset_index()
    out["rt_ms"] = out["rt_x_mcr"] / out["load"].where(out["load"] > 0)
    return out.drop(columns="rt_x_mcr")


def join_tables(resource: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Outer join: a minute present in only one table is kept, so cleaning sees the gap."""
    joined = (resource.merge(traffic, on=["msname", "minute"], how="outer")
                      .sort_values(["msname", "minute"])
                      .rename(columns={"msname": "service"}))
    joined["minute"] = joined["minute"].astype("int64")
    return joined[list(RAW_REQUIRED_COLUMNS)].reset_index(drop=True)


def ingest(services: list[str], start: int, end: int, cache_dir: str | Path,
           source: str = BASE_URL, log=print) -> pd.DataFrame:
    """The raw series of `services` over the minutes [start, end).

    Archives already read are taken from cache_dir/shards, if they were read for
    the same services and minutes."""
    if not services:
        raise ValueError("Select at least one service, e.g. MS_70785")
    wanted = set(services)
    meta = {"services": sorted(wanted), "minutes": [start, end]}
    rows = {}
    for table in (MSRESOURCE, MSRTMCR):
        cache = Path(cache_dir) / "shards" / table.name
        cache.mkdir(parents=True, exist_ok=True)
        parts = []
        indices = shard_indices(table, start, end)
        for n, idx in enumerate(indices, 1):
            data_file, meta_file = cache / f"{idx:05d}.csv.gz", cache / f"{idx:05d}.json"
            if data_file.exists() and meta_file.exists() \
                    and json.loads(meta_file.read_text()) == meta:
                parts.append(pd.read_csv(data_file))
                log(f"  {table.name} [{n}/{len(indices)}] archive {idx}: cached")
                continue
            t0 = time.time()
            data = read_shard(source, table, idx, wanted, start, end)
            tmp = data_file.with_name(data_file.name + ".tmp")
            data.to_csv(tmp, index=False, compression="gzip")
            tmp.replace(data_file)    # atomic: a half-written file is never reused
            meta_file.write_text(json.dumps(meta))
            parts.append(data)
            log(f"  {table.name} [{n}/{len(indices)}] archive {idx}: "
                f"{len(data):,} rows kept ({time.time() - t0:.0f}s)")
        data = (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=KEEP[table.name]))
        rows[table.name] = per_container_minute(data, table)

    series = join_tables(aggregate_resource(rows[MSRESOURCE.name]),
                         aggregate_traffic(rows[MSRTMCR.name]))
    missing = wanted - set(series["service"])
    if missing:
        warnings.warn("Not found in the trace: " + ", ".join(sorted(missing)), stacklevel=2)
    return series
