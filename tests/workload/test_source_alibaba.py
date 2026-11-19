import tarfile

import pandas as pd
import pytest

from scalizer.workload import WorkloadSource
from scalizer.workload import AlibabaTraceSource
from scalizer.workload.alibaba import MSRESOURCE, MSRTMCR, ingest, parse_when, shard_indices


def test_parse_when():
    assert parse_when("0d0") == 0
    assert parse_when("0d10") == 600
    assert parse_when("1d0") == 1440
    with pytest.raises(ValueError):
        parse_when("10")


def test_shard_indices_match_fetchdata():
    # fetchData.sh: one hour from 0d10 needs MSRTMCR 200..219 and MSResource 20..21
    assert shard_indices(MSRTMCR, 600, 660) == list(range(200, 220))
    assert shard_indices(MSRESOURCE, 600, 660) == [20, 21]
    # a window that does not start on an archive boundary
    assert shard_indices(MSRTMCR, 601, 604) == [200, 201]


def _write_archive(folder, table, idx, rows, header):
    name = f"{table.remote.split('/')[-1]}_{idx}"
    csv = folder / f"{name}.csv"
    pd.DataFrame(rows, columns=list(table.columns)).to_csv(csv, index=False, header=header)
    with tarfile.open(folder / f"{name}.tar.gz", "w:gz") as tar:
        tar.add(csv, arcname=csv.name)
    csv.unlink()


def _rtmcr_row(ts, ms, pod, rpc_mcr, rpc_rt, http_mcr=0.0, http_rt=0.0):
    row = {c: 0.0 for c in MSRTMCR.columns}
    row.update(timestamp=ts, msname=ms, msinstanceid=pod, nodeid="N1",
               providerrpc_mcr=rpc_mcr, providerrpc_rt=rpc_rt,
               consumerrpc_mcr=99.0, consumerrpc_rt=99.0,   # outbound: must be ignored
               http_mcr=http_mcr, http_rt=http_rt)
    return [row[c] for c in MSRTMCR.columns]


@pytest.fixture
def trace(tmp_path):
    """Minutes 600-602 (one MSRTMCR archive) of two services, with traps."""
    src = tmp_path / "src"
    src.mkdir()
    res = []
    for minute in (600, 601, 602):
        ts = minute * 60_000
        for pod in ("A_0", "A_1"):
            res.append([ts, "MS_A", pod, "N1", 0.30, 0.50])
        res.append([ts, "MS_B", "B_0", "N2", 0.90, 0.90])          # not selected
    res.append([600 * 60_000, "MS_A", "A_0", "N1", 0.30, 0.50])    # exact duplicate
    res.append([590 * 60_000, "MS_A", "A_0", "N1", 0.99, 0.99])    # outside the window
    # MSResource archive 20 covers minutes 600-629: written WITH a header
    _write_archive(src, MSRESOURCE, 20, res, header=True)

    rt = []
    for minute in (600, 601, 602):
        ts = minute * 60_000
        rt.append(_rtmcr_row(ts, "MS_A", "A_0", 0.02, 10.0, http_mcr=0.01, http_rt=40.0))
        rt.append(_rtmcr_row(ts, "MS_A", "A_1", 0.02, 20.0))
    rt.append(_rtmcr_row(601 * 60_000, "MS_A", "A_1", 3.5, 20.0))  # above 1: kept
    # MSRTMCR archive 200 covers minutes 600-602: written WITHOUT a header
    _write_archive(src, MSRTMCR, 200, rt, header=False)
    return src


def test_ingest_local_archives(trace, tmp_path):
    out = tmp_path / "out"
    series = ingest(["MS_A"], 600, 603, out, source=str(trace), log=lambda *_: None)
    s = series.set_index("minute")

    assert list(series.columns) == ["service", "minute", "load", "containers", "cpu", "mem",
                                    "rt_ms"]
    assert list(s.index) == [600, 601, 602]                   # header row skipped
    assert set(series["service"]) == {"MS_A"}                 # MS_B filtered out
    assert (s["containers"] == 2).all()                        # duplicate not counted twice
    assert s.loc[600, "cpu"] == pytest.approx(0.30)            # minute 590 excluded

    # load = inbound only (RPC + HTTP), summed over containers; outbound ignored
    assert s.loc[600, "load"] == pytest.approx(0.02 + 0.01 + 0.02)
    # response time weighted by calls: (0.02*10 + 0.01*40 + 0.02*20) / 0.05
    assert s.loc[600, "rt_ms"] == pytest.approx((0.2 + 0.4 + 0.4) / 0.05)
    # two readings of A_1 at minute 601 (0.02 and 3.5) conflict and are averaged
    assert s.loc[601, "load"] == pytest.approx(0.02 + 0.01 + (0.02 + 3.5) / 2)
    # only the cache of the archives is written
    assert {p.name for p in out.iterdir()} == {"shards"}


def test_ingest_warns_about_services_not_found(trace, tmp_path):
    with pytest.warns(UserWarning, match="MS_Z"):
        ingest(["MS_A", "MS_Z"], 600, 603, tmp_path / "out", source=str(trace),
               log=lambda *_: None)


def test_ingest_resumes_from_cache(trace, tmp_path):
    out = tmp_path / "out"
    ingest(["MS_A"], 600, 603, out, source=str(trace), log=lambda *_: None)
    for f in trace.iterdir():
        f.unlink()  # sources gone: a second run must use the cache only
    messages = []
    again = ingest(["MS_A"], 600, 603, out, source=str(trace), log=messages.append)
    assert all("cached" in m for m in messages)
    assert len(again) == 3


def test_missing_archive_is_an_error(trace, tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(["MS_A"], 600, 606, tmp_path / "out", source=str(trace), log=lambda *_: None)


def test_alibaba_source_implements_the_interface(trace, tmp_path):
    source = AlibabaTraceSource(str(trace), log=lambda *_: None)
    assert isinstance(source, WorkloadSource)
    assert source.default_cleaning().startup_minutes == 15
    raw = source.fetch(["MS_A"], 600, 603, tmp_path / "out")
    assert list(raw.columns[:2]) == ["service", "minute"] and len(raw) == 3
    with pytest.raises(ValueError):
        source.fetch(["MS_A"], None, None, tmp_path / "out")   # 2 TB: period required
