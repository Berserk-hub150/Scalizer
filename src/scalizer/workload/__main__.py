"""Command line of the Workload Analyzer.

    python -m scalizer.workload ingest --services MS_70785 --start 0d0 --end 1d0 --out data/MS_70785
    python -m scalizer.workload clean  --out data/MS_70785      # optional: inspect
    python -m scalizer.workload build  --out data/MS_70785 --service MS_70785 --version v1

`ingest` downloads and aggregates the Alibaba trace once and writes
<out>/raw_series.csv.gz; `clean` and `build` read it (or a series given with
--input) without downloading again.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .alibaba import BASE_URL, AlibabaTraceSource, parse_when
from .analyzer import WorkloadAnalyzer
from .clean import QUALITY_FLAGS, clean_series
from .dataset import FileRepository
from .interfaces import validate_raw_series

RAW_SERIES_FILE = "raw_series.csv.gz"


def cmd_ingest(args):
    services = [s for s in args.services.split(",") if s]
    start, end = parse_when(args.start), parse_when(args.end)
    if end <= start:
        sys.exit("--end must be after --start")
    print(f"Ingest {', '.join(services)}, minutes {start}-{end - 1}, from {args.source}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = AlibabaTraceSource(args.source).fetch(services, start, end, out)
    if raw.empty:
        sys.exit("No data for the requested services and period")
    raw.to_csv(out / RAW_SERIES_FILE, index=False, compression="gzip")
    print(f"{len(raw):,} service-minutes, services: {', '.join(raw['service'].unique())}")
    print(f"Written {out / RAW_SERIES_FILE}")


def _raw_series(args) -> pd.DataFrame:
    """The series written by ingest, or the one given with --input."""
    path = Path(args.input or Path(args.out) / RAW_SERIES_FILE)
    if not path.exists():
        sys.exit(f"{path} not found: run ingest first, or give --input")
    raw = pd.read_csv(path)
    if "service" not in raw and "msname" in raw:      # column name of the Alibaba scripts
        raw = raw.rename(columns={"msname": "service"})
    raw = raw.drop(columns="load_per_instance", errors="ignore")
    return validate_raw_series(raw)


def _cleaning(args):
    """Cleaning defaults of the Alibaba trace, with the overrides of the command line."""
    params = AlibabaTraceSource().default_cleaning()
    if args.max_gap is not None:
        params = replace(params, max_interpolated_gap=args.max_gap)
    if args.startup_minutes is not None:
        params = replace(params, startup_minutes=args.startup_minutes)
    return params


def _period(args, raw):
    """--start/--end, otherwise from the first to the last minute of the series."""
    start = parse_when(args.start) if args.start else int(raw["minute"].min())
    end = parse_when(args.end) if args.end else int(raw["minute"].max()) + 1
    return start, end


def cmd_clean(args):
    """Show the result of cleaning, one file per service (build cleans by itself)."""
    raw = _raw_series(args)
    services = [s for s in (args.services or "").split(",") if s]
    if services:
        raw = raw[raw["service"].isin(services)]
    start, end = _period(args, raw)
    folder = Path(args.out) / "clean"
    folder.mkdir(parents=True, exist_ok=True)
    for service, data in raw.groupby("service"):
        clean = clean_series(data, start, end, _cleaning(args))
        clean.to_csv(folder / f"{service}.csv.gz", index=False, compression="gzip")
        counts = clean["quality"].value_counts()
        print(f"{service}: " + ", ".join(f"{flag} {int(counts.get(flag, 0))}"
                                         for flag in QUALITY_FLAGS))
    print(f"Written to {folder}")


def cmd_build(args):
    raw = _raw_series(args)
    start, end = _period(args, raw[raw["service"] == args.service])
    analyzer = WorkloadAnalyzer(AlibabaTraceSource(), FileRepository(args.datasets),
                                cleaning=_cleaning(args))
    try:
        ds = analyzer.build_from_raw(raw, args.service, start, end, args.version)
    except (ValueError, FileExistsError) as error:
        sys.exit(str(error))
    cal, fit = ds.calibration, ds.fit
    cpu, lat = cal["cpu_model"], cal["latency_model"]
    p95, err = cal["p95_model"], cal["error_model"]
    print(f"Dataset {ds.location} (data_version {cal['data_version']})")
    print(f"  CPU model     cpu = {cpu['a']:.4g} x load_per_instance + {cpu['b']:.4f}"
          f"   R2 {fit['cpu_r2']:.3f}, test error {fit['cpu_test_error']:.1%}")
    print(f"  Latency model rt = {lat['rt_base']:.4g} + {lat['rt_queue']:.4g} x cpu/(1-cpu) ms"
          f"   test error {fit['latency_test_error']:.1%}")
    print(f"  Calibrated on {fit['minutes']} ok minutes; CPU observed up to "
          f"{lat['observed_cpu_max']:.3f}")
    if (fit["cpu_r2"] or 0) < 0.9 or fit["cpu_test_error"] > 0.10:
        print("  WARNING: the CPU model explains this service poorly; simulations based "
              "on it will be unreliable. Consider another service.")
    p95_text = ("learned from the data" if p95["method"] == "observed"
                else f"assumed: {p95['k']} x mean")
    print(f"  p95 latency   {p95_text}")
    print(f"  Errors        base rate {err['base_error_rate']:.3%} ({err['method']}), "
          f"growing above CPU {err['saturation_cpu']} (assumption)")
    print(f"  CPU basis     {cal['cpu_basis']}; load unit: {cal['load_unit']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scalizer.workload", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download and aggregate the Alibaba trace")
    p.add_argument("--services", required=True, help="comma-separated, e.g. MS_70785,MS_11344")
    p.add_argument("--start", required=True, help="<day>d<hour>, e.g. 0d0")
    p.add_argument("--end", required=True, help="<day>d<hour>, exclusive, e.g. 1d0")
    p.add_argument("--out", required=True, help="work folder")
    p.add_argument("--source", default=BASE_URL,
                   help="Alibaba URL (default) or a local folder with the archives or CSVs")
    p.set_defaults(func=cmd_ingest)

    for name, text in (("clean", "show the result of cleaning (optional)"),
                       ("build", "clean, calibrate and save the dataset")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="work folder used by ingest")
        p.add_argument("--input", help="a raw series made elsewhere (from the Alibaba trace)")
        p.add_argument("--start", help="<day>d<hour> (default: first minute of the series)")
        p.add_argument("--end", help="<day>d<hour> (default: last minute of the series)")
        p.add_argument("--max-gap", type=int, help="longest gap filled, in minutes (default 10)")
        p.add_argument("--startup-minutes", type=int,
                       help="minutes excluded at the start of the trace (default 15)")
        if name == "clean":
            p.add_argument("--services", help="only these services, comma-separated")
            p.set_defaults(func=cmd_clean)
        else:
            p.add_argument("--service", required=True, help="e.g. MS_70785")
            p.add_argument("--version", default="v1", help="dataset version (default v1)")
            p.add_argument("--datasets", default="datasets", help="datasets folder")
            p.set_defaults(func=cmd_build)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
