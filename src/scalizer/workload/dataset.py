"""The output: one versioned dataset per service, and where it is stored.

    datasets/<service>_<version>/
    ├── series.csv.gz       clean per-minute series, features and quality flags
    └── calibration.json    the calibrated models, units and data_version

make_dataset() builds a Dataset in memory; FileRepository stores it. A version
is immutable: "MS_70785 v1" always refers to the same data and choices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import pandas as pd

from .calibrate import CPU_FORMULA, LATENCY_FORMULA, add_features, calibrate
from .contracts import CalibrationParams
from .interfaces import RAW_OPTIONAL_COLUMNS

SERIES_COLUMNS = ("minute", "day", "hour_of_day", "load", "containers",
                  "load_per_instance", "cpu", "mem", "rt_ms")


@dataclass
class Dataset:
    service: str
    version: str
    series: pd.DataFrame
    calibration: dict
    location: str | None = field(default=None, compare=False)  # set by the repository
    fit: dict | None = field(default=None, compare=False)      # quality of the fit, not stored

    @property
    def data_version(self) -> str:
        return self.calibration["data_version"]


def make_dataset(service: str, clean: pd.DataFrame, version: str = "v1",
                 source=None, cleaning_params=None,
                 calibration_params: CalibrationParams = CalibrationParams()) -> Dataset:
    """Add features, calibrate, and fingerprint the result.

    `source` is the SourceDescription: its load unit and CPU basis say what the
    numbers of the models mean. The data_version is a hash of the series, of
    both parameter sets and of the forms of the models: any change gives a new one.
    """
    if source is not None and hasattr(source, "model_dump"):
        source = source.model_dump(mode="json")
    if not source:
        raise ValueError("A dataset needs the description of its source "
                         "(units of the load, basis of the CPU)")
    if is_dataclass(cleaning_params):
        cleaning_params = asdict(cleaning_params)

    optional = [c for c in RAW_OPTIONAL_COLUMNS if c in clean]
    series = add_features(clean)[[*SERIES_COLUMNS, *optional, "quality"]]
    models, fit = calibrate(series, calibration_params)

    digest = hashlib.sha256()
    for part in (series.to_csv(index=False).encode(), cleaning_params,
                 asdict(calibration_params), CPU_FORMULA, LATENCY_FORMULA):
        digest.update(part if isinstance(part, bytes)
                      else json.dumps(part, sort_keys=True, default=str).encode())
    calibration = {"service": service, "version": version,
                   "data_version": digest.hexdigest()[:12],
                   "load_unit": source["load_unit"], "cpu_basis": source["cpu_basis"],
                   **models}
    return Dataset(service=service, version=version, series=series,
                   calibration=calibration, fit=fit)


class FileRepository:
    """Stores each dataset in <root>/<service>_<version>/ (interfaces.Repository)."""
    SERIES_FILE = "series.csv.gz"
    CALIBRATION_FILE = "calibration.json"

    def __init__(self, root: str | Path = "datasets"):
        self.root = Path(root)

    def save(self, dataset: Dataset) -> str:
        target = self.root / f"{dataset.service}_{dataset.version}"
        calibration_path = target / self.CALIBRATION_FILE
        if calibration_path.exists():
            existing = json.loads(calibration_path.read_text())["data_version"]
            if existing != dataset.data_version:
                raise FileExistsError(
                    f"{target} already exists with different data (data_version "
                    f"{existing}). Use a new version, e.g. v2.")
        target.mkdir(parents=True, exist_ok=True)
        dataset.series.to_csv(target / self.SERIES_FILE, index=False, compression="gzip")
        calibration_path.write_text(json.dumps(dataset.calibration, indent=2, default=str))
        dataset.location = str(target)
        return str(target)

    def load(self, service: str, version: str) -> Dataset:
        path = self.root / f"{service}_{version}"
        if not (path / self.CALIBRATION_FILE).exists():
            raise FileNotFoundError(f"No dataset at {path}")
        return Dataset(service=service, version=version,
                       series=pd.read_csv(path / self.SERIES_FILE),
                       calibration=json.loads((path / self.CALIBRATION_FILE).read_text()),
                       location=str(path))

    def list(self) -> list[tuple[str, str]]:
        """All stored datasets, as (service, version) pairs."""
        found = []
        for path in sorted(self.root.glob(f"*/{self.CALIBRATION_FILE}")):
            calibration = json.loads(path.read_text())
            found.append((calibration["service"], calibration["version"]))
        return found
