"""Building datasets.

    analyzer = WorkloadAnalyzer(AlibabaTraceSource(), FileRepository("datasets"))
    analyzer.build("MS_70785", start=0, end=1440, version="v1")

The source and the repository are passed in: the analyzer depends only on their
interfaces (interfaces.py), so another data provider or a database can replace
them without changing it. Reading datasets is the job of reader.DatasetReader.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clean import clean_series
from .contracts import CalibrationParams, CleaningParams
from .dataset import Dataset, make_dataset
from .interfaces import Repository, WorkloadSource, validate_raw_series


class WorkloadAnalyzer:
    def __init__(self, source: WorkloadSource, repository: Repository,
                 workdir: str | Path = "data",
                 cleaning: CleaningParams | None = None,
                 calibration: CalibrationParams = CalibrationParams()):
        self.source = source
        self.repository = repository
        self.workdir = Path(workdir)           # cache of the source (e.g. the archives read)
        self.cleaning = cleaning               # None: the defaults of the source
        self.calibration = calibration

    def build(self, service: str, start: int | None, end: int | None,
              version: str = "v1") -> Dataset:
        """Fetch the data from the source, then build_from_raw."""
        raw = self.source.fetch([service], start, end, self.workdir / service)
        return self.build_from_raw(raw, service, start, end, version)

    def build_from_raw(self, raw: pd.DataFrame, service: str, start: int | None = None,
                       end: int | None = None, version: str = "v1") -> Dataset:
        """Clean, calibrate and store a raw series already fetched (for example
        the one saved by the ingest command)."""
        data = validate_raw_series(raw)
        data = data[data["service"] == service]
        if data.empty:
            raise ValueError(f"No data for {service}")
        # Period: the one asked for, otherwise from the first to the last minute.
        start = start if start is not None else int(data["minute"].min())
        end = end if end is not None else int(data["minute"].max()) + 1
        params = self.cleaning or self.source.default_cleaning()

        clean = clean_series(data, start, end, params)
        dataset = make_dataset(service, clean, version, source=self.source.describe(),
                               cleaning_params=params, calibration_params=self.calibration)
        self.repository.save(dataset)   # refused if the version exists with other data
        return dataset

    def list_datasets(self) -> list[tuple[str, str]]:
        return self.repository.list()
