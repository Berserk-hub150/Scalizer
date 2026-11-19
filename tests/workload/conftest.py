"""Shared test helpers."""
import pandas as pd

from scalizer.workload import CleaningParams, CpuBasis, SourceDescription, validate_raw_series


class InMemorySource:
    """A WorkloadSource over a raw series already in memory. Any object with the
    three methods is a source: no inheritance needed."""

    def __init__(self, raw: pd.DataFrame, load_unit="requests per second",
                 cpu_basis=CpuBasis.FRACTION_OF_REQUEST, startup_minutes=0):
        self.raw = raw
        self.load_unit, self.cpu_basis, self.startup_minutes = load_unit, cpu_basis, startup_minutes

    def describe(self):
        return SourceDescription(kind="in-memory", provider="test", load_unit=self.load_unit,
                                 cpu_basis=self.cpu_basis)

    def default_cleaning(self):
        return CleaningParams(startup_minutes=self.startup_minutes)

    def fetch(self, services, start, end, workdir):
        raw = self.raw
        if services:
            raw = raw[raw["service"].isin(services)]
        if start is not None:
            raw = raw[raw["minute"] >= start]
        if end is not None:
            raw = raw[raw["minute"] < end]
        return validate_raw_series(raw.reset_index(drop=True))
