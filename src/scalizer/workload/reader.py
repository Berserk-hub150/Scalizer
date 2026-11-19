"""Reading datasets: what the Forecasting, Simulation and Optimization Engines use.

    reader = DatasetReader(FileRepository("datasets"))
    reader.get_series("MS_70785", "v1", quality=("ok",))
    model = reader.get_service_model("MS_70785", "v1")
    model.predict(load=0.0635, instances=85)

Separate from building (analyzer.py): engines that only read need neither the
source nor the cleaning code. The engines should depend on the interfaces
WorkloadData, ServiceModels and ServiceModel (interfaces.py), not on this class.
"""
from __future__ import annotations

import pandas as pd

from .contracts import Calibration
from .dataset import Dataset
from .interfaces import Repository, ServiceModel
from .model import CalibratedServiceModel


class DatasetReader:
    """interfaces.WorkloadData and interfaces.ServiceModels, over a Repository."""

    def __init__(self, repository: Repository):
        self.repository = repository
        # Never refreshed: safe, because a version never changes.
        self._cache: dict[tuple[str, str], Dataset] = {}

    def _dataset(self, service: str, version: str) -> Dataset:
        key = (service, version)
        if key not in self._cache:
            self._cache[key] = self.repository.load(service, version)
        return self._cache[key]

    def list_datasets(self) -> list[tuple[str, str]]:
        return self.repository.list()

    def get_series(self, service: str, version: str, start: int | None = None,
                   end: int | None = None, quality: tuple[str, ...] | None = None
                   ) -> pd.DataFrame:
        """The clean series, optionally restricted to [start, end) and to some
        quality flags, e.g. quality=("ok",)."""
        series = self._dataset(service, version).series
        if start is not None:
            series = series[series["minute"] >= start]
        if end is not None:
            series = series[series["minute"] < end]
        if quality is not None:
            series = series[series["quality"].isin(quality)]
        return series.reset_index(drop=True)

    def get_calibration(self, service: str, version: str) -> Calibration:
        """calibration.json, validated against its contract."""
        return Calibration.model_validate(self._dataset(service, version).calibration)

    def get_service_model(self, service: str, version: str) -> ServiceModel:
        return CalibratedServiceModel(self.get_calibration(service, version))
