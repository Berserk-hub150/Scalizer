"""Workload Analyzer (component C1).

Turns the Alibaba trace into one versioned dataset per service: the clean
per-minute series (series.csv.gz) and the calibrated models (calibration.json).

    interfaces.py  required: WorkloadSource, Repository; provided: WorkloadData,
                   ServiceModels, ServiceModel; the raw-series format
    contracts.py   parameters and typed contracts (calibration.json)
    alibaba.py     AlibabaTraceSource: reads the trace
    clean.py       clean_series: one row per minute, a quality flag per minute
    calibrate.py   add_features, calibrate: the models of the service
    dataset.py     make_dataset, Dataset, FileRepository
    model.py       CalibratedServiceModel: predict, sample_parameters, instances_for
    analyzer.py    WorkloadAnalyzer: builds datasets
    reader.py      DatasetReader: reads datasets, for the other engines
    __main__.py    command line: ingest, clean, build

Other components import from here:

    from scalizer.workload import DatasetReader, FileRepository
"""
from .alibaba import AlibabaTraceSource
from .analyzer import WorkloadAnalyzer
from .calibrate import add_features, calibrate
from .clean import clean_series
from .contracts import (
    Calibration, CalibrationParams, CleaningParams, CpuBasis, ModelParams, Prediction,
    SourceDescription,
)
from .dataset import Dataset, FileRepository, make_dataset
from .interfaces import (
    Repository, ServiceModel, ServiceModels, WorkloadData, WorkloadSource, validate_raw_series,
)
from .model import CalibratedServiceModel
from .reader import DatasetReader

__all__ = [
    "WorkloadAnalyzer", "DatasetReader", "CalibratedServiceModel",
    "WorkloadSource", "Repository", "WorkloadData", "ServiceModels", "ServiceModel",
    "AlibabaTraceSource", "FileRepository",
    "Calibration", "CalibrationParams", "CleaningParams", "CpuBasis", "ModelParams",
    "Prediction", "SourceDescription", "Dataset", "validate_raw_series",
    "add_features", "calibrate", "clean_series", "make_dataset",
]
