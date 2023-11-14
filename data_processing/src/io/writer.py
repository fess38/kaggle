import abc
from pathlib import Path
from typing import Any

import fsspec
from fess38.util.filesystem import fs_for_path

from .common import FileDatasetIOMixin
from .dataset_reference import FileOutputDatasetReference, OutputDatasetReference


class DatasetWriterBase(abc.ABC):
    @abc.abstractmethod
    def write(self, record: Any):
        ...

    @abc.abstractmethod
    def close(self):
        ...


class FileDatasetWriter(DatasetWriterBase, FileDatasetIOMixin):
    def __init__(self, dataset_reference: FileOutputDatasetReference):
        self._dataset_reference = dataset_reference
        self._validate_record_class()
        fs = fs_for_path(self.data_path)
        self._create_dataset_directory(fs)
        self._file = fs.open(
            self.data_path,
            mode=self.record_formatter.write_mode,
        )
        self._records = []

    def write(self, record: Any):
        self._records.append(record)

    def close(self):
        self.record_formatter.write(self._file, self._records)
        self._records.clear()
        self._file.close()

    def _create_dataset_directory(self, fs: fsspec.AbstractFileSystem):
        if not self.dataset_reference.allow_overwrite and fs.exists(self.data_path):
            raise ValueError(
                f"Output dataset {self.data_path} already exists and overwriting is"
                " disabled."
            )

        fs.makedirs(Path(self.data_path).parent, exist_ok=True)

    @property
    def dataset_reference(self) -> FileOutputDatasetReference:
        return self._dataset_reference


def create_dataset_writer(
    dataset_reference: OutputDatasetReference,
) -> DatasetWriterBase:
    if isinstance(dataset_reference, FileOutputDatasetReference):
        return FileDatasetWriter(dataset_reference)

    raise ValueError(
        f"Unrecognized dataset reference type {type(dataset_reference).__name__}."
    )
