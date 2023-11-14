import abc
from collections.abc import Iterator
from typing import Any

from fess38.util.filesystem import fs_for_path

from .common import FileDatasetIOMixin
from .dataset_reference import FileInputDatasetReference, InputDatasetReference


class DatasetReaderBase(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def close(self):
        ...


class FileDatasetReader(DatasetReaderBase, FileDatasetIOMixin):
    def __init__(self, dataset_reference: FileInputDatasetReference):
        self._dataset_reference = dataset_reference
        self._validate_record_class()
        fs = fs_for_path(self.data_path)
        self._data_file = fs.open(self.data_path, mode=self.record_formatter.read_mode)

    def __iter__(self) -> Iterator[Any]:
        self._data_file.seek(0)
        return self.record_formatter.read(self._data_file)

    def close(self):
        self._data_file.close()

    @property
    def dataset_reference(self) -> FileInputDatasetReference:
        return self._dataset_reference


def create_dataset_reader(
    dataset_reference: InputDatasetReference,
) -> DatasetReaderBase:
    if isinstance(dataset_reference, FileInputDatasetReference):
        return FileDatasetReader(dataset_reference)

    raise ValueError(
        f"Unrecognized dataset reference type {type(dataset_reference).__name__}."
    )
