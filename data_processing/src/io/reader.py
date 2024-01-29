import abc
from collections.abc import Iterator
from typing import IO

from fess38.util.typing import PyTree

from .common import FileDatasetIOMixin
from .dataset_reference import FileInputDatasetReference, InputDatasetReference


class DatasetReaderBase(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[PyTree]:
        ...

    @abc.abstractmethod
    def close(self):
        ...


class FileDatasetReader(DatasetReaderBase, FileDatasetIOMixin):
    def __init__(self, dataset_reference: FileInputDatasetReference):
        self._dataset_reference = dataset_reference
        self._data_file: IO = None
        self._validate_record_class()

    def __iter__(self) -> Iterator[PyTree]:
        if self._data_file is None:
            self._data_file = self._fs.open(
                self.data_path,
                mode=self.record_formatter.read_mode,
                compression="infer",
            )

        self._data_file.seek(0)
        return self.record_formatter.read(self._data_file)

    def close(self):
        if self._data_file is not None and not self._data_file.closed:
            self._data_file.close()


def create_dataset_reader(
    dataset_reference: InputDatasetReference,
) -> DatasetReaderBase:
    if isinstance(dataset_reference, FileInputDatasetReference):
        return FileDatasetReader(dataset_reference)

    raise ValueError(
        f"Unrecognized dataset reference type {type(dataset_reference).__name__}."
    )
