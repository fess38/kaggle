import abc
from pathlib import Path
from typing import IO

from fess38.util.typing import PyTree

from .common import FileDatasetIOMixin
from .dataset_reference import FileOutputDatasetReference, OutputDatasetReference


class DatasetWriterBase(abc.ABC):
    @abc.abstractmethod
    def write(self, record: PyTree):
        ...

    @abc.abstractmethod
    def close(self):
        ...


class FileDatasetWriter(DatasetWriterBase, FileDatasetIOMixin):
    def __init__(self, dataset_reference: FileOutputDatasetReference):
        self._dataset_reference = dataset_reference
        self._data_file: IO = None
        self._records = []
        self._validate_record_class()
        self._validate_allow_overwrite()

    def write(self, record: PyTree):
        if self._data_file is None:
            self._fs.makedirs(Path(self.data_path).parent, exist_ok=True)
            self._data_file = self._fs.open(
                self.data_path, mode=self.record_formatter.write_mode
            )

        self._records.append(record)

    def close(self):
        if self._data_file is not None and not self._data_file.closed:
            self.record_formatter.write(self._data_file, self._records)
            self._records.clear()
            self._data_file.close()

    def _validate_allow_overwrite(self):
        if not self._dataset_reference.allow_overwrite and self._fs.exists(
            self.data_path
        ):
            raise ValueError(
                f"Output dataset '{self.data_path}' already exists and overwriting is"
                " disabled."
            )


def create_dataset_writer(
    dataset_reference: OutputDatasetReference,
) -> DatasetWriterBase:
    if isinstance(dataset_reference, FileOutputDatasetReference):
        return FileDatasetWriter(dataset_reference)

    raise ValueError(
        f"Unrecognized dataset reference type {type(dataset_reference).__name__}."
    )
