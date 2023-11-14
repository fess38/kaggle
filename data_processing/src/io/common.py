import abc

from fess38.util.reflection import find_class
from pydantic import BaseModel

from .dataset_reference import FileInputDatasetReference, FileOutputDatasetReference
from .record_formatter import RecordFormatter


class FileDatasetIOMixin:
    @property
    @abc.abstractmethod
    def dataset_reference(
        self,
    ) -> FileInputDatasetReference | FileOutputDatasetReference:
        ...

    @property
    def data_path(self) -> str:
        return self.dataset_reference.path

    @property
    def record_formatter(self) -> RecordFormatter:
        return self.dataset_reference.record_formatter

    def _validate_record_class(self):
        if self.dataset_reference.record_class is not None:
            record_class = find_class(self.dataset_reference.record_class)
            if not issubclass(record_class, BaseModel):
                raise TypeError(f"{record_class.__name__} is not BaseModel subclass")
