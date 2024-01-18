import abc
from functools import cached_property

import fsspec
from fess38.util.filesystem import fs_for_path
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
        return self._dataset_reference.path

    @property
    def record_formatter(self) -> RecordFormatter:
        return self._dataset_reference.record_formatter

    @cached_property
    def _fs(self) -> fsspec.AbstractFileSystem:
        return fs_for_path(self.data_path)

    def _validate_record_class(self):
        if self._dataset_reference.record_class is not None:
            record_class = find_class(self._dataset_reference.record_class)
            if not issubclass(record_class, BaseModel):
                raise TypeError(f"{record_class.__name__} is not BaseModel subclass")
