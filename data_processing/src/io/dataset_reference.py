from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase

from .record_formatter import RecordFormatter


class DatasetReferenceBase(ConfigBase):
    record_class: str | None = None
    role: str | None = None


class InputDatasetReferenceBase(DatasetReferenceBase):
    ...


class OutputDatasetReferenceBase(DatasetReferenceBase):
    allow_overwrite: bool = False


class FileDatasetReferenceMixin(DatasetReferenceBase):
    type: Literal["file"] = "file"
    path: str
    record_formatter: RecordFormatter


class FileInputDatasetReference(
    FileDatasetReferenceMixin,
    InputDatasetReferenceBase,
):
    ...


class FileOutputDatasetReference(
    FileDatasetReferenceMixin,
    OutputDatasetReferenceBase,
):
    ...


InputDatasetReference = Annotated[
    FileInputDatasetReference,
    pydantic.Field(discriminator="type"),
]

OutputDatasetReference = Annotated[
    FileOutputDatasetReference,
    pydantic.Field(discriminator="type"),
]
