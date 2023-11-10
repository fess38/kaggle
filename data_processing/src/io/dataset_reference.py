from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase

from .record_formatter import RecordFormatter


class DatasetReferenceBase(ConfigBase):
    role: str | None = None


class InputDatasetReferenceBase(DatasetReferenceBase):
    ...


class OutputDatasetReferenceBase(DatasetReferenceBase):
    allow_overwrite: bool = False


class FileDatasetReferenceMixin(DatasetReferenceBase):
    type: Literal["file"] = "file"
    path: str


class FileInputDatasetReference(
    FileDatasetReferenceMixin,
    InputDatasetReferenceBase,
):
    record_formatter: RecordFormatter


class FileOutputDatasetReference(
    FileDatasetReferenceMixin,
    OutputDatasetReferenceBase,
):
    record_formatter: RecordFormatter


InputDatasetReference = Annotated[
    FileInputDatasetReference,
    pydantic.Field(discriminator="type"),
]

OutputDatasetReference = Annotated[
    FileOutputDatasetReference,
    pydantic.Field(discriminator="type"),
]
