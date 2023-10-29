from typing import Annotated, Literal

import pydantic
from fess38.utils.config import ConfigBase

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
    mode: str


class FileInputDatasetReference(
    FileDatasetReferenceMixin,
    InputDatasetReferenceBase,
):
    mode: str = "rb"
    record_formatter: RecordFormatter


class FileOutputDatasetReference(
    FileDatasetReferenceMixin,
    OutputDatasetReferenceBase,
):
    mode: str = "wb"
    record_formatter: RecordFormatter


InputDatasetReference = Annotated[
    FileInputDatasetReference,
    pydantic.Field(discriminator="type"),
]

OutputDatasetReference = Annotated[
    FileOutputDatasetReference,
    pydantic.Field(discriminator="type"),
]
