from typing import Literal

from ...io.dataset_reference import FileInputDatasetReference
from .. import operation_library
from ..config import CreateOpConfigBase


@operation_library(
    "fess38.data_processing.operations.create.from_kaggle.CreateFromKaggleOp"
)
class CreateFromKaggleOpConfig(CreateOpConfigBase):
    type: Literal["create_from_kaggle"] = "create_from_kaggle"
    competition_id: str
    file_infos: dict[str, FileInputDatasetReference]
