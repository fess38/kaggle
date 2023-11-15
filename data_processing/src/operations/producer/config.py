from typing import Literal

from ...io.dataset_reference import FileInputDatasetReference
from .. import operation_library
from ..config import ProduceOpConfigBase


@operation_library(
    "fess38.data_processing.operations.producer.from_kaggle.ProduceFromKaggleOp"
)
class ProduceFromKaggleOpConfig(ProduceOpConfigBase):
    type: Literal["produce_from_kaggle"] = "produce_from_kaggle"
    competition_id: str
    file_infos: dict[str, FileInputDatasetReference]
