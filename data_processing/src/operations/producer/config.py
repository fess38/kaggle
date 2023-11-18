from typing import Literal

from ...io.dataset_reference import FileInputDatasetReference
from .. import operation_library
from ..config import ProduceOpConfigBase


@operation_library(
    "fess38.data_processing.operations.producer.from_kaggle.FromKaggleProduceOp"
)
class FromKaggleProduceOpConfig(ProduceOpConfigBase):
    type: Literal["from_kaggle"] = "from_kaggle"
    competition_id: str
    file_infos: dict[str, FileInputDatasetReference]
