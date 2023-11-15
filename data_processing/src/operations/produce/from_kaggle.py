import logging
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from ...io.dataset_reference import FileInputDatasetReference
from ...io.reader import create_dataset_reader
from ...io.record import OutputIterable, OutputRecord
from ..protocol import ProduceFn
from .base import ProduceOpBase
from .config import ProduceFromKaggleOpConfig

logger = logging.getLogger(__name__)


class ProduceFromKaggleOp(ProduceOpBase):
    def __init__(self, config: ProduceFromKaggleOpConfig):
        super().__init__(config, [self._create_produce_fn(config)])

    def _create_produce_fn(self, config: ProduceFromKaggleOpConfig) -> ProduceFn:
        def produce_fn() -> OutputIterable:
            api = KaggleApi()
            api.authenticate()

            for file_name, dataset_reference in config.file_infos.items():
                logger.info(
                    f"Start downloading file '{file_name}' to"
                    f" '{dataset_reference.path}'"
                )
                dir = Path(dataset_reference.path).parent
                api.competition_download_file(
                    config.competition_id, file_name, str(dir)
                )
                yield from self._process_file(file_name, dataset_reference)

        return produce_fn

    def _process_file(
        self, file_name: str, dataset_reference: FileInputDatasetReference
    ) -> OutputIterable:
        reader = create_dataset_reader(dataset_reference)
        for record in reader:
            yield OutputRecord(value=record, role=file_name)
