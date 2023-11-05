import logging
from collections import defaultdict
from typing import Iterable, Sequence

import tqdm

from ..io.reader import DatasetReaderBase, create_dataset_reader
from ..io.record_collector import OutputRecordCollector
from ..io.writer import DatasetWriterBase, create_dataset_writer
from ..operations.config import (
    CreateTransformConfigBase,
    MapReduceTransformConfigBase,
    MapTransformConfigBase,
    TransformConfigBase,
)
from ..operations.protocol import CreateFn, MapFn, MapReduceMapFn, MapReduceReduceFn
from .base import BackendBase
from .config import LocalBackendConfig

logger = logging.getLogger(__name__)


class LocalBackend(BackendBase):
    def __init__(self, config: LocalBackendConfig):
        self._config = config

    def run_create(
        self,
        config: CreateTransformConfigBase,
        create_fns: Sequence[CreateFn],
    ):
        logger.info(
            f"Running create operation {config.name or type(config)} using local backend "
            f"with {len(config.outputs)} outputs and "
            f"{len(create_fns)} shards."
        )

        writers = self._create_writers(config)

        output_collector = OutputRecordCollector(config.outputs, writers)
        for create_fn in create_fns:
            output_iterable = create_fn(output_collector)
            output_collector.add_from_iterable(output_iterable)

        self._close_writers(writers)

    def run_map(self, config: MapTransformConfigBase, map_fn: MapFn):
        logger.info(
            f"Running map operation {config.name or type(config)} using local backend with "
            f"{len(config.inputs)} inputs and "
            f"{len(config.outputs)} outputs."
        )

        readers = self._create_readers(config)
        writers = self._create_writers(config)

        for input_ref, reader in zip(config.inputs, readers):
            logger.info(f"Running map function on input <{input_ref}>.")

            output_collector = OutputRecordCollector(config.outputs, writers)
            for input_record in tqdm.tqdm(reader):
                output_iterable = map_fn(input_record, input_ref.role)
                output_collector.add_from_iterable(output_iterable)

        self._close_readers(readers)
        self._close_writers(writers)

    def run_map_reduce(
        self,
        config: MapReduceTransformConfigBase,
        map_fn: MapReduceMapFn,
        reduce_fn: MapReduceReduceFn,
    ):
        logger.info(
            f"Running map-reduce operation {config.name or type(config)} using local backend "
            f"with {len(config.inputs)} inputs and "
            f"{len(config.outputs)} outputs."
        )

        readers = self._create_readers(config)
        writers = self._create_writers(config)

        key_to_recs = defaultdict(list)
        for input_ref, reader in zip(config.inputs, readers):
            logger.info(f"Running map-reduce map function on input <{input_ref}>.")
            for input_record in tqdm.tqdm(reader):
                key, recs = map_fn(input_record, input_ref.role)
                for rec in recs:
                    key_to_recs[key].append(rec)

        logger.info("Running map-reduce reduce function.")
        output_collector = OutputRecordCollector(config.outputs, writers)
        for key, recs in tqdm.tqdm(key_to_recs.items()):
            reduce_fn(key, recs, output_collector)

        self._close_readers(readers)
        self._close_writers(writers)

    def _create_readers(self, config: TransformConfigBase) -> list[DatasetReaderBase]:
        return [create_dataset_reader(input_ref) for input_ref in config.inputs]

    def _create_writers(self, config: TransformConfigBase) -> list[DatasetWriterBase]:
        return [create_dataset_writer(output_ref) for output_ref in config.outputs]

    def _close_readers(self, readers: Iterable[DatasetReaderBase]):
        for reader in readers:
            reader.close()

    def _close_writers(self, writers: Iterable[DatasetWriterBase]):
        for writer in writers:
            writer.close()
