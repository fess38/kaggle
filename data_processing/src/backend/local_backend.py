import logging
from collections import defaultdict
from typing import Iterable, Sequence

import tqdm
from fess38.util.reflection import find_class

from ..io.reader import DatasetReaderBase, create_dataset_reader
from ..io.record_collector import OutputRecordCollector
from ..io.writer import DatasetWriterBase, create_dataset_writer
from ..operations.config import (
    ConsumeOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
    OpConfigBase,
    ProduceOpConfigBase,
)
from ..operations.protocol import (
    ConsumeFn,
    MapFn,
    MapReduceMapFn,
    MapReduceReduceFn,
    ProduceFn,
)
from .base import BackendBase
from .config import LocalBackendConfig

logger = logging.getLogger(__name__)


class LocalBackend(BackendBase):
    def __init__(self, config: LocalBackendConfig):
        self._config = config

    def run_consume(
        self,
        config: ConsumeOpConfigBase,
        consume_fn: ConsumeFn,
    ):
        logger.info(
            f"Running consume operation {config.name or type(config)} using local"
            f" backend with {len(config.inputs)} inputs."
        )

        readers = self._create_readers(config)
        for input, reader in zip(config.inputs, readers):
            logger.info(f"Running consume function on input <{input}>.")

            for input_record in tqdm.tqdm(reader):
                if input.record_class is not None:
                    record_class = find_class(input.record_class)
                    input_record = record_class(**input_record)
                consume_fn(input_record, input.role)

        self._close_readers(readers)

    def run_produce(
        self,
        config: ProduceOpConfigBase,
        produce_fns: Sequence[ProduceFn],
    ):
        logger.info(
            f"Running produce operation {config.name or type(config)} using local"
            f" backend with {len(config.outputs)} outputs and {len(produce_fns)}"
            " shards."
        )

        writers = self._create_writers(config)

        output_collector = OutputRecordCollector(config.outputs, writers)
        for produce_fn in produce_fns:
            output_iterable = produce_fn()
            output_collector.add_from_iterable(output_iterable)

        self._close_writers(writers)

    def run_map(self, config: MapOpConfigBase, map_fn: MapFn):
        logger.info(
            f"Running map operation {config.name or type(config)} using local backend"
            f" with {len(config.inputs)} inputs and {len(config.outputs)} outputs."
        )

        readers = self._create_readers(config)
        writers = self._create_writers(config)

        for input, reader in zip(config.inputs, readers):
            logger.info(f"Running map function on input <{input}>.")

            output_collector = OutputRecordCollector(config.outputs, writers)
            for input_record in tqdm.tqdm(reader):
                if input.record_class is not None:
                    record_class = find_class(input.record_class)
                    input_record = record_class(**input_record)
                output_iterable = map_fn(input_record, input.role)
                output_collector.add_from_iterable(output_iterable)

        self._close_readers(readers)
        self._close_writers(writers)

    def run_map_reduce(
        self,
        config: MapReduceOpConfigBase,
        map_fn: MapReduceMapFn,
        reduce_fn: MapReduceReduceFn,
    ):
        logger.info(
            f"Running map-reduce operation {config.name or type(config)} using local"
            f" backend with {len(config.inputs)} inputs and {len(config.outputs)}"
            " outputs."
        )

        readers = self._create_readers(config)
        writers = self._create_writers(config)

        key_to_records = defaultdict(list)
        for input, reader in zip(config.inputs, readers):
            logger.info(f"Running map-reduce map function on input <{input}>.")
            for input_record in tqdm.tqdm(reader):
                if input.record_class is not None:
                    record_class = find_class(input.record_class)
                    input_record = record_class(**input_record)
                key, records = map_fn(input_record, input.role)
                for record in records:
                    key_to_records[key].append(record)

        logger.info("Running map-reduce reduce function.")
        output_collector = OutputRecordCollector(config.outputs, writers)
        for key, records in tqdm.tqdm(key_to_records.items()):
            records = (
                field_value
                for record in records
                for field_name, field_value in record.__dict__
                if field_name != "key" and field_value is not None
            )
            output_iterable = reduce_fn(key, records)
            output_collector.add_from_iterable(output_iterable)

        self._close_readers(readers)
        self._close_writers(writers)

    def _create_readers(self, config: OpConfigBase) -> list[DatasetReaderBase]:
        return [create_dataset_reader(input_ref) for input_ref in config.inputs]

    def _create_writers(self, config: OpConfigBase) -> list[DatasetWriterBase]:
        return [create_dataset_writer(output_ref) for output_ref in config.outputs]

    def _close_readers(self, readers: Iterable[DatasetReaderBase]):
        for reader in readers:
            reader.close()

    def _close_writers(self, writers: Iterable[DatasetWriterBase]):
        for writer in writers:
            writer.close()
