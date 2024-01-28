import logging
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import tqdm
from fess38.util.reflection import find_class

from ..io.reader import DatasetReaderBase, InputDatasetReference, create_dataset_reader
from ..io.record_collector import OutputRecordCollector
from ..io.writer import DatasetWriterBase, create_dataset_writer
from ..operation.config import (
    ConsumeOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
    OpConfigBase,
    ProduceOpConfigBase,
    RunOpConfigBase,
)
from ..operation.protocol import (
    ConsumeAggregatorFn,
    ConsumeFn,
    MapAggregatorFn,
    MapFn,
    MapReduceMapAggregatorFn,
    MapReduceMapFn,
    MapReduceReduceAggregatorFn,
    MapReduceReduceFn,
    ProduceFn,
    RunFn,
)
from .base import BackendBase
from .config import LocalBackendConfig

logger = logging.getLogger(__name__)


def _is_first_parameter_iterable(func: Callable) -> bool:
    return list(func.__annotations__.values())[0].__name__ == "Iterable"


class LocalBackend(BackendBase):
    def __init__(self, config: LocalBackendConfig):
        self._config = config

    def run(self, config: RunOpConfigBase, run_fn: RunFn):
        logger.info(
            f"Running run operation {config.name or type(config)} using local backend"
        )

        run_fn()

    def run_consume(
        self,
        config: ConsumeOpConfigBase,
        consume_fn: ConsumeFn | ConsumeAggregatorFn,
    ):
        logger.info(
            f"Running consume operation {config.name or type(config)} using local"
            f" backend with {len(config.inputs)} inputs."
        )

        readers = self._create_readers(config)
        for input, reader in zip(config.inputs, readers):
            logger.info(f"Running consume function on input <{input}>.")
            self._run_consume_on_input(input, reader, consume_fn)

        self._close_readers(readers)

    def _run_consume_on_input(
        self,
        input: InputDatasetReference,
        reader: DatasetReaderBase,
        consume_fn: ConsumeFn | ConsumeAggregatorFn,
    ):
        record_class = None
        if input.record_class:
            record_class = find_class(input.record_class)

        input_iterable = (
            record_class(**input_record) if record_class else input_record
            for input_record in reader
        )

        if _is_first_parameter_iterable(consume_fn):
            consume_fn(input_iterable, input.role)
        else:
            for input_record in tqdm.tqdm(input_iterable):
                consume_fn(input_record, input.role)

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

    def run_map(self, config: MapOpConfigBase, map_fn: MapFn | MapAggregatorFn):
        logger.info(
            f"Running map operation {config.name or type(config)} using local backend"
            f" with {len(config.inputs)} inputs and {len(config.outputs)} outputs."
        )

        readers = self._create_readers(config)
        writers = self._create_writers(config)
        output_collector = OutputRecordCollector(config.outputs, writers)

        for input, reader in zip(config.inputs, readers):
            logger.info(f"Running map function on input <{input}>.")
            self._run_map_for_input(input, reader, map_fn, output_collector)

        self._close_readers(readers)
        self._close_writers(writers)

    def _run_map_for_input(
        self,
        input: InputDatasetReference,
        reader: DatasetReaderBase,
        map_fn: MapFn | MapAggregatorFn,
        output_collector: OutputRecordCollector,
    ):
        record_class = None
        if input.record_class:
            record_class = find_class(input.record_class)

        input_iterable = (
            record_class(**input_record) if record_class else input_record
            for input_record in reader
        )

        if _is_first_parameter_iterable(map_fn):
            output_iterable = map_fn(input_iterable, input.role)
            output_collector.add_from_iterable(output_iterable)
        else:
            for input_record in tqdm.tqdm(input_iterable):
                output_iterable = map_fn(input_record, input.role)
                output_collector.add_from_iterable(output_iterable)

    def run_map_reduce(
        self,
        config: MapReduceOpConfigBase,
        map_fn: MapReduceMapFn | MapReduceMapAggregatorFn,
        reduce_fn: MapReduceReduceFn | MapReduceReduceAggregatorFn,
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
            self._run_map_reduce_map(input, reader, map_fn, key_to_records)

        logger.info("Running map-reduce reduce function.")
        output_collector = OutputRecordCollector(config.outputs, writers)
        self._run_map_reduce_reduce(key_to_records, output_collector, reduce_fn)

        self._close_readers(readers)
        self._close_writers(writers)

    def _run_map_reduce_map(
        self,
        input: InputDatasetReference,
        reader: DatasetReaderBase,
        map_fn: MapReduceMapFn | MapReduceMapAggregatorFn,
        key_to_records: dict[Any, list],
    ):
        record_class = None
        if input.record_class:
            record_class = find_class(input.record_class)

        input_iterable = (
            record_class(**input_record) if record_class else input_record
            for input_record in reader
        )

        if _is_first_parameter_iterable(map_fn):
            output_iterable = map_fn(input_iterable, input.role)
            for key, record in output_iterable:
                key_to_records[key].append(record)
        else:
            for input_record in tqdm.tqdm(input_iterable):
                output_iterable = map_fn(input_record, input.role)
                for key, record in output_iterable:
                    key_to_records[key].append(record)

    def _run_map_reduce_reduce(
        self,
        key_to_records: dict[Any, list],
        output_collector: OutputRecordCollector,
        reduce_fn: MapReduceReduceFn | MapReduceReduceAggregatorFn,
    ):
        reducer_input_iterable = (
            (
                key,
                (
                    field_value
                    for record in records
                    for field_name, field_value in record.__dict__
                    if field_name != "key" and field_value is not None
                ),
            )
            for key, records in tqdm.tqdm(key_to_records.items())
        )

        if _is_first_parameter_iterable(reduce_fn):
            output_iterable = reduce_fn(reducer_input_iterable)
            output_collector.add_from_iterable(output_iterable)
        else:
            for key, records in tqdm.tqdm(reducer_input_iterable):
                output_iterable = reduce_fn(key, records)
                output_collector.add_from_iterable(output_iterable)

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
