import logging
from typing import Any, Sequence

from .dataset_reference import OutputDatasetReference
from .record import OutputIterable, OutputRecord
from .writer import DatasetWriterBase

logger = logging.getLogger(__name__)


class OutputRecordCollector:
    def __init__(
        self,
        output_configs: Sequence[OutputDatasetReference],
        writers: Sequence[DatasetWriterBase],
    ):
        if len(output_configs) != len(writers):
            raise ValueError(
                f"Number of output configs ({len(output_configs)}) "
                f"does not match number of writers ({len(writers)})."
            )

        self._writers = writers
        self._role_to_writer: dict[str, DatasetWriterBase] = {}
        for config, writer in zip(output_configs, writers):
            if config.role is not None:
                if config.role in self._role_to_writer:
                    raise ValueError(f"Multiple writers found for role {config.role}.")
                self._role_to_writer[config.role] = writer

    def add(self, record: Any, index=None, role=None):
        if role is None:
            self.add_at_index(record, index if index is not None else 0)
        else:
            if index is not None:
                raise ValueError("Cannot specify both index and role.")
            self.add_for_role(record, role)

    def add_at_index(self, record: Any, index: int):
        if index < 0 or index >= len(self._writers):
            raise ValueError(f"Invalid writer index: {index}")

        self._writers[index].write(record)

    def add_for_role(self, record: Any, role: str):
        if role not in self._role_to_writer:
            raise ValueError(f"Invalid writer role: {role}")

        self._role_to_writer[role].write(record)

    def add_from_iterable(self, iterable: OutputIterable):
        for record in iterable:
            if isinstance(record, OutputRecord):
                self.add(record.value, record.index, record.role)
            else:
                self.add(record)
