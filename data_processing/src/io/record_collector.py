import logging
from typing import Sequence

from fess38.util.typing import PyTree

from .dataset_reference import OutputDatasetReference
from .record import OutputIterable, OutputRecord
from .writer import DatasetWriterBase

logger = logging.getLogger(__name__)


class OutputRecordCollector:
    def __init__(
        self,
        outputs: Sequence[OutputDatasetReference],
        writers: Sequence[DatasetWriterBase],
    ):
        if len(outputs) != len(writers):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) does not match number of writers"
                f" ({len(writers)})."
            )

        self._outputs = outputs
        self._writers = writers
        self._role_to_writer: dict[str, DatasetWriterBase] = {}
        for output, writer in zip(outputs, writers):
            if output.role is not None:
                if output.role in self._role_to_writer:
                    raise ValueError(f"Multiple writers found for role {output.role}.")
                self._role_to_writer[output.role] = writer

    def add(self, record: PyTree, index=None, role=None):
        if role is None:
            self.add_at_index(record, index if index is not None else 0)
        else:
            if index is not None:
                raise ValueError("Cannot specify both index and role.")
            self.add_for_role(record, role)

    def add_at_index(self, record: PyTree, index: int):
        if index < 0 or index >= len(self._writers):
            raise ValueError(f"Invalid writer index: {index}")

        if self._outputs[index].record_class is not None:
            record = record.dict()

        self._writers[index].write(record)

    def add_for_role(self, record: PyTree, role: str):
        if role not in self._role_to_writer:
            raise ValueError(f"Invalid writer role: {role}")

        self._role_to_writer[role].write(record)

    def add_from_iterable(self, iterable: OutputIterable):
        for record in iterable:
            if isinstance(record, OutputRecord):
                self.add(record.value, record.index, record.role)
            else:
                self.add(record)
