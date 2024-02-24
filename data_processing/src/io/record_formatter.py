import abc
from collections.abc import Iterable, Iterator
from typing import IO, Annotated, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import ujson as json
from fess38.util.config import ConfigBase
from fess38.util.pytree import (
    delete_field_by_path,
    get_field_by_path_safe,
    set_field_by_path,
)
from fess38.util.typing import PyTree, PyTreePath
from pyarrow import csv


class RecordFormatterBase(ConfigBase, abc.ABC):
    read_mode: str = "rb"
    write_mode: str = "wb"
    paths_to_delete_on_read: set[PyTreePath] | None = None
    paths_to_delete_on_write: set[PyTreePath] | None = None
    paths_to_move_on_read: dict[PyTreePath, PyTreePath] = {}
    paths_to_move_on_write: dict[PyTreePath, PyTreePath] = {}
    columns_to_keep_on_read: set[str] | None = None
    columns_to_keep_on_write: set[str] | None = None

    def read(self, f: IO) -> Iterator[PyTree]:
        for record in self._read_impl(f):
            self._delete_paths(record, self.paths_to_delete_on_read)
            self._move_paths(record, self.paths_to_move_on_read)
            self._keep_only_columns(record, self.columns_to_keep_on_read)
            yield record

    @abc.abstractmethod
    def _read_impl(self, f: IO) -> Iterator[PyTree]:
        ...

    def write(self, f: IO, records: Iterable[PyTree]):
        def _post_process_records() -> Iterable[PyTree]:
            for record in records:
                self._delete_paths(record, self.paths_to_delete_on_write)
                self._move_paths(record, self.paths_to_move_on_write)
                self._keep_only_columns(record, self.columns_to_keep_on_write)
                yield record

        self._write_impl(f, _post_process_records())

    @abc.abstractmethod
    def _write_impl(self, f: IO, records: Iterable[PyTree]):
        ...

    def _delete_paths(self, record: PyTree, paths_to_delete: set[PyTreePath]):
        if paths_to_delete:
            for path_to_delete in paths_to_delete:
                delete_field_by_path(record, path_to_delete)

    def _move_paths(self, record: PyTree, paths_to_move: dict[PyTreePath, PyTreePath]):
        if paths_to_move:
            for old_path, new_path in paths_to_move.items():
                value = get_field_by_path_safe(record, old_path)[1]
                delete_field_by_path(record, old_path)
                set_field_by_path(record, new_path, value)

    def _keep_only_columns(self, record: PyTree, columns_to_keep: set[str]):
        if columns_to_keep:
            for column in list(record.keys()):
                if column not in columns_to_keep:
                    delete_field_by_path(record, column)


class CsvRecordFormatter(RecordFormatterBase):
    type: Literal["csv"] = "csv"
    skip_rows: int = 0
    column_names: list[str] | None = None
    delimiter: str = ","
    quote_char: str = '"'
    strings_can_be_null: bool = True
    include_columns: list[str] | None = None

    def _read_impl(self, f: IO) -> Iterator[PyTree]:
        yield from csv.read_csv(
            f,
            read_options=csv.ReadOptions(
                skip_rows=self.skip_rows,
                column_names=self.column_names,
            ),
            parse_options=csv.ParseOptions(
                delimiter=self.delimiter,
                quote_char=self.quote_char,
            ),
            convert_options=csv.ConvertOptions(
                strings_can_be_null=True,
                include_columns=self.include_columns,
            ),
        ).to_pylist()

    def _write_impl(self, f: IO, records: Iterable[PyTree]):
        csv.write_csv(
            pa.Table.from_pylist(list(records)),
            f,
            write_options=csv.WriteOptions(delimiter=self.delimiter),
        )


class JsonlRecordFormatter(RecordFormatterBase):
    type: Literal["jsonl"] = "jsonl"
    read_mode: str = "rt"
    write_mode: str = "wt"

    def _read_impl(self, f: IO) -> Iterator[PyTree]:
        for row in f:
            yield json.loads(row)

    def _write_impl(self, f: IO, records: Iterable[PyTree]):
        for record in records:
            f.write(f"{json.dumps(record, ensure_ascii=False)}\n")


class ParquetRecordFormatter(RecordFormatterBase):
    type: Literal["parquet"] = "parquet"
    batch_size: int = 2**20
    columns: list[str] | None = None
    compression: Literal["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"] = "SNAPPY"

    def _read_impl(self, f: IO) -> Iterator[PyTree]:
        parquet_file = pq.ParquetFile(f)
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=self.columns,
        ):
            yield from batch.to_pylist()

    def _write_impl(self, f: IO, records: Iterable[PyTree]):
        pq.write_table(
            table=pa.Table.from_pylist(list(records)),
            where=f,
            compression=self.compression,
        )


RecordFormatter = Annotated[
    (CsvRecordFormatter | JsonlRecordFormatter | ParquetRecordFormatter),
    pydantic.Field(discriminator="type"),
]
