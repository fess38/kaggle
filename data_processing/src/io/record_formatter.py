import abc
from collections.abc import Iterable, Iterator
from typing import IO, Annotated, Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import ujson as json
from fess38.util.config import ConfigBase
from pyarrow import csv


class RecordFormatterBase(ConfigBase, abc.ABC):
    read_mode: str = "rb"
    write_mode: str = "wb"
    column_renames: dict[str, str] = None
    columns_to_delete: list[str] = None

    def read(self, f: IO) -> Iterator[Any]:
        for record in self._read_impl(f):
            self._delete_columns(record)
            self._rename_columns(record)
            yield record

    @abc.abstractmethod
    def _read_impl(self, f: IO) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def write(self, f: IO, records: Iterable[Any]):
        ...

    def _rename_columns(self, record: dict[str, Any]):
        if self.column_renames:
            for old_key, new_key in self.column_renames.items():
                record[new_key] = record.pop(old_key, None)

    def _delete_columns(self, record: dict[str, Any]):
        if self.columns_to_delete:
            for column_to_delete in self.columns_to_delete:
                record.pop(column_to_delete, None)


class CsvRecordFormatter(RecordFormatterBase):
    type: Literal["csv"] = "csv"
    skip_rows: int = 0
    column_names: list[str] = None
    delimiter: str = ","
    quote_char: str = '"'
    strings_can_be_null: bool = True
    include_columns: list[str] = None

    def _read_impl(self, f: IO) -> Iterator[dict[str, Any]]:
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

    def write(self, f: IO, records: Iterable[dict[str, Any]]):
        csv.write_csv(
            pa.Table.from_pylist(records),
            f,
            write_options=csv.WriteOptions(delimiter=self.delimiter),
        )


class JsonlRecordFormatter(RecordFormatterBase):
    type: Literal["jsonl"] = "jsonl"
    read_mode: str = "rt"
    write_mode: str = "wt"

    def _read_impl(self, f: IO) -> Iterator[Any]:
        for row in f:
            yield json.loads(row)

    def write(self, f: IO, records: Iterable[dict[str, Any]]):
        for record in records:
            f.write(f"{json.dumps(record, ensure_ascii=False)}\n")


class ParquetRecordFormatter(RecordFormatterBase):
    type: Literal["parquet"] = "parquet"
    batch_size: int = 2**20
    columns: list[str] | None = None
    compression: Literal["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"] = "SNAPPY"

    def _read_impl(self, f: IO) -> Iterator[Any]:
        parquet_file = pq.ParquetFile(f)
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=self.columns,
        ):
            yield from batch.to_pylist()

    def write(self, f: IO, records: Iterable[Any]):
        pq.write_table(
            table=pa.Table.from_pylist(records),
            where=f,
            compression=self.compression,
        )


RecordFormatter = Annotated[
    (CsvRecordFormatter | JsonlRecordFormatter | ParquetRecordFormatter),
    pydantic.Field(discriminator="type"),
]
