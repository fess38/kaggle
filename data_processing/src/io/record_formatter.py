import abc
from collections.abc import Iterable, Iterator
from typing import IO, Annotated, Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
from fess38.util.config import ConfigBase
from pyarrow import csv


class RecordFormatterBase(ConfigBase, abc.ABC):
    read_mode: str = "rb"
    write_mode: str = "wb"
    column_renames: dict[str, str] = None

    @abc.abstractmethod
    def read(self, f: IO) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def write(self, f: IO, records: Iterable[Any]):
        ...

    def _rename_columns(self, record: dict[str, Any]):
        if self.column_renames:
            for old_key, new_key in self.column_renames.items():
                record[new_key] = record.pop(old_key, None)

        return record


class CsvRecordFormatter(RecordFormatterBase):
    type: Literal["csv"] = "csv"
    skip_rows: int = 0
    column_names: list[str] = None
    delimiter: str = ","
    quote_char: str = '"'
    strings_can_be_null: bool = True
    include_columns: list[str] = None

    def read(self, f: IO) -> Iterator[dict[str, Any]]:
        records_iterator = csv.read_csv(
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
        yield from (self._rename_columns(record) for record in records_iterator)

    def write(self, f: IO, records: Iterable[dict[str, Any]]):
        csv.write_csv(
            pa.Table.from_pylist(records),
            f,
            write_options=csv.WriteOptions(delimiter=self.delimiter),
        )


class ParquetRecordFormatter(RecordFormatterBase):
    type: Literal["parquet"] = "parquet"
    batch_size: int = 2**20
    columns: list[str] | None = None
    compression: Literal["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"] = "SNAPPY"

    def read(self, f: IO) -> Iterator[Any]:
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
    (CsvRecordFormatter | ParquetRecordFormatter),
    pydantic.Field(discriminator="type"),
]
