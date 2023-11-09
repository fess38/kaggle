import abc
from collections.abc import Iterable, Iterator
from typing import IO, Annotated, Any, Literal

import pandas as pd
import pyarrow
import pydantic
from fess38.util.config import ConfigBase
from pyarrow import csv


class RecordFormatterBase(ConfigBase, abc.ABC):
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


class TextRecordFormatter(RecordFormatterBase):
    type: Literal["text"] = "text"
    default_key: str = "data"

    def read(self, f: IO) -> Iterator[dict[str, str]]:
        for line in f:
            yield {self.default_key: line.rstrip("\n")}

    def write(self, f: IO, records: Iterable[dict[str, str]]):
        for record in records:
            f.write(record[self.default_key])
            f.write("\n")


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
            pyarrow.Table.from_pylist(records),
            f,
            write_options=csv.WriteOptions(delimiter=self.delimiter),
        )


class ParquetRecordFormatter(RecordFormatterBase):
    type: Literal["parquet"] = "parquet"

    def read(self, f: IO) -> Iterator[Any]:
        # TODO read in batches
        df = pd.read_parquet(f)
        for index, row in df.iterrows():
            yield row.to_dict()

    def write(self, f: IO, records: Iterable[Any]):
        # TODO write as table
        df = pd.DataFrame(records)
        df.to_parquet(f)


RecordFormatter = Annotated[
    (CsvRecordFormatter | ParquetRecordFormatter | TextRecordFormatter),
    pydantic.Field(discriminator="type"),
]
