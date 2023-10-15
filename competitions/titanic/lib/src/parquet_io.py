from pathlib import Path
from typing import TypeVar

import dacite
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import asdict

T = TypeVar("T")


def convert_table(table: pa.Table, cls: T) -> list[T]:
    return [dacite.from_dict(data_class=cls, data=item) for item in table.to_pylist()]


def read_parquet(file_name: str | Path, cls: T) -> list[T]:
    return convert_table(pq.read_table(file_name), cls)


def write_parquet(file_name: str | Path, values: list[T]):
    pq.write_table(pa.Table.from_pylist(list(map(asdict, values))), file_name)
