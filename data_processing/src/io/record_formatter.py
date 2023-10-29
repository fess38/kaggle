import abc
from collections.abc import Iterable, Iterator
from typing import IO, Annotated, Any, Literal

import pandas as pd
import pydantic
from fess38.utils.config import ConfigBase


class RecordFormatterBase(ConfigBase, abc.ABC):
    @abc.abstractmethod
    def read(self, f: IO) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def write(self, f: IO, records: Iterable[Any]):
        ...


class TextRecordFormatter(RecordFormatterBase):
    type: Literal["text"] = "text"

    def read(self, f: IO) -> Iterator[dict[str, str]]:
        for line in f:
            yield {"value": line.rstrip("\n")}

    def write(self, f: IO, records: Iterable[dict[str, str]]):
        for record in records:
            f.write(record["value"])
            f.write("\n")


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
    (TextRecordFormatter | ParquetRecordFormatter),
    pydantic.Field(discriminator="type"),
]
