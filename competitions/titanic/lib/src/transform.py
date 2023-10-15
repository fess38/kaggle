from pathlib import Path

import dacite
from pyarrow import csv

from .parquet_io import convert_table
from .passenger import Label, Passenger

PARSE_OPTIONS = csv.ParseOptions(delimiter=",", quote_char='"')
LABEL_FIELD = "Survived"


def read_passengers(file_name: str | Path) -> list[Passenger]:
    table = csv.read_csv(
        file_name,
        parse_options=PARSE_OPTIONS,
        convert_options=csv.ConvertOptions(strings_can_be_null=True),
    )

    if LABEL_FIELD in table.schema.names:
        table = table.remove_column(table.schema.get_field_index(LABEL_FIELD))

    table = table.rename_columns(Passenger.__dataclass_fields__.keys())

    return convert_table(table, Passenger)


def read_labels(file_name: str | Path) -> list[Label]:
    table = csv.read_csv(
        file_name,
        parse_options=PARSE_OPTIONS,
        convert_options=csv.ConvertOptions(include_columns=["PassengerId", LABEL_FIELD]),
    )

    table = table.rename_columns(Label.__dataclass_fields__.keys())

    return convert_table(table, Label)
