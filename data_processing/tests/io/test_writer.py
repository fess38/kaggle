from itertools import product
from pathlib import Path

import pytest
from fess38.data_processing.io.dataset_reference import (
    FileInputDatasetReference,
    FileOutputDatasetReference,
)
from fess38.data_processing.io.reader import FileDatasetReader
from fess38.data_processing.io.record_formatter import (
    CsvRecordFormatter,
    JsonlRecordFormatter,
    ParquetRecordFormatter,
    RecordFormatterBase,
)
from fess38.data_processing.io.writer import FileDatasetWriter


@pytest.mark.parametrize(
    ["records", "record_formatter"],
    list(
        product(
            [CsvRecordFormatter(), JsonlRecordFormatter(), ParquetRecordFormatter()],
            [
                ([{}],),
                ([{"a": 1, "b": 10}, {"a": 2, "b": 11}],),
                ([{"a": 1, "b": ["q", "w"]}, {"a": 2, "b": None}],),
            ],
        )
    ),
)
def test_file_dataset_writer(
    tmp_path: Path, records: list[dict], record_formatter: RecordFormatterBase
):
    path = str(tmp_path / type(record_formatter).__name__)
    record_formatter = JsonlRecordFormatter()

    writer = FileDatasetWriter(
        FileOutputDatasetReference(path=path, record_formatter=record_formatter)
    )
    for record in records:
        writer.write(record)
    writer.close()

    reader = FileDatasetReader(
        FileInputDatasetReference(path=path, record_formatter=record_formatter)
    )
    actual = list(iter(reader))
    reader.close()

    actual == records
