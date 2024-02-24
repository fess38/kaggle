from pathlib import Path

import pytest
from fess38.data_processing.io.record_formatter import (
    CsvRecordFormatter,
    JsonlRecordFormatter,
    ParquetRecordFormatter,
)
from fess38.util.typing import PyTree


@pytest.mark.parametrize(
    ("params", "records"),
    [
        (
            {
                "skip_rows": 1,
                "column_names": ["a", "b"],
                "delimiter": "\t",
                "include_columns": ["a", "b"],
            },
            [{"a": 1, "b": "c"}, {"a": 2, "b": "d"}],
        ),
        (
            {
                "skip_rows": 1,
                "column_names": ["a", "b"],
                "delimiter": "\t",
                "include_columns": ["a", "b"],
            },
            [{"a": 1, "b": "c"}, {"a": 2, "b": "d"}],
        ),
    ],
)
def test_csv_record_formatter(tmp_path: Path, params: dict, records: list[PyTree]):
    file = tmp_path / "foo.csv"
    record_formatter = CsvRecordFormatter.model_validate(params)
    with file.open(record_formatter.write_mode) as f:
        record_formatter.write(f, records)

    with file.open(record_formatter.read_mode) as f:
        assert list(record_formatter.read(f)) == records


@pytest.mark.parametrize(
    ("params", "records"),
    [
        ({}, [{"a": 1, "b": {"c": "d"}}]),
        ({}, [{"a": 1, "b": {"c": "d"}}, {"a": 2, "b": "d"}]),
    ],
)
def test_jsonl_record_formatter(tmp_path: Path, params: dict, records: list[PyTree]):
    file = tmp_path / "foo.jsonl"
    record_formatter = JsonlRecordFormatter.model_validate(params)
    with file.open(record_formatter.write_mode) as f:
        record_formatter.write(f, records)

    with file.open(record_formatter.read_mode) as f:
        assert list(record_formatter.read(f)) == records


@pytest.mark.parametrize(
    ("params", "records", "expected"),
    [
        ({"paths_to_delete_on_read": {"b"}}, [{"a": 1, "b": 2}], [{"a": 1}]),
        ({"paths_to_delete_on_write": {"b"}}, [{"a": 1, "b": 2}], [{"a": 1}]),
        (
            {"paths_to_move_on_read": {"b": "c.d"}},
            [{"a": 1, "b": 2}],
            [{"a": 1, "c": {"d": 2}}],
        ),
        (
            {"paths_to_move_on_write": {"b": "c.d"}},
            [{"a": 1, "b": 2}],
            [{"a": 1, "c": {"d": 2}}],
        ),
        ({"columns_to_keep_on_read": {"b"}}, [{"a": 1, "b": 2}], [{"b": 2}]),
        ({"columns_to_keep_on_write": {"b"}}, [{"a": 1, "b": 2}], [{"b": 2}]),
    ],
)
def test_base_record_formatter(
    tmp_path: Path, params: dict, records: list[PyTree], expected: list[PyTree]
):
    file = tmp_path / "foo.jsonl"
    record_formatter = JsonlRecordFormatter.model_validate(params)
    with file.open(record_formatter.write_mode) as f:
        record_formatter.write(f, records)

    with file.open(record_formatter.read_mode) as f:
        assert list(record_formatter.read(f)) == expected


@pytest.mark.parametrize(
    ("params", "records"),
    [
        ({}, [{"a": 1, "b": "q"}]),
        ({}, [{"a": 1, "b": "q"}, {"a": 2, "b": "w"}]),
    ],
)
def test_parquet_record_formatter(tmp_path: Path, params: dict, records: list[PyTree]):
    file = tmp_path / "foo.parquet"
    record_formatter = ParquetRecordFormatter.model_validate(params)
    with file.open(record_formatter.write_mode) as f:
        record_formatter.write(f, records)

    with file.open(record_formatter.read_mode) as f:
        assert list(record_formatter.read(f)) == records
