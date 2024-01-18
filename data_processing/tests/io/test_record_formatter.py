from pathlib import Path

import pytest
from fess38.data_processing.io.record_formatter import CsvRecordFormatter


@pytest.mark.parametrize(
    ["csv_str", "params", "expected"],
    [
        ("a,b\n1,c", {"column_renames": {"a": "z"}}, {"z": 1, "b": "c"}),
        ("a,b\n2,d", {"column_renames": {"b": "y"}}, {"a": 2, "y": "d"}),
        (
            "a\tb\td\n1\tc\tp",
            {
                "skip_rows": 1,
                "column_names": ["z", "x", "c"],
                "delimiter": "\t",
                "include_columns": ["x", "c"],
            },
            {"x": "c", "c": "p"},
        ),
        (
            "a\tb\td\n2\t\tf",
            {
                "skip_rows": 1,
                "column_names": ["z", "x", "c"],
                "delimiter": "\t",
                "include_columns": ["z", "x"],
            },
            {"z": 2, "x": None},
        ),
    ],
)
def test_csv_record_formatter(
    csv_str: str, params: dict, expected: dict, tmp_path: Path
):
    csv_file = tmp_path / "foo.csv"
    with csv_file.open("wt") as f:
        f.write(csv_str)

    lines = list(CsvRecordFormatter(**params).read(csv_file))
    assert lines[0] == expected
