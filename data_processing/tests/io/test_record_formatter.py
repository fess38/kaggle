from pathlib import Path

from fess38.data_processing.io.record_formatter import CsvRecordFormatter


def test_csv_record_formatter_default(tmp_path: Path):
    csv_file = tmp_path / "foo.csv"
    with csv_file.open("wt") as f:
        f.write("a,b\n1,c\n2,d")

    lines = list(CsvRecordFormatter().read(csv_file))
    assert len(lines) == 2
    assert lines[0] == {"a": 1, "b": "c"}
    assert lines[1] == {"a": 2, "b": "d"}


def test_csv_record_formatter_custom(tmp_path: Path):
    csv_file = tmp_path / "foo.csv"
    with csv_file.open("wt") as f:
        f.write("a\tb\td\n1\tc\tp\n2\t\tf")

    lines = list(
        CsvRecordFormatter(
            skip_rows=1,
            column_names=["z", "x", "c"],
            delimiter="\t",
            include_columns=["x", "c"],
        ).read(csv_file)
    )
    assert len(lines) == 2
    assert lines[0] == {"x": "c", "c": "p"}
    assert lines[1] == {"x": None, "c": "f"}
