from pathlib import Path

import pytest
from fess38.util.config import ConfigBase


class Foo(ConfigBase):
    bar: int
    baz: str


def test_from_file(tmp_path: Path):
    yaml_str = """
        bar: 123
        baz: foo
    """
    config_file_path = tmp_path / "config.yaml"
    with config_file_path.open("wt") as f:
        f.write(yaml_str)

    config = Foo.from_file(config_file_path)
    assert config.bar == 123
    assert config.baz == "foo"


def test_from_file_raises(tmp_path: Path):
    yaml_str = """
        bar: "foo"
        baz: 123
    """
    config_file_path = tmp_path / "config.yaml"
    with config_file_path.open("wt") as f:
        f.write(yaml_str)

    with pytest.raises(ValueError):
        Foo.from_file(config_file_path)
