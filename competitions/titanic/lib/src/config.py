from dataclasses import dataclass


@dataclass
class BaseLocalConfig:
    work_dir: str = "."
    code_dir: str = "."
    source_data: str = "source_data"
    prepared_data: str = "prepared_data"


@dataclass
class LocalConfig(BaseLocalConfig):
    source_train: str = "train.csv"
    source_test: str = "test.csv"
    prepared_train: str = "train.parquet"
    prepared_test: str = "test.parquet"
    labels: str = "labels.parquet"


@dataclass
class DvcConfig:
    competition_id: str
    local: LocalConfig = LocalConfig()
