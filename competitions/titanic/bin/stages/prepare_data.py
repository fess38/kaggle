import logging
import os
from pathlib import Path

import hydra
from fess38.competitions.titanic.lib import DvcConfig, read_labels, read_passengers, write_parquet

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.getcwd(), config_name="dvc_config.yaml", version_base=None)
def main(config: DvcConfig):
    source_data_dir = Path(config.local.work_dir) / config.local.source_data
    prepared_data_dir = Path(config.local.work_dir) / config.local.prepared_data
    os.makedirs(prepared_data_dir, exist_ok=True)

    train_passengers = read_passengers(source_data_dir / config.local.source_train)
    test_passengers = read_passengers(source_data_dir / config.local.source_test)
    labels = read_labels(source_data_dir / config.local.source_train)

    write_parquet(prepared_data_dir / config.local.prepared_train, train_passengers)
    write_parquet(prepared_data_dir / config.local.prepared_test, test_passengers)
    write_parquet(prepared_data_dir / config.local.labels, labels)


if __name__ == "__main__":
    main()
