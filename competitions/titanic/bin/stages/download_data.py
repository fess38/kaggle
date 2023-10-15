import logging
import os
import shutil
import subprocess
from pathlib import Path

import hydra
from fess38.competitions.titanic.lib import DvcConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.getcwd(), config_name="dvc_config.yaml", version_base=None)
def main(config: DvcConfig):
    source_data_dir = Path(config.local.work_dir) / config.local.source_data
    os.makedirs(source_data_dir, exist_ok=True)
    os.chdir(source_data_dir)
    subprocess.check_call(f"kaggle competitions download {config.competition_id} -o", shell=True)
    for archive in source_data_dir.glob("*.zip"):
        logger.info(f"Unzipping {archive}")
        shutil.unpack_archive(archive, source_data_dir, "zip")
        archive.unlink()


if __name__ == "__main__":
    main()
