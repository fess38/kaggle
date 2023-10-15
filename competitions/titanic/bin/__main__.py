import logging
import os
import shutil
import subprocess

import hydra
from fess38.competitions.titanic.lib.config import DvcConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def dvc_exp_run():
    subprocess.check_call(f"dvc exp run -S dvc_config.yaml:++hydra.run.dir={os.getcwd()}", shell=True)


def prepare_dvc():
    shutil.copy(f"{hydra.utils.get_original_cwd()}/configs/dvc.yaml", ".")
    logger.info(f"Saved dvc.yaml to {os.getcwd()}/dvc.yaml")
    subprocess.check_call("dvc init -fq --subdir", shell=True)


def build_dvc_config():
    cs = ConfigStore.instance()
    cs.store(name="base_dvc_config", node=DvcConfig)

    dvc_config = hydra.compose(
        config_name="dvc_config",
        overrides=[
            f"local.work_dir={os.getcwd()}",
            f"local.code_dir={hydra.utils.get_original_cwd()}",
        ],
    )
    with open("dvc_config.yaml", "wt") as f:
        f.write(OmegaConf.to_yaml(dvc_config))
        logger.info(f"Saved dvc config to {os.getcwd()}/dvc_config.yaml")


@hydra.main(config_path="./configs", config_name="master_config.yaml", version_base=None)
def main(_: DictConfig):
    logger.info(f"Starting in {os.getcwd()}")
    build_dvc_config()
    prepare_dvc()
    dvc_exp_run()


if __name__ == "__main__":
    main()
