import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class Config:
    project_dir: str
    params_path: str
    dvc_config_path: str
    operation_runner_path: str


def _set_operation_runner(dvc_config: DictConfig, operation_runner_path: str):
    command = f"python {hydra.utils.get_original_cwd()}/{operation_runner_path} ++hydra.run.dir=."
    dvc_config["vars"].append({"operation_runner": command})


def _expand_deps_outs(dvc_config: DictConfig, params: DictConfig):
    for stage_name, stage in dvc_config["stages"].items():
        if stage_name not in params:
            continue

        if "deps" not in stage:
            stage["deps"] = []
        for input in params[stage_name]["inputs"]:
            stage["deps"].append(input["path"])

        if "outs" not in stage:
            stage["outs"] = []
        for output in params[stage_name]["outputs"]:
            stage["outs"].append(output["path"])


def _expand_cmd(dvc_config: DictConfig):
    for stage_name, stage in dvc_config["stages"].items():
        if isinstance(stage["cmd"], str):
            stage["cmd"] = [stage["cmd"]]

        for i in range(len(stage["cmd"])):
            if "${operation_runner}" in stage["cmd"][i]:
                stage["cmd"][i] += f" ++config_path=stages/{stage_name}.yaml"


def _split_params_to_stages(params: DictConfig):
    for stage_name, command in params.items():
        with open(f"stages/{stage_name}.yaml", "wt") as f:
            f.write(OmegaConf.to_yaml({"transforms": command if isinstance(command, list) else [command]}))


def prepare_dvc_configs(config: Config):
    project_dir = Path(hydra.utils.get_original_cwd()) / config.project_dir
    params = OmegaConf.to_object(OmegaConf.load(project_dir / config.params_path))
    if "vars" in params:
        del params["vars"]
    _split_params_to_stages(params)

    dvc_config = OmegaConf.to_container(OmegaConf.load(project_dir / config.dvc_config_path), resolve=False)
    _set_operation_runner(dvc_config, config.operation_runner_path)
    _expand_cmd(dvc_config)
    _expand_deps_outs(dvc_config, params)

    with open("params.yaml", "wt") as f:
        f.write(OmegaConf.to_yaml(params))
    with open("dvc.yaml", "wt") as f:
        f.write(OmegaConf.to_yaml(dvc_config))


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    logger.info(f"Starting dvc experiment in {os.getcwd()}")

    config = Config(**config)
    prepare_dvc_configs(config)
    if not Path(".dvc").exists():
        subprocess.check_call("dvc init --subdir", shell=True)
    subprocess.check_call(f"dvc exp run", shell=True)


if __name__ == "__main__":
    main()