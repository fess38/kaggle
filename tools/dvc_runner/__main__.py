import logging
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def set_gitignore():
    with open(".gitignore", "wt") as f:
        f.write(".hydra/\n")
        f.write("data/\n")
        f.write("stages/\n")
        f.write("tmp/\n")
        f.write("wandb/\n")


def prepare_env(project_dir: str):
    os.environ["DVC_NO_ANALYTICS"] = "1"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["WANDB_PROJECT"] = project_dir.replace("/", "_")


def _prepare_stage(
    stage_id: str,
    stage_name: str,
    operation: dict,
    operation_runner_command: str,
) -> dict:
    path = f"stages/{stage_id}_{stage_name}.yaml"
    with open(path, "wt") as f:
        operation["name"] = stage_name
        f.write(OmegaConf.to_yaml({"ops": [operation]}))

    stage = {
        "desc": stage_name,
        "cmd": f"{operation_runner_command} ++config_path={path}",
        "deps": [path],
        "outs": [],
    }

    for params, stage_param in [("input_files", "deps"), ("output_files", "outs")]:
        for value in operation.get(params, {}).values():
            stage[stage_param].append(value)

    for params, stage_param in [("inputs", "deps"), ("outputs", "outs")]:
        for value in operation.get(params, []):
            stage[stage_param].append(value["path"])

    return stage


def _prepare_stages(
    dvc_config: dict,
    config: DictConfig,
    operation_runner_path: str,
):
    os.makedirs("stages", exist_ok=True)
    dvc_config["stages"] = {}
    operation_runner_command = (
        f"python {hydra.utils.get_original_cwd()}/{operation_runner_path}"
        " ++hydra.run.dir=."
    )
    for index, (stage_name, operation) in enumerate(config.items()):
        stage_id = f"{index + 1:02d}"
        dvc_config["stages"][stage_id] = _prepare_stage(
            stage_id,
            stage_name,
            operation,
            operation_runner_command,
        )


def prepare_dvc_configs(project_dir: str, config_path: str, operation_runner_path: str):
    project_dir = Path(hydra.utils.get_original_cwd()) / project_dir
    config = OmegaConf.to_object(OmegaConf.load(project_dir / config_path))
    if "vars" in config:
        del config["vars"]

    dvc_config = {}
    _prepare_stages(dvc_config, config, operation_runner_path)

    with open("params.yaml", "wt") as f:
        f.write(OmegaConf.to_yaml(config))
    with open("dvc.yaml", "wt") as f:
        f.write(OmegaConf.to_yaml(dvc_config))


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    logger.info(f"Starting dvc experiment in {os.getcwd()}")

    prepare_dvc_configs(
        config.project_dir,
        config.config_path,
        config.operation_runner_path,
    )
    prepare_env(config.project_dir)

    if not Path(".dvc").exists():
        subprocess.check_call("dvc init -q --subdir", shell=True)
        set_gitignore()
    subprocess.check_call(f"{config.dvc_command} {config.dvc_args}", shell=True)


if __name__ == "__main__":
    main()
