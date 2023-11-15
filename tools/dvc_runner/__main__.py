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


def _prepare_stage(
    config: DictConfig,
    stage_name: str,
    operation: str | list[str],
    operation_runner_command: str,
) -> dict:
    path = f"stages/{stage_name}.yaml"
    with open(path, "wt") as f:
        operations = operation if isinstance(operation, list) else [operation]
        f.write(OmegaConf.to_yaml({"ops": operations}))

    stage = {
        "cmd": f"{operation_runner_command} ++config_path={path}",
        "deps": [path],
        "outs": [],
    }
    for input in config[stage_name].get("inputs", []):
        stage["deps"].append(input["path"])
    for output in config[stage_name].get("outputs", []):
        stage["outs"].append(output["path"])

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
    for stage_name, operation in config.items():
        dvc_config["stages"][stage_name] = _prepare_stage(
            config,
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

    prepare_dvc_configs(**config)
    if not Path(".dvc").exists():
        subprocess.check_call("dvc init --subdir", shell=True)
        set_gitignore()
    subprocess.check_call("dvc exp run", shell=True)


if __name__ == "__main__":
    main()
