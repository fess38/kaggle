import os

import wandb

from .config import ConfigBase


def wandb_init(config: ConfigBase):
    config_dict = config.model_dump()
    return wandb.init(
        config={config_dict.get("name"): config_dict},
        name=os.getenv("DVC_EXP_NAME"),
        resume="auto",
        id=os.getenv("DVC_EXP_NAME"),
    )
