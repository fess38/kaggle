import os

import wandb


def wandb_init(config: dict):
    return wandb.init(
        config=config,
        name=os.getenv("DVC_EXP_NAME"),
        resume="auto",
        id=os.getenv("DVC_EXP_NAME"),
    )
