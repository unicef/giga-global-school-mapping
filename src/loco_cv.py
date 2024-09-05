import os
import shutil
import time
import argparse
import logging
import pandas as pd
import copy

import torch
import wandb
import torch.nn as nn

from utils import config_utils
import train_model


# Get device
cwd = os.getcwd()

if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument(
        "--lr_finder", help="Learning rate finder (boolean indicator)", default=None
    )
    parser.add_argument(
        "--project", default="GIGAv1", help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.config)
    c = config_utils.load_config(config_file)
    name = c["name"]

    if args.lr_finder:
        args.lr_finder = bool(eval(args.lr_finder))
        c["lr_finder"] = args.lr_finder

    iso_codes = c["iso_codes"]
    for iso_code in iso_codes:
        c["iso_codes"] = [x for x in iso_codes if x != iso_code]
        c["exp_dir"] = "loco/"
        c["project"] = args.project
        c["name"] = f"{name}_{iso_code}"
        c["iso_code"] = c["name"]

        log_c = {
            key: val
            for key, val in c.items()
            if ("url" not in key)
            and ("dir" not in key)
            and ("file" not in key)
            and ("school" not in key)
            and ("exclude" not in key)
            and ("ms_dict" not in key)
        }
        logging.info(log_c)

        wandb.init(project=f"{c['project']}", config=log_c)
        path = os.path.join(cwd, c["exp_dir"], c["project"])
        if not os.path.exists(path):
            train_model.main(c, wandb)
