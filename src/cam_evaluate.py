import os
import logging
import argparse
import pandas as pd

from utils import data_utils
from utils import config_utils
from utils import post_utils
from utils import model_utils
from utils import cam_utils


def main(args):
    model_config = config_utils.load_config(args.model_config)

    data = model_utils.load_data(
        model_config,
        iso_code=args.iso_code,
        attributes=["rurban", "iso"],
        verbose=False,
    )
    filepaths = data_utils.get_image_filepaths(
        model_config, data[data.dataset == "test"]
    )
    cam_scores_all, cam_scores_mean = cam_utils.compare_cams(
        args.iso_code, model_config, filepaths, show=False
    )
    results = pd.DataFrame(cam_scores_mean, index=["score"]).T

    exp_name = f"{model_config['iso_code']}_{model_config['config_name']}"
    exp_dir = os.path.join(
        os.getcwd(), model_config["exp_dir"], model_config["project"], exp_name
    )
    results.to_csv(os.path.join(exp_dir, "cam_results"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAM Evaluation")
    parser.add_argument("--model_config", help="Model config file")
    parser.add_argument("--iso_code", help="ISO 3166-1 alpha-3 code")
    args = parser.parse_args()

    main(args)
