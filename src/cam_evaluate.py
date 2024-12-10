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
    config = config_utils.load_config(args.config)
    best_model_config = model_utils.get_best_models(args.iso_code, config)[0]
    model_config = config_utils.load_config(best_model_config)
    logging.info(f"Best model: {best_model_config}")

    exp_name = f"{args.iso_code}_{model_config['config_name']}"
    exp_dir = os.path.join(
        os.getcwd(), model_config["exp_dir"], model_config["project"], exp_name
    )
    out_file = os.path.join(exp_dir, "cam_results.csv")

    if os.path.exists(out_file):
        results = pd.read_csv(out_file)
        print(results)
        return results

    data = model_utils.load_data(
        model_config,
        iso_code=args.iso_code,
        attributes=["rurban", "iso"],
        verbose=False,
    )
    filepaths = data_utils.get_image_filepaths(
        model_config, data[(data["dataset"] == "test") & (data["class"] == "school")]
    )
    cam_scores_all, cam_scores_mean = cam_utils.compare_cams(
        args.iso_code,
        model_config,
        filepaths,
        float(args.percentile),
        metrics=True,
        show=False,
    )
    results = pd.DataFrame(cam_scores_mean, index=["score"]).T

    results = results.reset_index()
    results.columns = ["method", "score"]
    results.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAM Evaluation")
    parser.add_argument("--iso_code", help="ISO 3166-1 alpha-3 code")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="ISO 3166-1 alpha-3 code"
    )
    parser.add_argument("--percentile", help="Percentile", default=90)
    args = parser.parse_args()

    main(args)
