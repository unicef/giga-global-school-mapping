import os
import argparse
import logging
import pandas as pd
import geopandas as gpd
import logging
import joblib
import torch

import sat_download
from utils import config_utils
from utils import model_utils
from utils import eval_utils
from utils import data_utils
from utils import pred_utils
from utils import post_utils
from utils import cam_utils

logging.basicConfig(level=logging.INFO)


def main(args):
    cwd = os.getcwd()

    data_config_file = os.path.join(cwd, args.data_config)
    data_config = config_utils.load_config(data_config_file)

    sat_config_file = os.path.join(cwd, args.sat_config)
    sat_creds_file = os.path.join(cwd, args.sat_creds)

    sat_config = config_utils.load_config(sat_config_file)
    sat_creds = config_utils.create_config(sat_creds_file)

    if args.project:
        data_config["project"] = args.project
        sat_config["project"] = args.project

    if args.shapename:
        shapenames = [args.shapename]
    else:
        shapenames = pred_utils.get_shapenames(
            data_config, args.iso_code, adm_level=args.adm_level
        )

    for index, shapename in enumerate(shapenames[int(args.start_index) :]):
        print(
            f"\nProcessing {shapename} ({int(args.start_index)+index}/{len(shapenames)})..."
        )
        tiles = pred_utils.generate_pred_tiles(
            data_config,
            iso_code=args.iso_code,
            spacing=args.spacing,
            buffer_size=args.buffer_size,
            adm_level=args.adm_level,
            shapename=shapename,
        )
        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > args.sum_threshold].reset_index(drop=True)

        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(cwd, "output", args.iso_code, "images", shapename)
        print(f"Downloading {tiles.shape[0]} satellite images for {shapename} ...")
        sat_download.download_sat_images(
            sat_creds, sat_config, data=data, out_dir=sat_dir
        )

        print(f"Generating predictions for {shapename}...")
        model_configs = model_utils.get_ensemble_configs(args.iso_code, data_config)

        # Calculate the threshold that optimizes the F2 score of the validation set
        val_output = model_utils.ensemble_models(
            args.iso_code, data_config, phase="val"
        )
        val_results = eval_utils.evaluate(
            y_true=val_output["y_true"],
            y_pred=val_output["y_preds"],
            y_prob=val_output["y_probs"],
            beta=2,
        )
        threshold = val_results["optim_threshold"]
        threshold = min(0.5, threshold)

        print(f"Setting threshold to {threshold}...")
        results = pred_utils.ensemble_predict(
            data=tiles,
            iso_code=args.iso_code,
            shapename=shapename,
            model_configs=model_configs,
            threshold=threshold,
            in_dir=sat_dir,
        )

        print(f"Generating GeoTIFFs for {shapename}...")
        subdata = results[results["pred"] == model_configs[0]["pos_class"]]
        geotiff_dir = data_utils.makedir(
            os.path.join("output", args.iso_code, "geotiff", shapename)
        )
        cam_utils.georeference_images(subdata, sat_config, sat_dir, geotiff_dir)

        print(f"Generating CAMs for {shapename}...")
        results = cam_utils.cam_predict(
            args.iso_code, model_configs[0], subdata, geotiff_dir, shapename
        )

    preds = post_utils.load_preds(
        args.iso_code, data_config, buffer_size=args.overlap_buffer_size
    )
    post_utils.save_results(args.iso_code, preds, source="preds", config=data_config)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--data_config", help="Data config file")
    parser.add_argument("--sat_config", help="Maxar config file")
    parser.add_argument("--sat_creds", help="Credentials file")
    parser.add_argument("--shapename", help="Model shapename", default=None)
    parser.add_argument("--adm_level", help="Admin level", default="ADM2")
    parser.add_argument("--spacing", help="Tile spacing", default=150)
    parser.add_argument("--buffer_size", help="Buffer size", default=150)
    parser.add_argument("--overlap_buffer_size", help="Buffer size", default=25)
    parser.add_argument("--sum_threshold", help="Pixel sum threshold", default=0)
    parser.add_argument("--project", help="Overwrite project name", default=None)
    parser.add_argument("--start_index", help="Starting index", default=0)
    parser.add_argument("--iso_code", help="ISO code")
    args = parser.parse_args()
    logging.info(args)

    main(args)
