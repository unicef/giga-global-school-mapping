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

    model_config_file = os.path.join(cwd, args.model_config)
    model_config = config_utils.load_config(model_config_file)

    sat_config_file = os.path.join(cwd, args.sat_config)
    sat_creds_file = os.path.join(cwd, args.sat_creds)
    sat_config = config_utils.load_config(sat_config_file)
    sat_creds = config_utils.create_config(sat_creds_file)

    geoboundary = data_utils.get_geoboundaries(
        data_config, args.iso_code, adm_level="ADM2"
    )
    shapenames = [args.shapename] if args.shapename else geoboundary.shapeName.unique()
    model_config["iso_codes"] = [args.iso_code]

    for shapename in shapenames:
        logging.info(f"Processing {shapename}...")
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
        logging.info(f"Total tiles: {tiles.shape}")

        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(cwd, "output", args.iso_code, "images", shapename)
        logging.info(f"Downloading satellite images for {shapename}...")
        sat_download.download_sat_images(
            sat_creds, sat_config, data=data, out_dir=sat_dir
        )

        print(f"Generating predictions for {shapename}...")
        model_configs = model_utils.get_ensemble_configs(args.iso_code, model_config)

        val_output = model_utils.ensemble_models(args.iso_code, model_config, phase="val")
        val_results = eval_utils.evaluate(
            y_true=val_output["y_true"], 
            y_pred=val_output["y_preds"], 
            y_prob=val_output["y_probs"], 
            pos_label=1, 
            neg_label=0,
            beta=2
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
            args.iso_code,
            model_configs[0],
            subdata,
            geotiff_dir,
            shapename,
            #cam_method=args.cam_method,
        )

    preds = post_utils.load_preds(
        args.iso_code, 
        data_config, 
        model_config, 
        #args.cam_method, 
        sum_threshold=-1,
        buffer_size=25
    )
    post_utils.save_results(
        args.iso_code, 
        preds, 
        model_config, 
        cam_method=args.cam_method, 
        source="preds"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--data_config", help="Data config file")
    parser.add_argument("--model_config", help="Model config file")
    parser.add_argument("--sat_config", help="Maxar config file")
    parser.add_argument("--sat_creds", help="Credentials file")
    #parser.add_argument(
    #    "--cam_method", help="Class activation map method", default=None
    #)
    parser.add_argument("--shapename", help="Model shapename", default=None)
    parser.add_argument("--adm_level", help="Admin level", default="ADM2")
    parser.add_argument("--spacing", help="Tile spacing", default=150)
    parser.add_argument("--buffer_size", help="Buffer size", default=150)
    #parser.add_argument(
    #    "--threshold", type=float, help="Probability threshold", default=0.5
    #)
    parser.add_argument("--sum_threshold", help="Pixel sum threshold", default=5)
    parser.add_argument("--iso_code", help="ISO code")
    args = parser.parse_args()
    logging.info(args)

    main(args)
