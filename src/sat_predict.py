import os
import argparse
import logging
import pandas as pd
import geopandas as gpd
import logging
import joblib
import torch

import sys
sys.path.insert(0, "../src")
import sat_download

sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import pred_utils
import embed_utils
import calibrators
from torchcam.methods import LayerCAM

logging.basicConfig(level=logging.INFO)


def main(args):
    cwd = os.path.dirname(os.getcwd())
    
    data_config_file = os.path.join(cwd, args.data_config)
    data_config = config_utils.load_config(data_config_file)

    model_config_file = os.path.join(cwd, args.model_config)
    model_config = config_utils.load_config(model_config_file)
    
    sat_config_file = os.path.join(cwd, args.sat_config)
    sat_creds_file = os.path.join(cwd, args.sat_creds)
    sat_config = config_utils.load_config(sat_config_file)
    sat_creds = config_utils.create_config(sat_creds_file)
    
    if not args.cam_model_config:
        cam_model_config = model_config
    else:
        cam_model_config_file = os.path.join(cwd, args.cam_model_config)
        cam_model_config = config_utils.load_config(cam_model_config_file)

    geoboundary = data_utils._get_geoboundaries(
        data_config, args.iso, adm_level="ADM2"
    )
    shapenames = [args.shapename] if args.shapename else geoboundary.shapeName.unique()
    model_config["iso_codes"] = [args.iso]
    
    for shapename in shapenames:
        logging.info(f"Processing {shapename}...")
        tiles = pred_utils.generate_pred_tiles(
            data_config, 
            args.iso, 
            args.spacing, 
            args.buffer_size, 
            args.adm_level, 
            shapename
        )
        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > args.sum_threshold].reset_index(drop=True)
        logging.info(f"Total tiles: {tiles.shape}")
        
        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(cwd, "output", args.iso, "images", shapename)
        logging.info(f"Downloading satellite images for {shapename}...")
        sat_download.download_sat_images(sat_creds, sat_config, data=data, out_dir=sat_dir)
    
        print(f"Generating predictions for {shapename}...")
        results = pred_utils.cnn_predict(
            data=tiles, 
            iso_code=args.iso, 
            shapename=shapename, 
            config=model_config, 
            threshold=args.threshold,
            in_dir=sat_dir, 
            n_classes=2
        )
        
        print(f"Generating GeoTIFFs for {shapename}...")
        subdata = results[results["pred"] == model_config["pos_class"]]
        geotiff_dir = data_utils._makedir(os.path.join("output", args.iso, "geotiff", shapename))
        pred_utils.georeference_images(subdata, sat_config, sat_dir, geotiff_dir)

        print(f"Generating CAMs for {shapename}...")
        config_name = model_config["config_name"]        
        results = pred_utils.cam_predict(
            args.iso, 
            cam_model_config, 
            subdata, 
            geotiff_dir, 
            shapename
        )            
            
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--data_config", help="Data config file")
    parser.add_argument("--model_config", help="Model config file")
    parser.add_argument("--cam_model_config", help="Model config file", default=None)
    parser.add_argument("--sat_config", help="Maxar config file")
    parser.add_argument("--sat_creds", help="Credentials file")
    parser.add_argument("--shapename", help="Model shapename", default=None)
    parser.add_argument("--adm_level", help="Admin level", default="ADM2")
    parser.add_argument("--spacing", help="Tile spacing", default=150)
    parser.add_argument("--buffer_size", help="Buffer size", default=50)
    parser.add_argument("--threshold", type=float, help="Probability threhsold", default=0.5)
    parser.add_argument("--sum_threshold", help="Pixel sum threshold", default=5)
    parser.add_argument("--iso", help="ISO code")
    args = parser.parse_args()
    logging.info(args)

    main(args)

    