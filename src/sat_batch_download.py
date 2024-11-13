import argparse
import pandas as pd
import geopandas as gpd
import logging
import joblib
import torch

import os 
from src import sat_download
from utils import data_utils
from utils import config_utils
from utils import pred_utils


def batch_download(args):
    data_config = config_utils.load_config(
        os.path.join(
            os.getcwd(), args.data_config
        )
    )
    sat_config = config_utils.load_config(
        os.path.join(
            os.getcwd(), args.sat_config
        )
    )
    sat_creds = config_utils.create_config(
        os.path.join(
            os.getcwd(), args.sat_creds
        )
    )
    pred_utils.batch_download_sat_images(
        sat_config, 
        sat_creds, 
        data_config, 
        args.iso_code, 
        args.spacing, 
        args.buffer_size, 
        args.sum_threshold, 
        args.adm_level
    )


def main():
    # Load arguments from parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--data_config", help="Path to the data configuration file")
    parser.add_argument("--sat_config", help="Path to the satellite configuration file")
    parser.add_argument("--sat_creds", help="Path to the satellite credentials file")
    parser.add_argument("--iso_code", help="ISO 3166-1 alpha-3 code")
    parser.add_argument("--adm_level", default="ADM2", help="Administrative level (default ADM2)")
    parser.add_argument("--sum_threshold", default=5, help="Pixel sum threshold (default 5)")
    parser.add_argument("--buffer_size", default=150, help="Buffer size (default 150)")
    parser.add_argument("--spacing", default=150, help="Sliding window spacing (default 150)")
    args = parser.parse_args()

    # Download satellite images
    batch_download(args)


if __name__ == "__main__":    
    main()
