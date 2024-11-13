import os
import logging
import argparse
import pandas as pd
import geopandas as gpd

from utils import data_utils
from utils import clean_utils
from utils import config_utils
import sat_download

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def main(args):
    config_file = os.path.join(os.getcwd(), args.config)
    config = config_utils.load_config(config_file)

    for iso_code in data_utils.create_progress_bar(config["iso_codes"]):
        category = config["pos_class"] if args.clean_pos else config["neg_class"]
        clean_utils.clean_data(
            iso_code,
            config=config,
            name=args.name,
            category=category,
            sources=args.sources,
        )
        if args.download_sat:
            sat_config = config_utils.load_config(args.sat_config)
            sat_creds = config_utils.create_config(args.sat_creds)
            sat_download.download_sat_images(
                sat_creds, 
                sat_config, 
                category=category, 
                iso_code=iso_code
            )


if __name__ == "__main__":
    # Load arguments from parser
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--sat_config", help="Path to the satellite config file")
    parser.add_argument("--sat_creds", help="Path to the credentials file")
    parser.add_argument("--name", help="Dataset name", default="clean")
    parser.add_argument(
        "--sources", help="Sources (string, e.g. unicef, osm, overture)", default=[], nargs="+"
    )
    parser.add_argument("--imb_ratio", help="Imbalance ratio (int)", type=int, default=2)
    parser.add_argument(
        "--clean_pos", help="Clean positive samples (bool)", type=str, default="True"
    )
    parser.add_argument(
        "--clean_neg", help="Clean negative samples (bool)", type=str, default="True"
    )
    parser.add_argument(
        "--download_sat", help="Download satellite images (bool)", type=str, default="True"
    )
    args = parser.parse_args()

    # Convert to boolean data type
    args.clean_pos = bool(eval(args.clean_pos))
    args.clean_neg = bool(eval(args.clean_neg))
    args.clean_neg = bool(eval(args.download_sat))
    print(args.sources)

    main(args)
