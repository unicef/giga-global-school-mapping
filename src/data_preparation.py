import os
import logging
import argparse
import pandas as pd
import geopandas as gpd

from utils import data_utils
from utils import clean_utils
from utils import config_utils

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def main(args):    
    config_file = os.path.join(os.getcwd(), args.config)
    config = config_utils.load_config(config_file)

    for iso_code in data_utils.create_progress_bar(config["iso_codes"]):
        if args.clean_pos:
            clean_utils.clean_data(
                iso_code,
                config=config, 
                name=args.name, 
                category=config["pos_class"], 
                sources=args.sources
            )
            
        if args.clean_neg:
            clean_utils.clean_data(
                iso_code,
                config=config, 
                name=args.name, 
                category=config["neg_class"], 
                sources=args.sources
            )


if __name__ == "__main__":
    # Load arguments from parser
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--name", help="Folder name", default="clean")
    parser.add_argument("--sources", help="Sources (e.g. unicef, osm, overture)", default=[], nargs='+')
    parser.add_argument("--clean_pos", help="Clean positive samples", type=str, default="True")
    parser.add_argument("--clean_neg", help="Clean negative samples", type=str, default="True")
    args = parser.parse_args()

    # Convert to boolean data type
    args.clean_pos = bool(eval(args.clean_pos))
    args.clean_neg = bool(eval(args.clean_neg))
    print(args.sources)
    
    main(args)
