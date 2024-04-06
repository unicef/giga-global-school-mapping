import os
import pandas as pd
import geopandas as gpd
import logging
import argparse

import sys

sys.path.insert(0, "../utils/")
import clean_utils
import config_utils

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--name", help="Folder name", default="clean")
    parser.add_argument("--sources", help="Folder name", default=[], nargs='+')
    parser.add_argument("--clean_pos", help="Clean positive samples", default="true")
    parser.add_argument("--clean_neg", help="Clean neg samples", default="true")
    args = parser.parse_args()
    
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, args.config)
    config = config_utils.load_config(config_file)

    if args.clean_pos == "true":
        clean_utils.clean_data(
            config, 
            name=args.name, 
            category=config["pos_class"], 
            sources=args.sources
        )
    if args.clean_neg == "true":
        clean_utils.clean_data(
            config, 
            name=args.name, 
            category=config["neg_class"], 
            sources=args.sources
        )


if __name__ == "__main__":
    main()
