import os
import pandas as pd
import geopandas as gpd
import argparse

from utils import download_utils
from utils import config_utils

import warnings
import logging

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)


def download_data(config, profile, in_file):
    logging.info("Downloading UNICEF data...")
    download_utils.download_unicef(config, profile, in_file=in_file)

    logging.info("Downloading Overture Maps data...")
    download_utils.download_overture(config, category="school")
    download_utils.download_overture(config, category="non_school", exclude="school")

    logging.info("Downloading OSM data...")
    download_utils.download_osm(config, category="school")
    download_utils.download_osm(config, category="non_school")

    logging.info("Downloading Microsoft Building Footprints...")
    download_utils.download_buildings(config, source="ms", verbose=True)
    logging.info("Downloading Google Open Buildings...")
    download_utils.download_buildings(config, source="google", verbose=True)

    logging.info("Downloading GHSL BUILT-C...")
    download_utils.download_ghsl(config, type="built_c")
    logging.info("Downloading GHSL SMOD...")
    download_utils.download_ghsl(config, type="smod")


def main():
    # Load arguments from parser
    parser = argparse.ArgumentParser(description="Data Download")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--profile", help="Path to the profile file")
    parser.add_argument("--in_file", help="Path to the profile file", default=None)
    args = parser.parse_args()

    # Load config file
    config_file = os.path.join(os.getcwd(), args.config)
    profile = os.path.join(os.getcwd(), args.profile)
    config = config_utils.load_config(config_file)

    # Commence data download
    download_data(config, profile, args.in_file)


if __name__ == "__main__":
    main()
