import os
import pandas as pd
import geopandas as gpd
import argparse

from utils import download_utils
from utils import config_utils

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import logging
logging.basicConfig(level=logging.INFO)


def download_data(config, profile):
    logging.info("Downloading UNICEF data...")
    unicef = download_utils.download_unicef(config, profile)

    logging.info("Downloading Overture Maps data...")
    overture_schools = download_utils.download_overture(config, category="school")
    overture_nonschools = download_utils.download_overture(config, category="non_school", exclude="school")

    logging.info("Downloading OSM data...")
    osm_schools = download_utils.download_osm(config, category="school")
    osm_nonschools = download_utils.download_osm(config, category="non_school")

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
    args = parser.parse_args()
    
    # Load config file
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, args.config)
    profile = os.path.join(cwd, args.profile)
    config = config_utils.load_config(config_file)

    # Commence data download
    download_data(config, profile)
    

if __name__ == "__main__":
    main()
