import os
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
import model_utils

import logging
logging.basicConfig(level=logging.INFO)


def load_master(iso_code, data_config, name="unicef_clean"):
    cwd = os.path.dirname(os.getcwd())
    master_dir = os.path.join(cwd, data_config["vectors_dir"], data_config["pos_class"], name)
    master_file = os.path.join(master_dir, f"{iso_code}_{name}.geojson")
    master = gpd.read_file(master_file)
    master = master.rename(columns={'UID': 'MUID'})
    master.loc[(master[name] == 1) | (master[name] == 3), name] = 0
    logging.info(master.unicef_clean.value_counts())
    logging.info(f"Data dimensions: {master.shape}")
    return master


def load_reference(iso_code, data_config, model_config, sum_threshold=5, adm_level="ADM2"):
    model_config["iso_codes"] = [iso_code]
    cwd = os.path.dirname(os.getcwd())
    out_dir = os.path.join(cwd, "output", iso_code, "results")
    geoboundary = data_utils._get_geoboundaries(data_config, iso_code, adm_level=adm_level)
    
    data = []
    for shapename in geoboundary.shapeName.unique():
        filename = f"{iso_code}_{shapename}_{model_config['config_name']}_cam.gpkg"
        subdata = gpd.read_file(os.path.join(out_dir, filename))
        data.append(subdata)
    
    data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry", crs="EPSG:3857")
    data = data.to_crs("EPSG:4326").drop_duplicates("geometry")
    data = data[data["sum"] > sum_threshold].reset_index()
    
    out_dir = os.path.join(cwd, "output", iso_code)
    data = data.rename(columns={'UID': 'PUID'})
    data.to_file(os.path.join(out_dir, f"{iso_code}_results.geojson"), driver="GeoJSON")
    logging.info(f"Data dimensions: {data.shape}")

    return data