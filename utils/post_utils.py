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
    master_dir = os.path.join(cwd, data_config["vectors_dir"], data_config["project"], name)
    master_file = os.path.join(master_dir, f"{iso_code}_{name}.geojson")
    master = gpd.read_file(master_file)
    master = master.rename(columns={'UID': 'MUID'})
    master.loc[(master[name] == 1), name] = 0
    master.loc[(master[name] == 3), name] = 2
    master[name] = master[name].replace({4: 1})
    logging.info(master[name].value_counts())
    logging.info(f"Data dimensions: {master.shape}")
    return master


def load_reference(iso_code, data_config, model_config, sum_threshold=5, adm_level="ADM2", source="pred"):
    cwd = os.path.dirname(os.getcwd())
    
    if "osm" in source:
        filename = os.path.join(
            cwd, data_config["vectors_dir"], 
            data_config["project"],
            data_config["pos_class"], 
            source, 
            f"{iso_code}_{source}.geojson"
        )
        reference = gpd.read_file(filename)
        logging.info(f"Data dimensions: {reference.shape}")
        return reference

    if "unicef" in source:
        filename = os.path.join(
            cwd, 
            data_config["vectors_dir"],
            data_config["project"],
            data_config["pos_class"], 
            source, 
            f"{iso_code}_school_geolocation_coverage_master.csv"
        )
        reference = pd.read_csv(filename).reset_index(drop=True)
        reference["geometry"] = gpd.GeoSeries.from_xy(reference["lon"], reference["lat"])
        reference = gpd.GeoDataFrame(reference, geometry="geometry")
        logging.info(f"Data dimensions: {reference.shape}")
        return reference

    model_config["iso_codes"] = [iso_code]
    out_dir = os.path.join(cwd, "output", iso_code, "results", model_config['project'])
    geoboundary = data_utils._get_geoboundaries(data_config, iso_code, adm_level=adm_level)
    
    data = []
    filenames = next(os.walk(out_dir), (None, None, []))[2] 
    filenames = [filename for filename in filenames if 'cam.gpkg' in filename]
    for filename in filenames:
        subdata = gpd.read_file(os.path.join(out_dir, filename))
        subdata = subdata[subdata["sum"] > sum_threshold]
        data.append(subdata)
    
    data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry", crs="EPSG:3857")
    data = data.drop_duplicates("geometry").reset_index(drop=True)
    
    out_dir = os.path.join(cwd, "output", iso_code)
    data = data.rename(columns={'UID': 'PUID'})
    data.to_file(os.path.join(out_dir, f"{iso_code}_results.geojson"), driver="GeoJSON")
    logging.info(f"Data dimensions: {data.shape}")

    return data