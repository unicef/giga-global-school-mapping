import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import joblib
import torch
import sys

from src import sat_download
from utils import data_utils
from utils import calib_utils
from utils import config_utils
from utils import pred_utils
from utils import model_utils
from utils import eval_utils

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import matplotlib.patches as mpatches
import matplotlib as mpl

from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)


def get_lat_lon_pairs(data):
    lat_lon_pairs = np.empty((len(data), 2))
    for i, row in enumerate(data.itertuples()):
        lat_lon_pairs[i] = [row.geometry.y, row.geometry.x]
    return lat_lon_pairs


def calculate_distance(lat1, lon1, lat2, lon2, R=6371e3):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def calculate_nearest_distance(refs, source, source_name, source_uid):
    tree = cKDTree(get_lat_lon_pairs(source))
    refs = refs.to_crs("EPSG:4326")
    source = source.to_crs("EPSG:4326")
    for index1, ref in refs.iterrows():
        _, index2 = tree.query([ref.geometry.y, ref.geometry.x], 1)
        nearest = source.iloc[index2]
        distance = calculate_distance(
            ref.geometry.y, ref.geometry.x, nearest.geometry.y, nearest.geometry.x
        )
        refs.loc[index1, f"distance_to_nearest_{source_name}"] = distance
        refs.loc[index1, source_uid] = nearest[source_uid]
    return refs


def calculate_nearest_distances(preds, master, osm_overture):
    preds = calculate_nearest_distance(
        preds, master, source_name="master", source_uid="MUID"
    )
    preds = calculate_nearest_distance(
        preds, osm_overture, source_name="osm_overture", source_uid="SUID"
    )
    master = calculate_nearest_distance(
        master, preds, source_name="pred", source_uid="PUID"
    )
    osm_overture = calculate_nearest_distance(
        osm_overture, preds, source_name="pred", source_uid="PUID"
    )
    return preds, master, osm_overture


def join_with_geoboundary(iso_code, data, config, adm_levels=["ADM1", "ADM2", "ADM3"]):
    columns = list(data.columns)
    for adm_level in adm_levels:
        columns = columns + [adm_level]
        admin = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)
        admin = admin.to_crs(data.crs)
        admin = admin.rename(columns={"shapeName": adm_level})
        data = gpd.sjoin(data, admin[["geometry", adm_level]], how="left")
        data = data[columns]
    return data


def load_master(iso_code, config, source="master", colname="clean"):
    filename = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        "sources",
        f"{iso_code}_{source}.geojson",
    )
    data = gpd.read_file(filename)
    data.loc[(data[colname] == 1), colname] = 0
    data.loc[(data[colname] == 3), colname] = 1
    data.loc[data.duplicated("geometry") & (data[colname] == 1), colname] = 1
    data.loc[data.duplicated("geometry") & (data[colname] != 1), colname] = 2
    data = join_with_geoboundary(iso_code, data, config)
    data = data.rename({"UID": "MUID"}, axis=1)
    logging.info(data[colname].value_counts())
    logging.info(f"Data dimensions: {data.shape}")
    return data


def load_osm_overture(iso_code, config, source="osm_overture", colname="clean"):
    filename = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        "sources",
        f"{iso_code}_{source}.geojson",
    )
    data = gpd.read_file(filename).drop_duplicates("UID").reset_index(drop=True)
    data = join_with_geoboundary(iso_code, data[data[colname] == 0], config)
    data = data.rename({"UID": "SUID"}, axis=1)
    logging.info(data[colname].value_counts())
    logging.info(f"Data dimensions: {data.shape}")
    return data


def load_preds(
    iso_code,
    data_config,
    model_config,
    sum_threshold=0,
    buffer_size=0,
    calibrator=None,
    source="pred",
    colname="clean",
):
    out_dir = os.path.join(
        os.getcwd(),
        "output",
        iso_code,
        "results",
        model_config["project"],
        "cams",
        model_config["config_name"],
    )
    filenames = next(os.walk(out_dir), (None, None, []))[2]

    data = []
    for filename in (pbar := data_utils.create_progress_bar(filenames)):
        pbar.set_description(f"Reading files for {iso_code}...")
        subdata = gpd.read_file(os.path.join(out_dir, filename))
        subdata = subdata[subdata["sum"] > sum_threshold]
        data.append(subdata)

    data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry", crs="EPSG:3857")
    data = data.drop_duplicates("geometry").reset_index(drop=True)
    data = data_utils.connect_components(data, buffer_size=buffer_size)
    data = data.sort_values("prob", ascending=False).drop_duplicates(["group"])
    data = join_with_geoboundary(iso_code, data, data_config)

    data["geometry"] = data["geometry"].centroid
    data["PUID"] = data["ADM2"] + "_" + data["UID"].astype(str)
    data = data[~data.duplicated("PUID")]

    if calibrator:
        model = calib_utils.load_calibrator(iso_code, model_config, calibrator)
        data["prob_cal"] = model.transform(data["prob"].values)

    logging.info(f"Data dimensions: {data.shape}")
    return data


def save_results(iso_code, data, config, source):
    out_dir = os.path.join(
        os.getcwd(),
        "output",
        iso_code,
        "results",
        config["project"],
    )
    out_file = f"{iso_code}_{source}.geojson"
    if source == "preds":
        out_dir = os.path.join(out_dir, "cams")
        out_file = f"{iso_code}_{config['config_name']}_cams.geojson"

    out_file = os.path.join(out_dir, out_file)
    data.to_file(out_file, driver="GeoJSON")
