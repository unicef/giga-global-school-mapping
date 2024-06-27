import os
import copy
import logging
import operator
import joblib

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np

import torch
import torch.nn.functional as nn

import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn.functional as nnf

from PIL import Image
import matplotlib.pyplot as plt
from shapely import geometry
import rasterio as rio
from rasterio.mask import mask
import rasterio.plot

from src import sat_download
from utils import cnn_utils
from utils import data_utils
from utils import config_utils
from utils import model_utils

SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = f"{SEED}"
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def cnn_predict_images(data, model, config, in_dir, classes, threshold):
    files = data_utils.get_image_filepaths(config, data, in_dir)

    preds, probs = [], []
    for file in data_utils.create_progress_bar(files):
        image = Image.open(file).convert("RGB")
        transforms = cnn_utils.get_transforms(config["img_size"])
        output = model(transforms["test"](image).to(device).unsqueeze(0))
        soft_outputs = nnf.softmax(output, dim=1).detach().cpu().numpy()
        probs.append(soft_outputs[:, 1][0])

    preds = np.array(probs)
    preds = preds > threshold
    preds = [str(classes[int(pred)]) for pred in preds]
    data["pred"] = preds
    data["prob"] = probs
    return data


def cnn_predict(
    data, iso_code, shapename, config, threshold, in_dir=None, n_classes=None
):
    cwd = os.path.dirname(os.getcwd())
    config_name = config["config_name"]

    out_dir = data_utils.makedir(
        os.path.join(
            "output", iso_code, "results", config["project"], "tiles", config_name
        )
    )

    name = f"{iso_code}_{shapename}"
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")
    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    classes = {1: config["pos_class"], 0: config["neg_class"]}
    exp_dir = os.path.join(
        cwd, config["exp_dir"], config["project"], f"{iso_code}_{config['config_name']}"
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = load_model(c=config, classes=classes, model_file=model_file)
    results = cnn_predict_images(data, model, config, in_dir, classes, threshold)
    results = results[["UID", "geometry", "pred", "prob"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")
    results.to_file(out_file, driver="GPKG")
    return results


def load_model(iso_code, config, verbose=True, save=True):
    exp_dir = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    classes = {1: config["pos_class"], 0: config["neg_class"]}

    model = cnn_utils.get_model(config["model"], len(classes))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.eval()
    model = model.to(device)

    if verbose:
        logging.info(f"Device: {device}")
        logging.info("Model file {} successfully loaded.".format(model_file))

    return model


def filter_by_buildings(iso_code, config, data, n_seconds=10):
    cwd = os.path.dirname(os.getcwd())
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    google_path = os.path.join(raster_dir, "google_buildings", f"{iso_code}_google.tif")

    pixel_sums = []
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(
        range(len(data)), total=len(data), mininterval=n_seconds, bar_format=bar_format
    )
    for index in pbar:
        subdata = data.iloc[[index]]
        pixel_sum = 0
        try:
            with rio.open(ms_path) as ms_source:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(ms_source, geometry, crop=True)
                image[image == 255] = 1
                pixel_sum = np.sum(image)
        except:
            pass
        if pixel_sum == 0 and os.path.exists(google_path):
            try:
                with rio.open(google_path) as google_source:
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(google_source, geometry, crop=True)
                    image[image == 255] = 1
                    pixel_sum = np.sum(image)
            except:
                pass
        pixel_sums.append(pixel_sum)

    data["sum"] = pixel_sums
    data = data[data["sum"] > 0]
    data = data.reset_index(drop=True)
    return data


def generate_pred_tiles(
    config, iso_code, spacing, buffer_size, shapename, adm_level="ADM2"
):
    out_dir = data_utils.makedir(os.path.join(os.getcwd(), "output", iso_code, "tiles"))
    out_file = os.path.join(out_dir, f"{iso_code}_{shapename}.gpkg")

    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        return data

    points = data_utils.generate_samples(
        config,
        iso_code=iso_code,
        buffer_size=buffer_size,
        spacing=spacing,
        adm_level=adm_level,
        shapename=shapename,
    )

    logging.info(f"Shapename: {shapename}")
    points["points"] = points["geometry"]
    points["geometry"] = points.buffer(buffer_size, cap_style=3)
    points["UID"] = list(points.index)

    filtered = filter_by_buildings(iso_code, config, points)
    filtered = filtered[["UID", "geometry", "shapeName", "sum"]]
    filtered.to_file(out_file, driver="GPKG", index=False)
    return filtered


def batch_process_tiles(
    config, iso_code, spacing, buffer_size, sum_threshold, adm_level
):
    geoboundary = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)
    for i, shapename in enumerate(geoboundary.shapeName.unique()):
        logging.info(f"{shapename} {i}/{len(geoboundary.shapeName.unique())}")
        tiles = generate_pred_tiles(
            config, iso_code, spacing, buffer_size, shapename, adm_level
        )
        tiles = tiles.reset_index(drop=True)
        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > sum_threshold].reset_index(drop=True)
        logging.info(f"Total tiles: {tiles.shape}")
    return tiles


def batch_download_sat_images(
    sat_config,
    sat_creds,
    data_config,
    iso_code,
    spacing,
    buffer_size,
    sum_threshold,
    adm_level,
):
    geoboundary = data_utils.get_geoboundaries(
        data_config, iso_code, adm_level=adm_level
    )
    for i, shapename in enumerate(geoboundary.shapeName.unique()[70:]):
        tiles = generate_pred_tiles(
            data_config, iso_code, spacing, buffer_size, shapename, adm_level
        ).reset_index(drop=True)

        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > sum_threshold].reset_index(drop=True)
        print(
            f"{shapename} {i}/{len(geoboundary.shapeName.unique())} total tiles: {tiles.shape}"
        )

        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(os.getcwd(), "output", iso_code, "images", shapename)
        sat_download.download_sat_images(
            sat_creds, sat_config, data=data, out_dir=sat_dir
        )
