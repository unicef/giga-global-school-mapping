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


def cnn_predict_images(data: dict, model: torch.nn.Module, config: dict, in_dir: str):
    """
    Predicts probabilities for images using a convolutional neural network (CNN).

    Args:
        data (dict): Dictionary containing the data and results, which will be updated with probabilities.
        model (torch.nn.Module): The CNN model used for predictions.
        config (dict): Configuration dictionary containing model and image settings, including "img_size".
        in_dir (str): Directory path where the images are stored.

    Returns:
        dict: The updated `data` dictionary, now including a "prob" key with the predicted probabilities.
    """
    # Retrieve file paths for images to be processed
    files = data_utils.get_image_filepaths(config, data, in_dir)

    probs = []
    for file in data_utils.create_progress_bar(files):
        # Open and preprocess the image
        image = Image.open(file).convert("RGB")
        transforms = cnn_utils.get_transforms(config["img_size"])

        # Perform the model prediction
        output = model(transforms["test"](image).to(device).unsqueeze(0))

        # Compute the softmax probabilities and extract the probability for the positive class
        soft_outputs = nnf.softmax(output, dim=1).detach().cpu().numpy()
        probs.append(soft_outputs[:, 1][0])

    # Update the data dictionary with the computed probabilities
    data["prob"] = probs
    return data


def cnn_predict(
    data: pd.DataFrame, iso_code: str, shapename: str, config: dict, in_dir: str = None
) -> gpd.GeoDataFrame:
    """
    Predicts probabilities using a CNN model and saves the results to a GeoPackage file.

    Args:
        data (pd.DataFrame): DataFrame containing the data to be processed.
        iso_code (str): ISO code for the region or dataset being processed.
        shapename (str): Name of the shape or region for the output file naming.
        config (dict): Configuration dictionary containing model settings and project details.
        in_dir (str, optional): Directory containing input images. Default is None.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the results with UID, geometry,
            and predicted probabilities.
    """
    # Define the output directory based on the configuration and ISO code
    config_name = config["config_name"]

    # Define the output file path
    out_dir = data_utils.makedir(
        os.path.join(
            "output", iso_code, "results", config["project"], "tiles", config_name
        )
    )

    name = f"{iso_code}_{shapename}"
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")

    # If the results file already exists, read and return it
    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    # Load the model and make predictions
    model = load_model(iso_code, config=config)
    results = cnn_predict_images(data, model, config, in_dir)

    # Prepare and save the results as a GeoDataFrame
    results = results[["UID", "geometry", "prob"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")
    results.to_file(out_file, driver="GPKG")

    return results


def ensemble_predict(
    data: pd.DataFrame,
    iso_code: str,
    shapename: str,
    model_configs: list,
    threshold: float,
    in_dir: str = None,
) -> gpd.GeoDataFrame:
    """
    Aggregates predictions from multiple models and saves the ensemble results to a GeoPackage file.

    Args:
        data (pd.DataFrame): DataFrame containing the data to be processed.
        iso_code (str): ISO code for the region or dataset being processed.
        shapename (str): Name of the shape or region for the output file naming.
        model_configs (list of dicts): List of configuration dictionaries for each model in the ensemble.
        threshold (float): Threshold value for converting probabilities to binary predictions.
        in_dir (str, optional): Directory containing input images. Default is None.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the ensemble results with UID,
            geometry, predicted probabilities, and class predictions.
    """
    # Define class labels based on the positive and negative class configurations from the first model
    classes = {1: model_configs[0]["pos_class"], 0: model_configs[0]["neg_class"]}

    # Create the output directory for saving the results
    out_dir = data_utils.makedir(
        os.path.join(
            "output",
            iso_code,
            "results",
            model_configs[0]["project"],
            "tiles",
            "ensemble",
        )
    )
    # Define the output file path
    name = f"{iso_code}_{shapename}"
    out_file = os.path.join(out_dir, f"{name}_ensemble_results.gpkg")

    # Initialize an array to accumulate probabilities from each model
    probs = 0
    for model_config in model_configs:
        print(f"Generating predictions with {model_config['config_name']}...")
        # Generate predictions for the current model
        results = cnn_predict(
            data=data,
            iso_code=iso_code,
            shapename=shapename,
            config=model_config,
            in_dir=in_dir,
        )
        # Accumulate probabilities from the current model
        probs = probs + results["prob"].to_numpy()

    # Average the probabilities across all models
    results["prob"] = probs / len(model_configs)

    # Convert probabilities to binary predictions based on the threshold
    preds = results["prob"] > threshold
    preds = [str(classes[int(pred)]) for pred in preds]
    results["pred"] = preds

    # Save the results to a GeoPackage file
    results.to_file(out_file, driver="GPKG")
    return results


def load_model(iso_code: str, config: dict, verbose: bool = True) -> torch.nn.Module:
    """
    Loads a pre-trained CNN model from a file and prepares it for evaluation.

    Args:
        iso_code (str): ISO code for the region or dataset, used to locate the model file.
        config (dict): Configuration dictionary containing model settings and paths.
        verbose (bool, optional): If True, logs information about the loading process. Default is True.

    Returns:
        torch.nn.Module: The loaded and prepared model.
    """
    # Construct the path to the model directory and file
    exp_dir = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")

    # Define the class labels based on the configuration
    classes = {1: config["pos_class"], 0: config["neg_class"]}

    # Initialize the model and load the pre-trained weights
    model = cnn_utils.get_model(config["model"], len(classes))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.eval()
    model = model.to(device)

    # Log information if verbose is True
    if verbose:
        logging.info(f"Device: {device}")
        logging.info("Model file {} successfully loaded.".format(model_file))

    return model


def filter_by_buildings(
    iso_code: str, config: dict, data: pd.DataFrame, n_seconds: int = 10
) -> pd.DataFrame:
    """
    Filters the data based on building presence by summing pixel values from building raster files.

    Args:
        iso_code (str): ISO code for the region, used to locate the raster files.
        config (dict): Configuration dictionary containing paths to raster directories.
        data (pd.DataFrame): DataFrame containing the data with geometries to be processed.
        n_seconds (int, optional): Interval in seconds to update the progress bar. Default is 10.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only entries with building presence.
    """
    # Define paths to the multispectral and Google building raster files
    cwd = os.getcwd()
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    google_path = os.path.join(raster_dir, "google_buildings", f"{iso_code}_google.tif")

    pixel_sums = []
    logging.info(f"Filtering buildings for {len(data)}")
    for index in tqdm(list(data.index), total=len(data)):
        # Extract the geometry for the current entry
        subdata = data.iloc[[index]]
        pixel_sum = 0

        # Try reading and processing the Microsoft buildings raster file
        try:
            with rio.open(ms_path) as ms_source:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(ms_source, geometry, crop=True)
                image[image == 255] = 1
                pixel_sum = np.sum(image)
        except:
            pass

        # If no building pixels found, try the Google buildings raster file
        if pixel_sum == 0 and os.path.exists(google_path):
            try:
                with rio.open(google_path) as google_source:
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(google_source, geometry, crop=True)
                    image[image == 255] = 1
                    pixel_sum = np.sum(image)
            except:
                pass

        # Append the pixel sum to the list
        pixel_sums.append(pixel_sum)

    # Update the DataFrame with the pixel sums and filter out entries with no building presence
    data["sum"] = pixel_sums
    data = data[data["sum"] > 0]
    data = data.reset_index(drop=True)

    return data


def generate_pred_tiles(
    config: dict,
    iso_code: str,
    spacing: float,
    buffer_size: float,
    shapename: str,
    adm_level: str = "ADM2",
) -> gpd.GeoDataFrame:
    """
    Generates prediction tiles by creating sample points, filtering them by building presence,
        and saving the results to a GeoPackage file.

    Args:
        config (dict): Configuration dictionary containing paths and parameters for sample generation.
        iso_code (str): ISO code for the region, used to locate and save the output files.
        spacing (float): Spacing between sample points.
        buffer_size (float): Buffer size around each sample point.
        shapename (str): Name of the shape or region for naming the output file.
        adm_level (str, optional): Administrative level for sample generation. Default is "ADM2".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered sample points with building presence.
    """
    # Define the output directory and file path
    out_dir = data_utils.makedir(os.path.join(os.getcwd(), "output", iso_code, "tiles"))
    out_file = os.path.join(out_dir, f"{iso_code}_{shapename}.gpkg")

    # Return the existing file if it already exists
    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        return data

    # Generate sample points based on the configuration and parameters
    points = data_utils.generate_samples(
        config,
        iso_code=iso_code,
        buffer_size=buffer_size,
        spacing=spacing,
        adm_level=adm_level,
        shapename=shapename,
    )

    logging.info(f"Shapename: {shapename}")

    # Create a buffer around each point and add a unique identifier
    points["points"] = points["geometry"]
    points["geometry"] = points.buffer(buffer_size, cap_style=3)
    points["UID"] = list(points.index)

    # Filter points by building presence and select relevant columns
    filtered = filter_by_buildings(iso_code, config, points)
    filtered = filtered[["UID", "geometry", "shapeName", "sum"]]

    # Save the filtered points to a GeoPackage file
    filtered.to_file(out_file, driver="GPKG", index=False)

    return filtered


def batch_process_tiles(
    config: dict,
    iso_code: str,
    spacing: float,
    buffer_size: float,
    sum_threshold: float,
    adm_level: str,
) -> gpd.GeoDataFrame:
    """
    Processes multiple tiles by generating prediction tiles for each shape,
        filtering by building presence, and applying a sum threshold.

    Args:
        config (dict): Configuration dictionary containing paths and parameters for tile generation.
        iso_code (str): ISO code for the region, used to locate and save the output files.
        spacing (float): Spacing between sample points for tile generation.
        buffer_size (float): Buffer size around each sample point.
        sum_threshold (float): Threshold value for filtering tiles based on building pixel sum.
        adm_level (str): Administrative level for sample generation.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all processed tiles that meet the sum threshold criteria.
    """
    # Retrieve geoboundaries for the specified administrative level
    geoboundary = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)
    for i, shapename in enumerate(geoboundary.shapeName.unique()):
        logging.info(f"{shapename} {i}/{len(geoboundary.shapeName.unique())}")

        # Generate prediction tiles for the current shape
        tiles = generate_pred_tiles(
            config, iso_code, spacing, buffer_size, shapename, adm_level
        )

        # Reset index and compute centroid of each tile's geometry
        tiles = tiles.reset_index(drop=True)
        tiles["points"] = tiles["geometry"].centroid

        # Filter tiles based on the sum threshold and reset index
        tiles = tiles[tiles["sum"] > sum_threshold].reset_index(drop=True)
        logging.info(f"Total tiles: {tiles.shape}")

    return tiles


def batch_download_sat_images(
    sat_config: dict,
    sat_creds: dict,
    data_config: dict,
    iso_code: str,
    spacing: float,
    buffer_size: float,
    sum_threshold: float,
    adm_level: str,
) -> None:
    """
    Downloads satellite images for multiple shapes based on generated prediction tiles.

    Args:
        sat_config (dict): Configuration dictionary for satellite image downloads.
        sat_creds (dict): Credentials for accessing satellite image services.
        data_config (dict): Configuration dictionary for data processing and tile generation.
        iso_code (str): ISO code for the region, used to locate and save the downloaded images.
        spacing (float): Spacing between sample points for tile generation.
        buffer_size (float): Buffer size around each sample point.
        sum_threshold (float): Threshold value for filtering tiles based on building pixel sum.
        adm_level (str): Administrative level for sample generation.

    Returns:
        None: This function performs side effects by downloading images and does not return a value.
    """
    # Retrieve geoboundaries for the specified administrative level
    geoboundary = data_utils.get_geoboundaries(
        data_config, iso_code, adm_level=adm_level
    )

    for i, shapename in enumerate(geoboundary.shapeName.unique()):
        # Generate prediction tiles for the current shape
        tiles = generate_pred_tiles(
            data_config, iso_code, spacing, buffer_size, shapename, adm_level
        ).reset_index(drop=True)

        # Compute centroid of each tile's geometry and filter by sum threshold
        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > sum_threshold].reset_index(drop=True)
        print(
            f"{shapename} {i+1}/{len(geoboundary.shapeName.unique())} total tiles: {tiles.shape}"
        )

        # Prepare data for satellite image download
        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(os.getcwd(), "output", iso_code, "images", shapename)

        # Download satellite images for the filtered tiles
        sat_download.download_sat_images(
            sat_creds, sat_config, data=data, out_dir=sat_dir
        )
