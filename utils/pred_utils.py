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
from exactextract import exact_extract

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
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.geojson")

    # If the results file already exists, read and return it
    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    # Load the model and make predictions
    model = load_model(iso_code, config=config)
    results = cnn_predict_images(data, model, config, in_dir)

    # Prepare and save the results as a GeoDataFrame
    results = results[["UID", "geometry", "prob"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")
    results.to_file(out_file, driver="GeoJSON")

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
    out_file = os.path.join(out_dir, f"{iso_code}_{shapename}_ensemble_results.geojson")

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
        if len(results) > 0:
            # Accumulate probabilities from the current model
            probs = probs + results["prob"].to_numpy()

    # Average the probabilities across all models
    results["prob"] = probs / len(model_configs)

    # Convert probabilities to binary predictions based on the threshold
    preds = results["prob"] > threshold
    preds = [str(classes[int(pred)]) for pred in preds]
    results["pred"] = preds

    # Save the results to a GeoPackage file
    results.to_file(out_file, driver="GeoJSON")
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
        print(f"Device: {device}")
        print("Model file {} successfully loaded.".format(model_file))

    return model


def filter_uninhabited(
    iso_code: str, config: dict, data: gpd.GeoDataFrame, in_vector: str = None
) -> pd.DataFrame:
    """
    Filters the data based on building presence by summing pixel values from building raster files.

    Args:
        iso_code (str): ISO code for the region, used to locate the raster files.
        config (dict): Configuration dictionary containing paths to raster directories.
        data (gpd.GeoDataFrame): DataFrame containing the data with geometries to be processed

    Returns:
        pd.DataFrame: Filtered DataFrame containing only entries with building presence.
    """

    def count_points(tiles: gpd.GeoDataFrame, path: str):
        """
        Counts the number of points from a given file that intersect with each polygon
        in the provided GeoDataFrame.

        Args:
            tiles (gpd.GeoDataFrame): A GeoDataFrame containing polygons with a "UID" column to group by.
            path (str): File path to a GeoJSON or similar file containing point geometries.

        Returns:
            Series: A pandas Series indexed by "UID", where each value represents the count
            of points intersecting with the corresponding polygon.
        """
        # Read the input file containing point data as a GeoDataFrame
        tiles_ = tiles.copy()
        points = gpd.GeoDataFrame.from_file(path)

        # Convert the geometry of the points to their centroids
        points["geometry"] = points.centroid

        # Count number of points in each tile
        bldg_sum = tiles_.join(
            gpd.sjoin(points, tiles).groupby("UID").size().rename("sum"),
            how="left",
        )
        bldg_sum["sum"] = bldg_sum["sum"].fillna(0)

        return bldg_sum["sum"].values

    # Get the current working directory
    cwd = os.getcwd()

    # Define the path to the raster directory
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    vector_dir = os.path.join(cwd, config["vectors_dir"])

    # Define the file paths for different building datasets (e.g., Microsoft, Google, GHSL)
    ms_path = os.path.join(
        vector_dir, "ms_buildings", f"{iso_code}_ms_EPSG3857.geojson"
    )
    google_path = os.path.join(
        vector_dir, "google_buildings", f"{iso_code}_google_EPSG3857.geojson"
    )
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])

    # Reset the index of the data to ensure consistency
    data = data.reset_index(drop=True)

    # Log the number of records being processed
    print(f"Filtering uninhabited locations for {len(data)} tiles...")

    # If the Microsoft building data exists, extract the sum of building areas within the vector
    ms_sum = 0
    if os.path.exists(ms_path):
        print("Filtering with Microsoft building footprints...")
        ms_sum = count_points(data, ms_path)

    # If the Google building data exists, extract the sum of building areas within the vector
    google_sum = 0
    if os.path.exists(google_path):
        print("Filtering with Google Open Buildings...")
        google_sum = count_points(data, google_path)

    # If the GHSL building data exists, extract the sum of building areas within the vector
    ghsl_sum = 0
    if os.path.exists(ghsl_path) and in_vector:
        print("Filtering with Global Human Settlements Layer (GHSL)...")
        data.to_crs("ESRI:54009").to_file(in_vector)
        ghsl_sum = exact_extract(
            ghsl_path, in_vector, "sum", output="pandas", progress=True
        )["sum"].values

        # Compute the total sum of building areas from all sources and add it as a new column
        data["sum"] = ms_sum + google_sum + ghsl_sum

    # Reset the index again to ensure the data is clean and ready for further processing
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
    out_file = os.path.join(out_dir, f"{iso_code}_{shapename}.geojson")

    # Return the existing file if it already exists
    if os.path.exists(out_file):
        points = gpd.read_file(out_file)
        return points

    # Generate sample points based on the configuration and parameters
    points = data_utils.generate_samples(
        config,
        iso_code=iso_code,
        buffer_size=buffer_size,
        spacing=spacing,
        adm_level=adm_level,
        shapename=shapename,
    )

    # Create a buffer around each point and add a unique identifier
    points["points"] = points["geometry"]
    points["geometry"] = points.buffer(buffer_size, cap_style=3)
    points["UID"] = list(points.index)

    # Filter points by building presence and select relevant columns
    columns = ["UID", "geometry", "shapeName"]
    points = points[columns]

    # Create temporary file for exact_extract
    temp_file = os.path.join(out_dir, f"{iso_code}_{shapename}_temp.geojson")
    if not os.path.exists(temp_file):
        points.to_file(temp_file, driver="GeoJSON", index=False)

    # Filter tiles by uninhabited locations
    filtered = filter_uninhabited(iso_code, config, points, in_vector=temp_file)
    filtered = filtered[columns + ["sum"]]
    filtered = filtered[filtered["sum"] > 0]

    # Save the filtered points to a GeoPackage file
    print(f"Saving {out_file}...")
    filtered.to_file(out_file, driver="GeoJSON", index=False)

    return filtered


def get_shapenames(data_config: dict, iso_code: str, adm_level: str) -> list:
    """
    Retrieves unique shape names for geographic boundaries at a specified administrative level.

    Args:
        data_config (dict): Configuration dictionary containing data-related settings.
        iso_code (str): ISO country code for the geographic region of interest.
        adm_level (str): Administrative level (e.g., "admin1" or "admin2") for the boundaries.

    Returns:
        list: A list of unique shape names sorted by geographic area in ascending order.
    """

    # Retrieve geographic boundaries for the specified country and administrative level
    geoboundary = data_utils.get_geoboundaries(
        data_config, iso_code, adm_level=adm_level
    ).to_crs(
        "EPSG:3857"
    )  # Convert to metric CRS for accurate area calculations

    # Dissolve boundaries by 'shapeName' to group features with the same name
    geoboundary = geoboundary.dissolve(by="shapeName").reset_index()

    # Calculate the area of each shape and sort by area in ascending order
    geoboundary["area"] = geoboundary["geometry"].area
    geoboundary = geoboundary.sort_values(["area"], ascending=True)

    # Extract all shape names as an ordered list
    shapenames = geoboundary.shapeName.values

    # Return the list of unique shape names
    return shapenames


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
        sat_creds (dict): Credentials for accessing Maxar satellite image services.
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
    shapenames = get_shapenames(data_config, iso_code, adm_level)

    for index, shapename in enumerate(shapenames):
        print(f"\nProcessing {shapename} ({index+1}/{len(shapenames)})...")
        # Generate prediction tiles for the current shape
        tiles = generate_pred_tiles(
            data_config, iso_code, spacing, buffer_size, shapename, adm_level
        ).reset_index(drop=True)

        # Compute centroid of each tile's geometry and filter by sum threshold
        tiles["points"] = tiles["geometry"].centroid
        tiles = tiles[tiles["sum"] > sum_threshold].reset_index(drop=True)
        print(f"{shapename} {index+1}/{len(shapenames)} total tiles: {tiles.shape}")

        # Prepare data for satellite image download
        data = tiles.copy()
        data["geometry"] = data["points"]
        sat_dir = os.path.join(os.getcwd(), "output", iso_code, "images", shapename)

        # Download satellite images for the filtered tiles
        sat_download.download_sat_images(
            sat_creds, sat_config, data=data, out_dir=sat_dir
        )
