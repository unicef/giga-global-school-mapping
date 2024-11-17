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
from utils import cam_utils
from utils import calib_utils
from utils import config_utils
from utils import pred_utils
from utils import model_utils
from utils import eval_utils
from utils import plot_utils

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import matplotlib.patches as mpatches
import matplotlib as mpl

from scipy.spatial import cKDTree
from typing import Tuple, List, Union

logging.basicConfig(level=logging.INFO)


def read_file(
    iso_code: str, config: dict, cam_method: str = "gradcam", source: str = "preds"
) -> gpd.GeoDataFrame:
    """
    Reads a GeoJSON file containing either master or prediction data.

    Args:
        iso_code (str): ISO code representing the geographical region.
        config (dict): Configuration dictionary containing project and configuration names.
        cam_method (str, optional): Method used for CAM (Class Activation Mapping). Defaults to "gradcam".
        source (str, optional): Data source to read from, either "master" or "preds". Defaults to "preds".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the data read from the file.
    """
    # Construct the base directory path for output files
    out_dir = os.path.join(
        os.getcwd(), "output", iso_code, "results", config["project"]
    )

    # Determine the file path based on the source
    if source == "preds":
        # Update output directory for CAM results
        out_dir = os.path.join(out_dir, "cams")
        # Construct filename with CAM method and config name
        filename = f"{iso_code}_{config[iso_code][0].split('/')[-1].split('.')[0]}_{cam_method}.geojson"
        out_file = os.path.join(out_dir, filename)
    else:
        out_file = os.path.join(out_dir, f"{iso_code}_{source}.geojson")

    # Read and return the GeoDataFrame from the specified file
    data = gpd.read_file(out_file)
    return data


def standardize_data(
    config: dict, 
    iso_code: str, 
    cam_method: str = None, 
    source: str = "preds", 
    uid: str = "PUID"
) -> gpd.GeoDataFrame:
    """
    Standardizes data for a given country and model configuration. This includes renaming columns, 
    generating unique IDs, adding administrative fields, and transforming to a standard coordinate system.

    Args:
        config (dict): Configuration dictionary for the model, including:
            - model (str): Name of the model used.
            - project (str): Project directory for storing outputs.
            - config_name (str): Configuration name used in naming files.
        iso_code (str): The ISO code for the country.
        cam_method (str, optional): The CAM method used in data generation. Default is "gradcam".
        source (str, optional): Specifies the data source type, either "preds" or "master". Default is "preds".
        uid (str, optional): Unique identifier for schools in the data. Default is "PUID".

    Returns:
        gpd.GeoDataFrame: Standardized geospatial data ready for further analysis.
    """
    # Get best CAM method
    model_config = config_utils.load_config(config)
    cam_method = cam_utils.get_best_cam_method(iso_code, model_config)

    # Read the input data file based on iso_code, config, CAM method, and source
    data = read_file(iso_code, config, cam_method=cam_method, source=source)

    # If the source is "preds", adjust column names and formats for predictions
    if source == "preds":
        data = data.rename(columns={"prob": "predicted_proba"})
        data[uid] = data[uid].astype(str).str.pad(width=8, side="right", fillchar="0")
        data[uid] = data[uid].apply(lambda x: f"ML-{iso_code}-SCHOOL-{x}")
        data["source"] = "ML"
        data["name"] = "Unknown"

    if source != "master":
        # For non-master sources, create a unique 'giga_id_school' identifier
        data["giga_id_school"] = data[uid]
    elif source == "master":
        # For master sources, set source type and handle one-hot encoding for "clean" column
        data["source"] = "MASTER"
        one_hot = pd.get_dummies(data["clean"])
        data = data.drop("clean", axis=1)
        data = data.join(one_hot)
        data = data.rename(columns={1: "uninhabited"})
        data = data.rename(columns={2: "is_duplicated"})

    # Assign school_id based on UID, used for joining with other datasets
    data["school_id"] = data[uid]

    # Add administrative fields based on configuration and country code
    data = add_admin_fields(config, iso_code, data)

    # Get rural-urban classification based on the model configuration
    data = model_utils.get_rurban_classification(config, data)

    # Transform the data to the standard coordinate system (EPSG:4326)
    data = data.to_crs("EPSG:4326")
    data["lon"] = data["geometry"].x
    data["lat"] = data["geometry"].y

    # Drop records with missing values in admin fields if source is "preds"
    if source == "preds":
        data = data.dropna(subset=["admin1", "admin2"])

    # Ensure 'giga_id_school' and 'school_id' columns are treated as strings
    data["giga_id_school"] = data["giga_id_school"].astype(str)
    data["school_id"] = data["school_id"].astype(str)

    # Add any missing columns from standard columns to the data as null values
    for column in config["standard_columns"]:
        if column not in data.columns:
            data[column] = None
    data = data[config["standard_columns"]]

    return data.reset_index(drop=True)


def add_admin_fields(
    config: dict, 
    iso_code: str, 
    data: gpd.GeoDataFrame, 
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Adds administrative fields (admin1 and admin2) to the data by spatially joining 
    it with administrative boundaries.

    Args:
        config (dict): Configuration dictionary containing paths and other parameters, including:
            - vectors_dir: Directory path for vector files, such as administrative boundaries.
        iso_code (str): ISO code for the country.
        data (gpd.GeoDataFrame): Input geospatial data to which admin fields are added.
        crs (str, optional): Coordinate reference system for data. Default is "EPSG:4326".

    Returns:
        gpd.GeoDataFrame: Data with added administrative fields, cleaned of duplicates.
    """

    # Convert data to specified CRS for alignment with admin layers
    data = data.to_crs(crs)

    # Load admin1 (first-level administrative boundaries) GeoJSON file
    admin1 = gpd.read_file(
        os.path.join(
            os.getcwd(),
            config["vectors_dir"],
            "mapbox",
            f"{iso_code}_admin1.geojson",
        )
    )

    # Load admin2 (second-level administrative boundaries) GeoJSON file
    admin2 = gpd.read_file(
        os.path.join(
            os.getcwd(),
            config["vectors_dir"],
            "mapbox",
            f"{iso_code}_admin2.geojson",
        )
    )

    # Rename 'name' column in admin1 to 'admin1' to represent first-level names
    admin1.rename(columns={"name": "admin1"}, inplace=True)

    # Rename 'name' column in admin2 to 'admin2' to represent second-level names
    admin2.rename(columns={"name": "admin2"}, inplace=True)

    # Extract admin1 ID from admin2 entries by slicing the first six characters
    admin2["admin1_id_giga"] = admin2["admin2_id_giga"].apply(lambda x: x[:6])

    # Map admin1 names onto admin2 entries using admin1 IDs
    admin2["admin1"] = admin2["admin1_id_giga"].map(
        dict(zip(admin1["admin1_id_giga"], admin1["admin1"]))
    )

    # Spatially join admin2 data to input data using 'intersects' predicate for matching
    data_joined = data.sjoin(
        admin2[["admin1", "admin1_id_giga", "admin2", "admin2_id_giga", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Remove 'index_right' column from the joined data (artifact of spatial join)
    data = data_joined.drop(columns=["index_right"])
    data = data.drop_duplicates(subset=["giga_id_school"], keep="first")

    return data.reset_index(drop=True)


def get_lat_lon_pairs(data: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extracts latitude and longitude pairs from a GeoDataFrame and returns them as a NumPy array.

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing geometry with latitude and longitude coordinates.

    Returns:
        np.ndarray: A NumPy array of shape (n, 2) where n is the number of rows in the input GeoDataFrame.
                       Each row contains the latitude and longitude of a point.
    """
    # Initialize an empty NumPy array to hold latitude and longitude pairs
    lat_lon_pairs = np.empty((len(data), 2))

    # Iterate through each row of the GeoDataFrame and extract latitude and longitude
    for i, row in enumerate(data.itertuples()):
        lat_lon_pairs[i] = [row.geometry.y, row.geometry.x]

    return lat_lon_pairs


def match_dataframes(
    left_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    right_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    distance_threshold: float,
    coordinate_columns: List[str] = ["lat", "lon"],
) -> List[Tuple[int, int, float]]:
    """
    Match rows between two dataframes based on spatial distance using KD-tree.

    Args:
        left_df: Left dataframe to match from
        right_df: Right dataframe to match to
        distance_threshold: Maximum allowed distance for matching, nonnegative float
        coordinate_columns: List of column names to use for distance calculation (when dataframe is provided)

    Returns:
        List of tuples containing (left_index, right_index, distance)
    """

    def get_coordinates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> np.ndarray:
        """Extract coordinates as numpy array."""
        if isinstance(df, gpd.GeoDataFrame):
            return df.get_coordinates().values

        return df[coordinate_columns].values

    def build_distance_list(
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_indices: np.ndarray,
        right_indices: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """
        Build initial list of all valid pairs within distance threshold.
        """
        kdtree = cKDTree(right_coords)
        distances_list = []

        # Find all neighbors within distance threshold for each left point
        for i, left_point in enumerate(left_coords):
            distances, indices = kdtree.query(
                left_point, k=len(right_coords), distance_upper_bound=distance_threshold
            )
            # Only keep valid matches
            valid_mask = indices < len(right_coords)
            if np.any(valid_mask):
                for idx, dist in zip(indices[valid_mask], distances[valid_mask]):
                    distances_list.append(
                        (
                            left_indices[i],  # Use original left index
                            right_indices[idx],  # Use original right index
                            dist,
                        )
                    )

        # Sort by distance for efficient access to minimum
        return sorted(distances_list, key=lambda x: x[2])

    def match_pairs(
        distances_list: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """
        Find best matches from the distance list.
        """
        matched_pairs = []
        used_left = set()
        used_right = set()

        # Iterate through sorted distances and find valid pairs
        for left_idx, right_idx, dist in distances_list:
            if left_idx not in used_left and right_idx not in used_right:
                matched_pairs.append((left_idx, right_idx, dist))
                used_left.add(left_idx)
                used_right.add(right_idx)

        return matched_pairs

    # Initialize coordinates and indices
    left_coords = get_coordinates(left_df)
    right_coords = get_coordinates(right_df)
    left_indices = np.array(left_df.index)
    right_indices = np.array(right_df.index)

    # Build initial distance list
    initial_distances = build_distance_list(
        left_coords, right_coords, left_indices, right_indices
    )

    # Find matches
    out = pd.DataFrame(match_pairs(initial_distances))
    out.columns = ["left_index", "right_index", "distance"]
    return out


def calculate_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, R: float = 6371e3
) -> float:
    """
    Calculates the Haversine distance between two points on the Earth's surface using their latitude and longitude.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
        R (float, optional): Radius of the Earth in meters. Default is 6371e3 meters (Earth's radius).

    Returns:
        float: The great-circle distance between the two points in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate the distance
    distance = R * c

    return distance


def calculate_nearest_distance(
    refs: gpd.GeoDataFrame, source: gpd.GeoDataFrame, source_name: str, source_uid: str
) -> gpd.GeoDataFrame:
    """
    Calculates the nearest distance from each reference point to the nearest point
        in the source dataset and updates the reference points with additional information.

    Args:
        refs (gpd.GeoDataFrame): GeoDataFrame containing reference points with geometry
            in latitude and longitude.
        source (gpd.GeoDataFrame): GeoDataFrame containing source points with geometry
            in latitude and longitude.
        source_name (str): Name of the source dataset, used for naming the distance column.
        source_uid (str): Column name in the source GeoDataFrame that contains unique identifiers.

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with distances to the nearest source
            points and additional source information.
    """
    # Create a KDTree for fast nearest neighbor search using latitude and longitude pairs
    tree = cKDTree(get_lat_lon_pairs(source))

    # Convert reference and source GeoDataFrames to the WGS84 coordinate system (EPSG:4326)
    refs = refs.to_crs("EPSG:4326")
    source = source.to_crs("EPSG:4326")

    # Iterate over each reference point to find the nearest source point
    for index1, ref in refs.iterrows():
        # Query the KDTree to find the index of the nearest source point
        _, index2 = tree.query([ref.geometry.y, ref.geometry.x], 1)
        nearest = source.iloc[index2]

        # Calculate the distance between the reference point and the nearest source point
        distance = calculate_distance(
            ref.geometry.y, ref.geometry.x, nearest.geometry.y, nearest.geometry.x
        )

        # Update the reference GeoDataFrame with the calculated distance and additional information
        refs.loc[index1, f"distance_to_nearest_{source_name}"] = distance
        refs.loc[index1, source_uid] = nearest[source_uid]
        for prob_col in ["prob", "prob_cal"]:
            if prob_col in source.columns:
                refs.loc[index1, prob_col] = nearest[prob_col]

    return refs


def calculate_nearest_distances(preds, master, osm_overture):
    """
    Calculates the nearest distances between prediction points and two reference datasets
        (master and OSM Overture), and updates each dataset with the nearest distances
        and relevant information.

    Args:
        preds (gpd.GeoDataFrame): GeoDataFrame containing prediction points with
            geometry in latitude and longitude.
        master (gpd.GeoDataFrame): GeoDataFrame containing master reference points
            with geometry in latitude and longitude.
        osm_overture (gpd.GeoDataFrame): GeoDataFrame containing OSM Overture reference points
            with geometry in latitude and longitude.

    Returns:
        tuple: A tuple containing three updated GeoDataFrames:
            - preds: Updated prediction GeoDataFrame with nearest distances to master and OSM Overture points.
            - master: Updated master GeoDataFrame with nearest distances to prediction points.
            - osm_overture: Updated OSM Overture GeoDataFrame with nearest distances to prediction points.
    """
    # Calculate nearest distances from prediction points to master reference points
    preds = calculate_nearest_distance(
        preds, master[master["clean"] == 0], source_name="master", source_uid="MUID"
    )
    # Calculate nearest distances from prediction points to OSM Overture reference points
    preds = calculate_nearest_distance(
        preds,
        osm_overture[osm_overture["clean"] == 0],
        source_name="osm_overture",
        source_uid="SUID",
    )
    # Calculate nearest distances from master points to prediction points
    master = calculate_nearest_distance(
        master, preds, source_name="pred", source_uid="PUID"
    )
    # Calculate nearest distances from OSM Overture points to prediction points
    osm_overture = calculate_nearest_distance(
        osm_overture, preds, source_name="pred", source_uid="PUID"
    )
    return preds, master, osm_overture


def join_with_geoboundary(
    iso_code: str,
    data: gpd.GeoDataFrame,
    config: dict,
    adm_levels: list = ["ADM1", "ADM2", "ADM3"],
) -> gpd.GeoDataFrame:
    """
    Joins a GeoDataFrame with administrative boundary information based on specified administrative levels.

    Args:
        iso_code (str): ISO code for the region, used to retrieve geoboundary data.
        data (gpd.GeoDataFrame): GeoDataFrame containing the main dataset with geometry.
        config (dict): Configuration dictionary for retrieving geoboundary data.
        adm_levels (list of str, optional): List of administrative levels to include in the join.
            Default is ["ADM1", "ADM2", "ADM3"].

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with additional columns for each administrative level.
    """
    # Initialize the list of columns to include in the final GeoDataFrame
    columns = list(data.columns)
    for adm_level in adm_levels:
        # Add the current administrative level to the list of columns
        columns = columns + [adm_level]
        # Retrieve geoboundary data for the current administrative level
        admin = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)
        # Reproject the geoboundary data to match the CRS of the main dataset
        admin = admin.to_crs(data.crs)
        # Rename the shapeName column to match the administrative level
        admin = admin.rename(columns={"shapeName": adm_level})
        # Perform a spatial join with the main dataset to include administrative boundary information
        data = gpd.sjoin(data, admin[["geometry", adm_level]], how="left")
        # Keep only the columns that were in the original dataset plus the new administrative level
        data = data[columns]

    return data


def load_master(
    iso_code: str, config: dict, source: str = "master", colname: str = "clean"
) -> gpd.GeoDataFrame:
    """
    Loads and processes a GeoJSON file containing master source data,
        updates the column values based on specific rules,
        and joins the data with administrative boundary information.

    Args:
        iso_code (str): ISO code for the region, used to construct the file path for loading data.
        config (dict): Configuration dictionary for specifying directories and project names.
        source (str, optional): The type of source data to load. Default is "master".
        colname (str, optional): The column name to be updated based on specific rules. Default is "clean".

    Returns:
        gpd.GeoDataFrame: Processed GeoDataFrame with updated column values and joined administrative boundaries.
    """
    # Construct the file path for the GeoJSON file based on the provided parameters
    filename = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        "sources",
        f"{iso_code}_{source}.geojson",
    )

    # Load the GeoJSON file into a GeoDataFrame
    data = gpd.read_file(filename)

    # Update the column values
    data.loc[(data[colname] == 1), colname] = 0
    data.loc[(data[colname] == 3), colname] = 1
    data.loc[data.duplicated("geometry") & (data[colname] == 1), colname] = 1
    data.loc[data.duplicated("geometry") & (data[colname] != 1), colname] = 2

    # Join the data with administrative boundary information
    data = join_with_geoboundary(iso_code, data, config)

    # Rename the "UID" column to "MUID"
    data = data.rename({"UID": "MUID"}, axis=1)

    # Log the value counts of the specified column and the dimensions of the data
    logging.info(data[colname].value_counts())
    logging.info(f"Data dimensions: {data.shape}")

    return data


def load_osm_overture(
    iso_code: str, config: dict, source: str = "osm_overture", colname: str = "clean"
) -> gpd.GeoDataFrame:
    """
    Loads and processes a GeoJSON file containing OSM and Overture source data,
        removes duplicate entries, filters data based on specified column values,
        joins the data with administrative boundary information, and renames the
        unique identifier column.

    Args:
        iso_code (str): ISO code for the region, used to construct the file path for loading data.
        config (dict): Configuration dictionary for specifying directories and project names.
        source (str, optional): The type of source data to load. Default is "osm_overture".
        colname (str, optional): The column name to be used for filtering the data. Default is "clean".

    Returns:
        gpd.GeoDataFrame: Processed GeoDataFrame with removed duplicates, filtered data,
            joined administrative boundaries, and renamed columns.
    """
    # Construct the file path for the GeoJSON file based on the provided parameters
    filename = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        "sources",
        f"{iso_code}_{source}.geojson",
    )

    # Load the GeoJSON file into a GeoDataFrame, drop duplicates based on the "UID" column
    data = gpd.read_file(filename).drop_duplicates("UID").reset_index(drop=True)

    # Filter the data based on the specified column value and join with administrative boundary
    data = join_with_geoboundary(iso_code, data[data[colname] == 0], config)

    # Rename the "UID" column to "SUID"
    data = data.rename({"UID": "SUID"}, axis=1)

    # Log the value counts of the specified column and the dimensions of the data
    logging.info(data[colname].value_counts())
    logging.info(f"Data dimensions: {data.shape}")

    return data


def load_preds(
    iso_code: str,
    data_config: dict,
    sum_threshold: float = 0,
    buffer_size: float = 25,
    calibrator=None,
    source: str = "pred",
    colname: str = "clean",
) -> gpd.GeoDataFrame:
    """
    Loads, processes, and calibrates prediction data from files, joins with administrative boundary information,
    and optionally applies a calibration model.

    Args:
        iso_code (str): ISO code for the region, used to construct file paths and retrieve configurations.
        data_config (dict): Configuration dictionary for data processing.
        model_config (dict): Configuration dictionary for the model, including project and model details.
        cam_method (str, optional): The method used for class activation mapping. Default is "gradcam".
        sum_threshold (float, optional): The threshold for filtering data based on the "sum" column. Default is 0.
        buffer_size (float, optional): Buffer size for connecting components in the data. Default is 0.
        calibrator (object, optional): Calibration model to transform probabilities. Default is None.
        source (str, optional): The source type for loading predictions. Default is "pred".
        colname (str, optional): The column name used for filtering data. Default is "clean".

    Returns:
        gpd.GeoDataFrame: Processed GeoDataFrame with predictions, optional calibration applied,
            and joined administrative boundaries.
    """
    # Get the ensemble model configurations
    model_configs = model_utils.get_ensemble_configs(iso_code)
    best_models = model_utils.get_best_models(iso_code)
    best_model_config = config_utils.load_config(best_models[0])
    cam_method = cam_utils.get_best_cam_method(iso_code, best_model_config)

    # Construct the output directory path for the CAM results
    out_dir = os.path.join(
        os.getcwd(),
        "output",
        iso_code,
        "results",
        model_config["project"],
        "cams",
        "ensemble",
        model_configs[0]["config_name"],
        cam_method,
    )
    # print(out_dir)

    # Retrieve the list of filenames from the output directory
    filenames = next(os.walk(out_dir), (None, None, []))[2]
    filenames = [
        filename 
        for filename in filenames 
        if filename.split(".")[-1] in [".gpkg", "geojson"]
    ]

    # Initialize a list to store data from each file
    data = []

    # Read and process each file
    for filename in (pbar := data_utils.create_progress_bar(filenames)):
        pbar.set_description(f"Reading files for {iso_code}...")
        subdata = gpd.read_file(os.path.join(out_dir, filename))
        if len(subdata) > 0:
            subdata = subdata[subdata["sum"] > sum_threshold]
            data.append(subdata)

    # Concatenate data from all files, remove duplicates, and reset index
    data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry", crs="EPSG:3857")
    data = data.drop_duplicates("geometry").reset_index(drop=True)

    # Connect components and sort data by probability, removing duplicates
    data = data_utils.connect_components(data, buffer_size=buffer_size)
    data = data.sort_values("prob", ascending=False).drop_duplicates(["group"])

    # Join with administrative boundary information
    data = join_with_geoboundary(iso_code, data, data_config)

    # Update geometry to centroid and create a unique ID for each prediction
    data["geometry"] = data["geometry"].centroid
    data["PUID"] = data["ADM2"] + "_" + data["UID"].astype(str)
    data = data[~data.duplicated("PUID")]

    # Apply calibration model if provided
    if calibrator:
        model = calib_utils.load_calibrator(iso_code, model_config, calibrator)
        data["prob_cal"] = model.transform(data["prob"].values)

    # Drop duplicates and reset index
    data = data.drop_duplicates("geometry").reset_index(drop=True)

    # Log the dimensions of the final data
    logging.info(f"Data dimensions: {data.shape}")

    return data


def save_results(
    iso_code: str,
    data: gpd.GeoDataFrame,
    source: str,
    config_file: str = 'configs/config.yaml'
):
    """
    Saves the provided data to a GeoJSON file in the specified output directory.

    Args:
        iso_code (str): ISO code for the region, used to construct file paths.
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be saved.
        config (dict): Configuration dictionary, used to retrieve project names.
        source (str): The source type for determining the output file name and directory.
        cam_method (str, optional): The method used for class activation mapping, affecting the file name.
            Default is None.
    """
    # Construct the base output directory path based on the configuration and ISO code
    config = config_utils.load_config(os.path.join(os.getcwd(), config_file))
    out_dir = os.path.join(
        os.getcwd(),
        "output",
        iso_code,
        "results",
        config["project"],
    )
    best_models = model_utils.get_best_models(iso_code)
    best_config = config_utils.load_config(best_models[0])
    cam_method = cam_utils.get_best_cam_method(iso_code, best_config)

    # Determine the output file name based on the source type and CAM method
    out_file = f"{iso_code}_{source}.geojson"
    if source == "preds":
        out_dir = os.path.join(out_dir, "cams")
        model_config = model_utils.get_ensemble_configs(iso_code)[0]
        out_file = f"{iso_code}_{model_config['config_name']}_{cam_method}.geojson"

    # Construct the full path for the output file
    out_file = os.path.join(out_dir, out_file)
    data.to_file(out_file, driver="GeoJSON")
