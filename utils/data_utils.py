import os
import uuid
import requests

import geojson
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from pyproj import Proj, Transformer
from scipy.sparse.csgraph import connected_components

import logging
pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)


def create_progress_bar(items: list) -> tqdm:
    """
    Create a progress bar for iterating over a list of items.

    Args:
        items (iterable): An iterable collection of items to iterate over.

    Returns:
        tqdm.std.tqdm: A tqdm progress bar object for the given items.
    """
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(items, total=len(items), bar_format=bar_format)
    return pbar


def makedir(out_dir: str) -> str:
    """
    Create a directory if it does not exist.

    Args:
        out_dir (str): The path of the directory to create.

    Returns:
        str: The absolute path of the created directory.
    """
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def get_iso_regions(config: dict, iso_code: str) -> tuple:
    """
    Retrieve the country, sub-region, and region names for a given ISO code.

    Args:
        config (dict): Configuration dictionary containing the URL for ISO codes.
        iso_code (str): The ISO 3166-1 alpha-3 code of the country.

    Returns:
        tuple: A tuple containing the country name (str), sub-region name (str), 
            and region name (str).
    """
    # Read the ISO codes CSV file from the provided URL
    codes = pd.read_csv(config["iso_codes_url"])
    # Query the DataFrame for the specified ISO code
    subcode = codes.query(f"`alpha-3` == '{iso_code}'")
    # Extract the country name
    country = subcode["name"].values[0]
    # Extract the sub-region name
    subregion = subcode["sub-region"].values[0]
    # Extract the region name
    region = subcode["region"].values[0]
    return country, subregion, region


def get_image_filepaths(
    config: dict, data: gpd.GeoDataFrame, in_dir: str = None, ext: str = ".tiff"
) -> list:    
    """
    Generate a list of file paths for images based on the given configuration and data.

    Args:
        config (dict): Configuration dictionary containing directory paths.
        data (gpd.GeoDataFrame): GeoDataFrame containing the image metadata, 
            including the following columns: 'UID', 'iso', and 'class' 
        in_dir (str, optional): Optional input directory to override the default path. 
            Defaults to None.
        ext (str, optional): File extension for the images. Defaults to ".tiff".

    Returns:
        list: A list of file paths for the images.
    """
    # List to store the file paths
    filepaths = []

    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        # Construct the filename using the 'UID' and the extension
        file = f"{row['UID']}{ext}"
        if not in_dir:
            filepath = os.path.join(
                os.getcwd(),
                config["rasters_dir"],
                config["maxar_dir"],
                config["project"],
                row["iso"],
                row["class"],
                file,
            )
        else:
            # If an input directory is specified, use it
            filepath = os.path.join(in_dir, file)
        # Add the file path to the list
        filepaths.append(filepath)
    return filepaths


def convert_crs(
    data: gpd.GeoDataFrame, 
    src_crs: str = "EPSG:4326", 
    target_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    Convert the coordinate reference system (CRS) of a GeoDataFrame from source CRS to target CRS.

    Args:
        data (gpd.GeoDataFrame): The input GeoDataFrame with geometries to be transformed.
        src_crs (str, optional): The source CRS of the input geometries. 
            Defaults to "EPSG:4326".
        target_crs (str, optional): The target CRS to transform the geometries to. 
            Defaults to "EPSG:3857".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with geometries transformed to the target CRS.
    """
    # Check if 'lat' and 'lon' columns are missing
    if ("lat" not in data.columns) or ("lon" not in data.columns):
        # Extract 'lon' and 'lat' from geometry if missing
        data["lon"], data["lat"] = data.geometry.x, data.geometry.y

    # Transform the 'lon' and 'lat' values using the transformer
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    data["lon"], data["lat"] = transformer.transform(
        data["lon"].values, data["lat"].values
    )
    # Create a new geometry column from the transformed 'lon' and 'lat' values
    geometry = gpd.GeoSeries.from_xy(data["lon"], data["lat"])

    # Create a new GeoDataFrame with the transformed geometry and target CRS
    data = pd.DataFrame(data.drop("geometry", axis=1))
    data = gpd.GeoDataFrame(data, geometry=geometry, crs=target_crs)
    return data


def concat_data(data: list, out_file: str = None, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Concatenate a list of GeoDataFrames into a single GeoDataFrame, remove duplicates,
    and optionally save it to a GeoJSON file.

    Args:
        data (list): A list of GeoDataFrames to concatenate.
        out_file (str, optional): Output file path to save the concatenated 
            GeoDataFrame as GeoJSON. Defaults to None.
        verbose (bool, optional): Whether to log verbose information. Defaults to False.

    Returns:
        gpd.GeoDataFrame: The concatenated and optionally saved GeoDataFrame.
    """
    data = pd.concat(data).reset_index(drop=True)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")

    if out_file:
        data.to_file(out_file, driver="GeoJSON")
        
    if verbose:
        logging.info(f"Generated {out_file}")
        logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    return data


def generate_uid(data: gpd.GeoDataFrame, category: str) -> gpd.GeoDataFrame:
    """
    Generate unique identifiers (UIDs) for each row in the input DataFrame.

    Args:
        data (gpd.GeoDataFrame): Input DataFrame containing columns:
            'source', 'iso', and any additional columns
        category (str): Category name to include in the UID.

    Returns:
        pd.DataFrame: DataFrame with added 'UID' column containing unique identifiers.
    """
    # Add 'index' column with zero-padded index values
    data["index"] = data.index.to_series().apply(lambda x: str(x).zfill(8))
    data["category"] = category
    
    # Concatenate 'source', 'iso', 'category', and 'index' columns to generate UIDs
    uids = data[["source", "iso", "category", "index"]].agg("-".join, axis=1)
    
    # Drop temporary columns 'index' and 'category' from the DataFrame
    data = data.drop(["index", "category"], axis=1)
    
    # Assign generated UIDs to a new column 'UID' in the DataFrame
    data["UID"] = uids.str.upper()
    
    return data


def prepare_data(
    config: dict, 
    data: gpd.GeoDataFrame, 
    iso_code: str, 
    category: str, 
    source: str, 
    columns: list, 
    out_file: str = None
) -> gpd.GeoDataFrame:
    """
    Prepare data by adding necessary columns, generating unique identifiers (UIDs),
    and optionally saving the processed DataFrame to a GeoJSON file.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
        data (gpd.GeoDataFrame): Input DataFrame containing the raw data.
        iso_code (str): ISO code of the country for which data is being prepared.
        category (str): Category name describing the type of data.
        source (str): Source identifier indicating the origin of the data.
        columns (list): List of columns that should be retained in the final processed DataFrame.
        out_file (str, optional): Output file path to save the processed DataFrame as GeoJSON. 
            Defaults to None.

    Returns:
        pd.DataFrame: Processed DataFrame with added columns, generated UIDs, and filtered columns.
    """
    # Add 'giga_id_school' column with unique identifiers if not already present
    if "giga_id_school" not in data.columns:
        data["giga_id_school"] = data.reset_index().index

    # Ensure all specified columns exist in the DataFrame, initializing with None if missing
    for column in columns:
        if column not in data.columns:
            data[column] = None

    # Fetch country, region, and subregion names based on ISO code
    country, region, subregion = get_iso_regions(config, iso_code)

    # Add metadata columns to the DataFrame
    data["source"] = source.upper()
    data["iso"] = iso_code
    data["country"] = country
    data["subregion"] = region
    data["region"] = subregion

    # Generate unique identifiers (UIDs) for each row in the DataFrame
    if len(data) > 0:
        data = generate_uid(data, category)
        
    # Select only the specified columns
    data = data[columns]
    # Remove duplicate rows based on selected columns
    data = data.drop_duplicates(columns)

    # Optionally save the processed DataFrame to a GeoJSON file if out_file is provided
    if out_file:
        data.to_file(out_file, driver="GeoJSON")
    return data


def get_geoboundaries(
    config: dict, 
    iso_code: str, 
    out_dir: str = None, 
    adm_level: str = "ADM0"
) -> gpd.GeoDataFrame:
    """
    Fetch geoboundaries for a specified ISO code and administrative level.
    
    Args:
        config (dict): Configuration dictionary containing necessary parameters.
        iso_code (str): ISO code of the country for which geoboundaries are requested.
        out_dir (str, optional): Output directory where the geoboundary file will be saved.
        adm_level (str, optional): Administrative level of the geoboundary. Default is "ADM0".
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the fetched geoboundaries.
    """
    # Set the default output directory if not provided
    if not out_dir:
        out_dir = os.path.join(
            os.getcwd(), config["vectors_dir"], config["project"], "geoboundaries"
        )
        
    # Ensure the output directory exists
    out_dir = makedir(out_dir)

    # Define the filename and full output file path
    filename = f"{iso_code}_{adm_level}_geoboundary.geojson"
    out_file = os.path.join(out_dir, filename)

    # Download geoboundary if it doesn't already exist
    if not os.path.exists(out_file):
        try:
            url = f"{config['gbhumanitarian_url']}{iso_code}/{adm_level}/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]
        except:
            url = f"{config['gbopen_url']}{iso_code}/ADM0/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]

        # Download and save the geoboundary as GeoJSON
        geoboundary = requests.get(download_path).json()
        with open(out_file, "w") as file:
            geojson.dump(geoboundary, file)

    # Read the GeoJSON file into a GeoDataFrame
    geoboundary = gpd.read_file(out_file).fillna("")

    # Clean up 'shapeName' column if present
    if "shapeName" in geoboundary.columns:
        geoboundary["shapeName"] = geoboundary["shapeName"].apply(
            lambda x: "".join(
                [char if char.isalnum() or char == "-" else " " for char in x]
            )
        )
    return geoboundary


def read_data(
    iso_code: str, 
    data_dir: str, 
    sources: list = []
) -> gpd.GeoDataFrame:
    """
    Read GeoJSON data files from a specified directory.

    Args:
        iso_code (str): ISO code of the country of interest.
        data_dir (str): Directory path containing GeoJSON data files.
        sources (list, optional): List of specific data sources to read. If provided, only files
                                  named after these sources will be read (default is an empty list).

    Returns:
        gpd.GeoDataFrame: Concatenated GeoDataFrame containing all read data, with CRS EPSG:4326.
    """
    # Ensure the data directory exists
    data_dir = makedir(data_dir)

    # Determine files to read based on sources and exclusions
    files = []
    if len(sources) > 0:
        for source in sources:
            file = os.path.join(data_dir, source, f"{iso_code}_{source}.geojson")
            files.append(file)
            
    data = []
    # Read each file and append to the data list
    for file in (pbar := create_progress_bar(files)):
        pbar.set_description(f"Reading {file.split('/')[-1]}")
        filename = os.path.join(data_dir, file)
        subdata = gpd.read_file(filename)
        data.append(subdata)

    # Concatenate all data into a single GeoDataFrame
    data = gpd.GeoDataFrame(pd.concat(data).copy(), crs="EPSG:4326")
    return data


def drop_duplicates(data: gpd.GeoDataFrame, priority: list) -> gpd.GeoDataFrame:
    """
    Drops duplicates from a GeoDataFrame based on a specified priority of sources.

    Args:
        data (gpd.GeoDataFrame): Input GeoDataFrame containing geometries.
        priority (list): List of source names in descending order of priority.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with duplicates dropped based on the specified priority.
    """
    
    data["temp_source"] = pd.Categorical(
        data["source"], categories=priority, ordered=True
    )
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    return data


def connect_components(data: gpd.GeoDataFrame, buffer_size: float) -> gpd.GeoDataFrame:
    """
    Connects components in a GeoDataFrame based on overlapping geometries within a buffer size.

    Args:
        data (gpd.GeoDataFrame): Input GeoDataFrame containing geometries to connect.
        buffer_size (float): Buffer size in the units of the GeoDataFrame's CRS.

    Returns:
        gpd.GeoDataFrame: Modified GeoDataFrame with a new 'group' column indicating connected components.
    """
    # Make a copy of the data to avoid modifying the original
    temp = data.copy()

    # Convert CRS if it's not already EPSG:3857 (Web Mercator)
    if data.crs != "EPSG:3857":
        temp = convert_crs(data, target_crs="EPSG:3857")

    # Create buffer around geometries
    geometry = temp["geometry"].buffer(buffer_size, cap_style=3)

    # Calculate overlap matrix
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)

    # Find connected components using scipy.sparse.csgraph.connected_components
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups
    
    return data


def generate_samples(
    config: dict, 
    iso_code: str, 
    buffer_size: float, 
    spacing: float, 
    adm_level: str = "ADM0", 
    shapename: str = None
) -> gpd.GeoDataFrame:
    """
    Generate sample points within the geographical boundaries of a specified ISO code.

    Args:
        config (dict): Configuration dictionary containing project settings.
        iso_code (str): ISO code of the country or region of interest.
        buffer_size (float): Buffer size for the geographical boundaries.
        spacing (float): Spacing between sample points.
        adm_level (str, optional): Administrative level for geographical boundaries. Default is "ADM0".
        shapename (str, optional): Name of the shape within boundaries to filter. Default is None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing generated sample points within the specified boundaries.
    """
    # Get geographical boundaries for the ISO code at the specified administrative level
    bounds = get_geoboundaries(config, iso_code, adm_level=adm_level)
    if shapename:
        bounds = bounds[bounds.shapeName == shapename]
    bounds = bounds.to_crs("EPSG:3857")  # Convert to EPSG:3857

    # Calculate bounds for generating XY coordinates
    xmin, ymin, xmax, ymax = bounds.total_bounds
    xcoords = [c for c in np.arange(xmin, xmax, spacing)]
    ycoords = [c for c in np.arange(ymin, ymax, spacing)]

    # Create all combinations of XY coordinates
    coordinate_pairs = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2)
    # Create a list of Shapely points
    geometries = gpd.points_from_xy(coordinate_pairs[:, 0], coordinate_pairs[:, 1])

    # Create a GeoDataFrame of points and perform spatial join with bounds
    points = gpd.GeoDataFrame(geometry=geometries, crs=bounds.crs).reset_index(drop=True)
    points = gpd.sjoin(points, bounds, predicate="within")
    points = points.drop(["index_right"], axis=1)

    return points