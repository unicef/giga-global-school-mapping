import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask

from utils import data_utils
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42


def filter_keywords(
    data: gpd.GeoDataFrame, exclude: list, column: str = "name"
) -> gpd.GeoDataFrame:
    """
    Filters data based on keyword exclusion.

    Args:
        data (GeoDataFrame): Input GeoDataFrame to filter.
        exclude (list of str): List of keywords to exclude.
        column (str, optional): Column in the DataFrame to apply filtering. Defaults to "name".

    Returns:
        GeoDataFrame: Filtered GeoDataFrame after applying keyword exclusion.
    """
    exclude = [f"\\b{x.upper()}\\b" for x in exclude]
    data = data[
        ~data[column].str.upper().str.contains(r"|".join(exclude), case=False, na=False)
    ]
    return data


def sample_points(
    iso_code: str, config: dict, buffer_size: float, spacing: float, name: str = "clean"
) -> gpd.GeoDataFrame:
    """
    Sample points for augmentation, filtering out those overlapping with positive class geometries.

    Args:
        iso_code (str): ISO code for the country to process.
        config (dict): Configuration dictionary containing paths and parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - pos_class (str): Positive class category.
            - rasters_dir (str): Directory where raster data is stored.
            - ghsl_built_c_file (str): File name for GHSL built-up raster data.
        buffer_size (float): Buffer size for geometry operations.
        spacing (float): Spacing for sampling points.
        name (str, optional): Name of the cleaned data file. Defaults to "clean".

    Returns:
        GeoDataFrame: Sampled GeoDataFrame of points for augmentation.
    """
    points = data_utils.generate_samples(config, iso_code, buffer_size, spacing)

    # Read positive data and perform buffer operation on geometries
    filename = f"{iso_code}_{name}.geojson"
    vector_dir = os.path.join(os.getcwd(), config["vectors_dir"], config["project"])
    pos_file = os.path.join(vector_dir, config["pos_class"], name, filename)
    pos_df = gpd.read_file(pos_file).to_crs("EPSG:3857")

    pos_df["geometry"] = pos_df["geometry"].buffer(buffer_size, cap_style=3)
    points["geometry"] = points["geometry"].buffer(buffer_size, cap_style=3)

    # Identify intersecting points and remove them
    points["index"] = points.index
    intersecting = pos_df.sjoin(points, how="inner")["index"]
    points = points[~points["index"].isin(intersecting)]
    points["geometry"] = points["geometry"].centroid

    # Sample points from the microsoft raster
    points = points.to_crs("EPSG:3857")
    coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]

    points = points.to_crs("ESRI:54009")
    ghsl_coord_list = [
        (x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)
    ]
    raster_dir = os.path.join(os.getcwd(), config["rasters_dir"])

    # Filter points with pixel value greater than 0 and convert back to EPSG:4326
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    if os.path.exists(ms_path):
        with rio.open(ms_path) as src:
            points["ms_val"] = [x[0] for x in src.sample(coord_list)]

    google_path = os.path.join(raster_dir, "google_buildings", f"{iso_code}_google.tif")
    if os.path.exists(google_path):
        with rio.open(google_path) as src:
            points["google_val"] = [x[0] for x in src.sample(coord_list)]

    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])
    with rio.open(ghsl_path) as src:
        col_val = "ghsl_val" if "ms_val" in points.columns else "pixel_val"
        points[col_val] = [x[0] for x in src.sample(ghsl_coord_list)]

    points["pixel_val"] = points[["ms_val", "google_val", "ghsl_val"]].max(axis=1)
    points = points[points["pixel_val"] > 0]
    points = points.to_crs("EPSG:4326")
    return points


def augment_negative_samples(
    iso_code: str, config: dict, name: str = "clean"
) -> gpd.GeoDataFrame:
    """
    Augments negative samples to balance with positive samples.

    Args:
        iso_code (str): ISO code for the country to process.
        config (dict): Configuration dictionary containing paths and parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - pos_class (str): Positive class category.
            - neg_class (str): Negative class category.
            - object_proximity (float): Proximity threshold for filtering POIs.
            - sample_spacing (float): Spacing for sampling points.
            - columns (list of str): List of columns to retain in the data.
        name (str, optional): Name of the cleaned data file. Defaults to "clean".

    Returns:
        GeoDataFrame: Augmented GeoDataFrame of negative samples.
    """
    logging.info(f"Augmenting negative samples for {iso_code}")

    # Read positive class file
    pos_file = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        config["pos_class"],
        name,
        f"{iso_code}_{name}.geojson",
    )
    positives = gpd.read_file(pos_file)

    # Read negative class file
    neg_file = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        config["neg_class"],
        name,
        f"{iso_code}_{name}.geojson",
    )
    negatives = gpd.read_file(neg_file)

    n_pos = len(positives)
    n_neg = len(negatives)

    # Check if negative samples need augmentation to balance with positive samples
    if n_pos * 2 > n_neg:
        points = sample_points(
            iso_code,
            config,
            buffer_size=config["object_proximity"] / 2,
            spacing=config["sample_spacing"],
            name=name,
        )

        # Prepare sampled points as negative class data
        points = data_utils.prepare_data(
            config=config,
            data=points,
            iso_code=iso_code,
            category=config["neg_class"],
            source="AUG",
            columns=config["columns"],
        )

        # Initialize 'clean' or 'validated' columns if present in negatives
        if "clean" in negatives.columns:
            points["clean"] = 0
        if "validated" in negatives.columns:
            points["validated"] = 0

        # Sample additional points to achieve balance
        logging.info(f"{iso_code} {points.shape} {n_pos*2 - n_neg}")
        points = points.sample((n_pos * 2 - n_neg), random_state=SEED)

        # Concatenate sampled points with existing negatives and save to file
        negatives = data_utils.concat_data([points, negatives], neg_file, verbose=False)

    return negatives


def filter_uninhabited_locations(
    iso_code: str,
    data: gpd.GeoDataFrame,
    config: dict,
    shape_name: str,
    sum_threshold: int = 0,
) -> gpd.GeoDataFrame:
    """
    Filters uninhabited locations based on building footprint raster data.

    Args:
        iso_code (str): ISO code for the country to process.
        data (GeoDataFrame): GeoDataFrame containing data to filter.
        config (dict): Configuration dictionary containing paths and parameters.
            - rasters_dir (str): Directory where raster data is stored.
            - ghsl_built_c_file (str): File name for GHSL built-up raster data.
            - filter_buffer_size (float): Buffer size for geometry operations.
        shape_name (str): Name of the shape being processed.
        sum_threshold (int, optional): Threshold sum of pixels to consider inhabited. Defaults to 0.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame with inhabited locations.
    """
    data = data.reset_index(drop=True)
    buffer_size = config["filter_buffer_size"]

    # Generate file paths based on the iso_code and source
    raster_dir = os.path.join(os.getcwd(), config["rasters_dir"])
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    google_path = os.path.join(raster_dir, "google_buildings", f"{iso_code}_google.tif")
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])

    pixel_sums = []
    for index in (pbar := data_utils.create_progress_bar(range(len(data)))):
        pbar.set_description(f"Processing {iso_code} {shape_name} {index}/{len(data)}")

        # Extract a single row from the DataFrame
        subdata = data.iloc[[index]]
        subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].buffer(buffer_size, cap_style=3)

        # Mask the raster data with the buffered geometry from Microsoft
        image = []
        pixel_sum = 0
        with rio.open(ms_path) as src:
            try:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image == 255] = 1
                pixel_sum = np.sum(image)
            except Exception as e:
                logging.info(e)
                pass

        # If no building pixels found, attempt with Google Open Buildings
        if pixel_sum == 0:
            with rio.open(google_path) as src:
                try:
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(src, geometry, crop=True)
                    image[image == 255] = 1
                    pixel_sum = np.sum(image)
                except Exception as e:
                    logging.info(e)
                    pass

        # If no building pixels found, attempt with GHSL data
        if pixel_sum == 0:
            with rio.open(ghsl_path) as src:
                subdata = subdata.to_crs("ESRI:54009")
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image == 255] = 0  # no pixel value
                pixel_sum = np.sum(image)

        # Appending the pixel sum to the list
        pixel_sums.append(pixel_sum)

    # Filter data based on pixel sums and updating DataFrame accordingly
    data["sum"] = pixel_sums
    data = data[data["sum"] > sum_threshold]
    data = data.reset_index(drop=True)

    return data


def filter_pois_within_object_proximity(
    iso_code: str, config: dict, proximity: float, sources: list, name: str = "clean"
) -> gpd.GeoDataFrame:
    """
    Filters points of interest (POIs) within a specified proximity of school locations.

    Args:
        iso_code (str): ISO code for the country to process.
        config (dict): Configuration dictionary containing paths and parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - pos_class (str): Positive class category.
            - neg_class (str): Negative class category.
            - columns (list of str): List of columns to retain in the data.
        proximity (float): Proximity threshold for filtering POIs.
        sources (list of str): List of data sources.
        name (str, optional): Name of the cleaned data file. Defaults to "clean".

    Returns:
        GeoDataFrame: Filtered GeoDataFrame of non-school POIs.
    """
    # Set the data directory and file paths
    data_dir = os.path.join(os.getcwd(), config["vectors_dir"], config["project"])
    filename = f"{iso_code}_{name}.geojson"

    # Read positive class data (e.g., school locations)
    pos_file = os.path.join(os.getcwd(), data_dir, config["pos_class"], name, filename)
    pos_sub = gpd.read_file(pos_file)

    # Read negative class data (e.g., non-school POIs)
    neg_dir = os.path.join(
        os.getcwd(), config["vectors_dir"], config["project"], config["neg_class"]
    )
    neg_sub = data_utils.read_data(iso_code, neg_dir, sources=sources)

    # Filter out validated and already cleaned positive samples
    if "clean" in pos_sub.columns:
        pos_sub = pos_sub[pos_sub["clean"] == 0]
    if "validated" in pos_sub.columns:
        pos_sub = pos_sub[pos_sub["validated"] == 0]

    # Convert school and non-school data CRS to EPSG:3857
    neg_temp = data_utils.convert_crs(neg_sub, target_crs="EPSG:3857")
    neg_temp["geometry"] = neg_temp["geometry"].buffer(proximity / 2, cap_style=3)
    neg_temp["index"] = neg_sub.index
    pos_temp = data_utils.convert_crs(pos_sub, target_crs="EPSG:3857")
    pos_temp["geometry"] = pos_temp["geometry"].buffer(proximity / 2, cap_style=3)

    # Filter out non-school POIs that intersect with buffered school locations
    intersecting = pos_temp.sjoin(neg_temp, how="inner")["index"]
    neg_sub = neg_sub[~neg_temp["index"].isin(intersecting)]

    # Save filtered country-level dataset
    subdata = neg_sub[config["columns"]]
    return subdata


def clean_data(
    iso_code: str,
    config: dict,
    category: str,
    name: str = "clean",
    id: str = "UID",
    sources: list = [],
) -> gpd.GeoDataFrame:
    """
    Cleans and processes data for a specified ISO code and category.

    Args:
        iso_code (str): ISO code for the country to process.
        config (dict): Configuration dictionary containing paths and parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - pos_class (str): Positive class category.
            - neg_class (str): Negative class category.
            - object_proximity (float): Proximity threshold for filtering POIs.
            - columns (list of str): List of columns to retain in the data.
            - exclude (list of str): List of keywords to exclude from data.
            - proximity (float): Proximity threshold for connecting components.
            - priority (str): Column name for priority in dropping duplicates.
        category (str): Data category.
        name (str, optional): Name of the cleaned data. Defaults to "clean".
        id (str, optional): Column name for unique identifier. Defaults to "UID".
        sources (list, optional): List of data sources. Defaults to an empty list.

    Returns:
        GeoDataFrame: Cleaned and processed GeoDataFrame.
    """

    def _get_condition(data, name, id, ids, shape_name):
        """
        Helper function to create a condition for data filtering.
        """
        return (
            (data[name] == 0)
            & (~data[id].isin(ids))
            & (data["shapeName"] == shape_name)
        )

    # Set default sources if none are provided
    if len(sources) == 0:
        if category == config["pos_class"]:
            sources = ["unicef", "osm", "overture"]
        if category == config["neg_class"]:
            sources = ["osm", "overture"]

    # Define output directory
    out_dir = os.path.join(config["vectors_dir"], config["project"], category, name)
    out_dir = data_utils.makedir(out_dir)

    # Read or filter data based on category
    if category == config["pos_class"]:
        data_dir = os.path.join(config["vectors_dir"], config["project"], category)
        data = data_utils.read_data(iso_code, data_dir, sources=sources)

    elif category == config["neg_class"]:
        data = filter_pois_within_object_proximity(
            iso_code, config, sources=sources, proximity=config["object_proximity"]
        )
    logging.info(f"Data dimensions: {data.shape}")

    # Output file path for cleaned data
    out_file = os.path.join(out_dir, f"{iso_code}_{name}.geojson")
    if not os.path.exists(out_file):
        data[name] = 0
        geoboundaries = data_utils.get_geoboundaries(config, iso_code, adm_level="ADM1")
        geoboundaries = geoboundaries[["shapeName", "geometry"]].dropna(
            subset=["shapeName"]
        )
        data = data.sjoin(geoboundaries, how="left", predicate="within")
        data.loc[data["geometry"].duplicated(keep="first"), "clean"] = 2

        # Split the data into smaller admin boundaries for scalability
        for shape_name in data.shapeName.unique():
            subdata = data[data["shapeName"] == shape_name]
            subdata = subdata[subdata.clean == 0]
            subdata = subdata[config["columns"]].reset_index(drop=True)
            if len(subdata) == 0:
                continue

            # Filter objects containing certain keywords
            if category == config["pos_class"]:
                subdata = filter_keywords(subdata, exclude=config["exclude"])[
                    config["columns"]
                ]
                ids = subdata[id].values
                condition = _get_condition(data, name, id, ids, shape_name)
                data.loc[condition, name] = 1

            # Remove objects within proximity of each other
            subdata = data_utils.connect_components(
                subdata, buffer_size=config["proximity"] / 2
            )
            subdata = data_utils.drop_duplicates(subdata, priority=config["priority"])[
                config["columns"]
            ]
            ids = subdata[id].values
            condition = _get_condition(data, name, id, ids, shape_name)
            data.loc[condition, name] = 2

            # Filter uninhabited locations based on specified buffer size
            subdata = filter_uninhabited_locations(
                iso_code, subdata, config, shape_name
            )[config["columns"]]
            ids = subdata[id].values
            condition = _get_condition(data, name, id, ids, shape_name)
            data.loc[condition, name] = 3

        # Save clean file as a GeoJSON
        data = data[config["columns"] + [name]].reset_index(drop=True)
        data = data_utils.concat_data([data], out_file=out_file)

        if category == config["neg_class"]:
            data = augment_negative_samples(iso_code, config, name=name)
            logging.info(f"Data dimensions: {data.shape}")

    # Read the cleaned data
    data = gpd.read_file(out_file).reset_index(drop=True)
    return data
