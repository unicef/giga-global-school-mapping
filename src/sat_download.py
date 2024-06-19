import os
import logging
import argparse

from tqdm import tqdm
import geopandas as gpd
from owslib.wms import WebMapService

from utils import data_utils
from utils import config_utils

SEED = 42
logging.basicConfig(level=logging.INFO)


def download_sat_images(
    creds: dict,
    config: dict,
    category: str = None,
    iso_code: str = None,
    sample_size: int = None,
    src_crs: str = "EPSG:4326",
    id_col: str = "UID",
    name: str = "clean",
    data: gpd.GeoDataFrame = None,
    filename: str = None,
    out_dir: str = None,
    download_validated: bool = False,
) -> None:
    """
    Download satellite images based on provided configurations and credentials.

    Args:
        creds (dict): Dictionary containing credentials for accessing the Web Map service.
            - connect_id (str): Connection ID for the Web Map service.
            - username (str): Username for the Web Map service.
            - password (str): Password for the Web Map service.
        config (dict): Configuration dictionary containing necessary parameters.
            - project (str): Name of the project.
            - vectors_dir (str): Directory where vector data is stored.
            - rasters_dir (str): Directory where raster data is stored.
            - maxar_dir (str): Directory where Maxar data is stored.
            - digitalglobe_url (str): URL for the DigitalGlobe service.
            - size (float): Size parameter for bounding box calculation.
            - layers (str): Layers to be fetched from the WMS service.
            - srs (str): Spatial reference system.
            - width (int): Width of the requested image.
            - height (int): Height of the requested image.
            - featureprofile (str): Feature profile for the WMS request.
            - coverage_cql_filter (str): CQL filter for coverage.
            - exceptions (str): Exceptions parameter for the WMS request.
            - transparent (bool): Transparency parameter for the WMS request.
            - format (str): Format of the requested image.
        category (str, optional): Category of the data. Defaults to None.
        iso_code (str, optional): ISO code of the country. Defaults to None.
        sample_size (int, optional): Number of samples to download. Defaults to None.
        src_crs (str, optional): Source coordinate reference system. Defaults to "EPSG:4326".
        id_col (str, optional): Column name for unique identifiers in the data. Defaults to "UID".
        name (str, optional): Name of the data file. Defaults to "clean".
        data (gpd.GeoDataFrame, optional): Geopandas GeoDataFrame containing the data. Defaults to None.
        filename (str, optional): Filename for reading the data. Defaults to None.
        out_dir (str, optional): Output directory for saving images. Defaults to None.
        download_validated (bool, optional): Whether to download validated images. Defaults to False.

    Returns:
        None
    """
    # Load data if not provided, load data
    if data is None:
        if not filename:
            filename = os.path.join(
                os.getcwd(),
                config["vectors_dir"],
                config["project"],
                category,
                name,
                f"{iso_code}_{name}.geojson",
            )
        data = gpd.read_file(filename).reset_index(drop=True)

    # Filter data based on 'clean' and 'validated' columns
    if "clean" in data.columns:
        data = data[data["clean"] == 0]
    if "validated" in data.columns and not download_validated:
        data = data[data["validated"] == 0]
    if "iso" in data.columns:
        data = data[data["iso"] == iso_code].reset_index(drop=True)
    if sample_size:
        data = data.iloc[:sample_size]

    # Convert data CRS
    data = data_utils.convert_crs(data, data.crs, config["srs"])
    # logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    # Determine output directory
    if not out_dir:
        out_dir = os.path.join(
            os.getcwd(),
            config["rasters_dir"],
            config["maxar_dir"],
            config["project"],
            iso_code,
            category,
        )
    out_dir = data_utils.makedir(out_dir)
    # logging.info(out_dir)

    # Check if all images already exist
    all_exists = True
    for index in range(len(data)):
        image_file = os.path.join(out_dir, f"{data[id_col][index]}.tiff")
        if not os.path.exists(image_file):
            all_exists = False
            break
    if all_exists:
        return

    # Initialize Web Map Service (WMS) connection
    url = f"{config['digitalglobe_url']}connectid={creds['connect_id']}"
    wms = WebMapService(url, username=creds["username"], password=creds["password"])

    # Download images with progress bar
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    for index in tqdm(range(len(data)), bar_format=bar_format):
        image_file = os.path.join(out_dir, f"{data[id_col][index]}.tiff")
        while not os.path.exists(image_file):
            try:
                # Define bounding box for the image request
                bbox = (
                    data.lon[index] - config["size"],
                    data.lat[index] - config["size"],
                    data.lon[index] + config["size"],
                    data.lat[index] + config["size"],
                )
                # Request image from WMS
                img = wms.getmap(
                    bbox=bbox,
                    layers=config["layers"],
                    srs=config["srs"],
                    size=(config["width"], config["height"]),
                    featureProfile=config["featureprofile"],
                    coverage_cql_filter=config["coverage_cql_filter"],
                    exceptions=config["exceptions"],
                    transparent=config["transparent"],
                    format=config["format"],
                )
                # Save the image to file
                with open(image_file, "wb") as file:
                    file.write(img.read())
            except Exception as e:
                logging.info(e)
                pass


def main():
    # Load arguments from parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--creds", help="Path to the credentials file")
    parser.add_argument("--category", help="Category (e.g. school or non_school)")
    parser.add_argument("--iso_code", help="ISO 3166-1 alpha-3 code")
    parser.add_argument("--filename", help="Filename of data (optional)", default=None)
    args = parser.parse_args()

    # Load config file
    config = config_utils.load_config(args.config)
    creds = config_utils.create_config(args.creds)

    # Download satellite images
    download_sat_images(
        creds,
        config,
        iso_code=args.iso_code,
        category=args.category,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
