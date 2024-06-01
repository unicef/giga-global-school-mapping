import re
import os
import time
import duckdb
import geojson
import overpass
import data_utils
import logging
import leafmap
import pyproj
import subprocess
import operator

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz
from country_bounding_boxes import country_subunits_by_iso_code

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()


def _query_osm(iso_code, out_file, query):
    """
    Execute an Overpass API query to download OpenStreetMap (OSM) data for a specific ISO code area.

    Args:
        iso_code (str): ISO 3166-1 alpha-2 code representing the country or region to query.
        out_file (str): Path to the file where the downloaded OSM data will be saved.
        query (str): Overpass API query string specifying the OSM data to download.

    Returns:
        GeoDataFrame: GeoDataFrame containing the downloaded OSM data.
    """

    # Initialize the Overpass API with a specified timeout
    api = overpass.API(timeout=1500)

    # Construct the Overpass query
    osm_query = f"""
        area["ISO3166-1"="{iso_code}"][admin_level=2];
        ({query});
        out center;
    """

    # Execute the query and get the data
    data = api.get(osm_query, verbosity="geom")

    # Save the data to a GeoJSON file
    with open(out_file, "w") as file:
        geojson.dump(data, file)

    # Load the data into a GeoDataFrame
    data = gpd.read_file(out_file)
    return data


def download_osm(config, category, source="osm"):
    """
    Download OpenStreetMap (OSM) data based on the provided configuration and category.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - iso_codes (list): List of ISO codes to process.
            - iso_codes_url (str): URL to fetch ISO code data.
            - columns (list): List of columns to include in the output data.
            - {category} (dict): 
                Dictionary of keywords for the OSM query, where the keys are OSM tags 
                and the values are lists of tag values to match. The {category} key should 
                be replaced with the actual category name.
        category (str): The category of OSM data to download (e.g., "buildings", "roads").
        source (str): Source of the data. Defaults to "osm".

    Returns:
        GeoDataFrame: Combined GeoDataFrame containing the downloaded OSM data.
    """

    # Construct the output directory path
    out_dir = os.path.join(config["vectors_dir"], config["project"], category, source)
    out_dir = data_utils._makedir(out_dir)

    # Define the path for the combined OSM file
    osm_file = os.path.join(os.path.dirname(out_dir), f"{source}.geojson")

    # List of ISO codes to process
    iso_codes = config["iso_codes"]

    # URL to fetch ISO code data
    url = config["iso_codes_url"]
    codes = pd.read_csv(url)

    # Construct OSM query from the provided category keywords
    keywords = config[category]
    query = "".join(
        [
            f"""node["{key}"~"^({"|".join(values)})"](area);
        way["{key}"~"^({"|".join(values)})"](area);
        rel["{key}"~"^({"|".join(values)})"](area);
        """
            for key, values in keywords.items()
            if key != "translated"
        ]
    )

    data = [] # List to store the downloaded data
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_{source}.geojson"
        out_subfile = os.path.join(out_dir, filename)

        # Check if the file already exists; if not, download it
        if not os.path.exists(out_subfile):
            alpha_2 = codes.query(f"`alpha-3` == '{iso_code}'")["alpha-2"].values[0]
            _query_osm(alpha_2, out_subfile, query)

        # Read the downloaded file into a GeoDataFrame
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            # Prepare and process the data
            subdata = data_utils._prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=config["columns"],
                out_file=out_subfile,
            )
            data.append(subdata) # Add the processed data to the list

    # Combine all the downloaded and processed data into a single GeoDataFrame
    data = data_utils._concat_data(data, osm_file)
    return data


def _query_overture(config, iso_code, out_file, query):
    """
    Execute a query on the Overture dataset to download geospatial data for a specific ISO code area.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - overture_url (str): URL to access the Overture dataset.
        iso_code (str): ISO 3166-1 alpha-2 code representing the country or region to query.
        out_file (str): Path to the file where the downloaded geospatial data will be saved.
        query (str): SQL query string specifying the data to download from the Overture dataset.

    Returns:
        GeoDataFrame: GeoDataFrame containing the downloaded geospatial data.
    """
    
    # Connect to DuckDB and install necessary extensions
    db = duckdb.connect()
    db.execute("INSTALL spatial")
    db.execute("INSTALL httpfs")
    db.execute(
        """
        LOAD spatial;
        LOAD httpfs;
        SET s3_region='us-west-2';
    """
    )

    # Fetch country bounding box
    url = config["overture_url"]
    bbox = [c.bbox for c in country_subunits_by_iso_code(iso_code)][0]

    # Construct the Overture query
    overture_query = f"""
        COPY(
            select
            JSON(names) AS names,
            ST_GeomFromWkb(geometry) AS geometry
        from
            read_parquet('{url}')
        where
            bbox.minx > {bbox[0]}
            and bbox.miny > {bbox[1]}
            and bbox.maxx < {bbox[2]}
            and bbox.maxy < {bbox[3]}
            and (
            {query}
            )
        ) TO '{out_file}'
        WITH (FORMAT GDAL, DRIVER 'GeoJSON')
    """

    # Execute the query and fetch all results
    db.execute(overture_query).fetchall()

    # Load the data into a GeoDataFrame
    data = gpd.read_file(out_file)
    return data


def download_overture(config, category, exclude=None, source="overture"):
    """
    Download geospatial data from the Overture dataset based on the provided configuration and category.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - iso_codes (list): List of ISO codes to process.
            - columns (list): List of columns to include in the output data.
            - {category} (dict): 
                Dictionary of keywords for querying Overture data, where the keys are 
                categories and the values are lists of keywords.
            - {exclude} (dict): 
                Dictionary of keywords to exclude from the query, where the keys are 
                categories and the values are lists of keywords.
            - overture_url (str): URL to access the Overture dataset.
        category (str): The category of data to download (e.g., "school", "hospital").
        exclude (str, optional): Category of keywords to exclude from the query. Defaults to None.
        source (str): Source of the data. Defaults to "overture".

    Returns:
        GeoDataFrame: Combined GeoDataFrame containing the downloaded Overture data.
    """
    
    # Generate output directory
    out_dir = os.path.join(config["vectors_dir"], config["project"], category, source)
    out_dir = data_utils._makedir(out_dir)

    # Define the path for the combined Overture file
    overture_file = os.path.join(os.path.dirname(out_dir), f"{source}.geojson")

    # List of ISO codes to process
    iso_codes = config["iso_codes"]

    # Fetch keywords for the category and generate the query
    keywords = config[category]
    query = " or ".join(
        [
            f"""UPPER(names) LIKE '%{keyword.replace('_', ' ').upper()}%'
        """
            for key, values in keywords.items()
            for keyword in values
        ]
    )

    # If exclude keywords are provided, add them to the query
    if exclude:
        exclude_keywords = config[exclude]
        exclude_query = " and ".join(
            [
                f"""UPPER(names) NOT LIKE '%{keyword.replace('_', ' ').upper()}%'
            """
                for key, values in exclude_keywords.items()
                for keyword in values
            ]
        )
        query = f""" ({query}) and ({exclude_query})"""

    data = [] # List to store the downloaded data
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_{source}.geojson"
        out_subfile = os.path.join(out_dir, filename)

        # Fetch Overture data if the file doesn't already exist
        if not os.path.exists(out_subfile):
            _query_overture(config, iso_code, out_subfile, query)

        # Read the downloaded file into a GeoDataFrame
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            # Extract name from Overture data
            if "names" in subdata.columns:
                geoboundary = data_utils._get_geoboundaries(config, iso_code)
                subdata["name"] = subdata["names"].apply(
                    lambda x: x["common"][0]["value"]
                )
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")

            # Prepare and process the data
            subdata = data_utils._prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=config["columns"],
                out_file=out_subfile,
            )
            data.append(subdata) # Add the processed data to the list

    # Combine all the downloaded and processed data into a single GeoDataFrame
    data = data_utils._concat_data(data, overture_file)
    return data


def load_unicef(config, category="school", source="unicef"):
    """
    Load and process UNICEF data based on the provided configuration and category.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - columns (list): List of columns to include in the output data.
        category (str): The category of data to load (e.g., "school", "hospital"). 
            Defaults to "school".
        source (str): Source of the data. Defaults to "unicef".

    Returns:
        GeoDataFrame: Combined GeoDataFrame containing the processed UNICEF data.
    """
    
    # Generate the data directory
    data_dir = os.path.join(config["vectors_dir"], config["project"], category, source)
    data_dir = data_utils._makedir(data_dir) # Create the directory if it doesn't exist

    # List all files in the data directory
    files = next(os.walk(data_dir), (None, None, []))[2]
    logging.info(f"Number of CSV files: {len(files)}")

    data = [] # List to store the processed data
    for file in (pbar := data_utils._create_progress_bar(files)):
        iso_code = file.split("_")[0]
        pbar.set_description(f"Processing {iso_code}")
        filename = os.path.join(data_dir, file)

        # Load the CSV file into a DataFrame
        subdata = pd.read_csv(filename).reset_index(drop=True)

        # Create geometry column from longitude and latitude
        subdata["geometry"] = gpd.GeoSeries.from_xy(subdata["lon"], subdata["lat"])
        columns = config["columns"]

        # Prepare and process the data
        subdata = data_utils._prepare_data(
            config=config,
            data=subdata,
            iso_code=iso_code,
            category=category,
            source=source,
            columns=columns,
        )
        data.append(subdata) # Add the processed data to the list

    # Combine all the processed data into a single GeoDataFrame and save to file
    out_dir = os.path.dirname(data_dir)
    out_file = os.path.join(out_dir, f"{source}.geojson")
    data = data_utils._concat_data(data, out_file)
    return data


def _get_country(config, source="ms"):
    """
    Match country names from ISO codes with Microsoft dataset or 
    return country names based on configuration.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - iso_codes (list): List of ISO codes to process.
            - microsoft_url (str): URL to access the Microsoft dataset.
        source (str): Source of the data. Defaults to "ms".

    Returns:
        dict: A dictionary mapping ISO codes to country names or Microsoft country names.
    """
    
    # List of ISO codes to process
    iso_codes = config["iso_codes"]

    # Load Microsoft dataset links
    msf_links = pd.read_csv(config["microsoft_url"])
    
    matches = dict() # Dictionary to store the matches
    for iso_code in iso_codes:
        # Get the country name and other region info based on the ISO code
        country, _, _ = data_utils._get_iso_regions(config, iso_code)
        
        if source == "ms":
            max_score = 0
            for msf_country in msf_links.Location.unique():
                # Adjust the Microsoft country name for better matching
                msf_country_ = re.sub(r"(\w)([A-Z])", r"\1 \2", msf_country)

                # Calculate the similarity score between the country names
                score = fuzz.partial_token_sort_ratio(country, msf_country_)
                if score > max_score:
                    max_score = score
                    matches[iso_code] = msf_country
        else:
            matches[iso_code] = country
            
    return matches


def download_buildings(config, source="ms", verbose=False):
    """
    Download building footprints for the specified ISO codes from either
    Microsoft or Google sources.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - iso_codes (list): List of ISO codes to process.
            - vectors_dir (str): Directory where vector data is stored.
            - rasters_dir (str): Directory where raster data is stored.
        source (str): Source of the building data. 
            Can be either "ms" (Microsoft) or "google". Defaults to "ms".
        verbose (bool): Flag to indicate verbosity of the download process. Defaults to False.

    Returns:
        None
    """

    # List of ISO codes to process
    iso_codes = config["iso_codes"]

    # Get country matches based on the source
    matches = _get_country(config, source)

    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")

        # Create output directories
        out_dir = data_utils._makedir(
            os.path.join(config["vectors_dir"], f"{source}_buildings")
        )
        temp_dir = data_utils._makedir(os.path.join(out_dir, iso_code))
        country = matches[iso_code]

        # Download building data if the output file doesn't already exist
        out_file = str(os.path.join(out_dir, f"{iso_code}_{source}_EPSG4326.geojson"))
        if not os.path.exists(out_file):
            quiet = operator.not_(verbose)
            try:
                if source == "ms":
                    leafmap.download_ms_buildings(
                        country,
                        out_dir=temp_dir,
                        merge_output=out_file,
                        quiet=quiet,
                        overwrite=True,
                    )
                elif source == "google":
                    leafmap.download_google_buildings(
                        country,
                        out_dir=temp_dir,
                        merge_output=out_file,
                        quiet=quiet,
                        overwrite=True,
                    )
            except:
                continue

        # Define paths for reprojected GeoJSON file and rasterized TIFF file
        out_file_epsg3857 = str(
            os.path.join(out_dir, f"{iso_code}_{source}_EPSG3857.geojson")
        )
        tif_dir = data_utils._makedir(
            os.path.join(config["rasters_dir"], f"{source}_buildings")
        )
        tif_file = str(os.path.join(tif_dir, f"{iso_code}_{source}.tif"))

        # Reproject and rasterize the data if necessary files don't exist
        if (
            (os.path.exists(out_file))
            and (not os.path.exists(out_file_epsg3857))
            and (not os.path.exists(tif_file))
        ):
            command1 = "ogr2ogr -s_srs EPSG:4326 -t_srs EPSG:3857 {} {}".format(
                out_file_epsg3857, out_file
            )
            command2 = (
                "gdal_rasterize -burn 255 -tr 10 10 -a_nodata 0 -at -l {} {} {}".format(
                    f"{iso_code}_{source}_EPSG4326", out_file_epsg3857, tif_file
                )
            )
            subprocess.Popen(f"{command1} && {command2}", shell=True)


def download_ghsl(config, type="built_c"):
    """
    Download and extract the Global Human Settlement Layer (GHSL) data.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
            - rasters_dir (str): Directory where raster data is stored.
            - ghsl_built_c_file (str): Filename for the built-up coverage GHSL data.
            - ghsl_smod_file (str): Filename for the settlement model GHSL data.
            - ghsl_built_c_url (str): URL to download the built-up coverage GHSL data.
            - ghsl_smod_url (str): URL to download the settlement model GHSL data.
        type (str): Type of GHSL data to download. Can be either 
            - "built_c" (built-up coverage) or 
            - "smod" (settlement model). 
            Defaults to "built_c".

    Returns:
        None
    """

    # Create the GHSL folder in the rasters directory
    ghsl_folder = os.path.join(config["rasters_dir"], "ghsl")
    ghsl_folder = data_utils._makedir(ghsl_folder)

    # Determine the GHSL file path based on the type
    if type == "built_c":
        ghsl_path = os.path.join(ghsl_folder, config["ghsl_built_c_file"])
    elif type == "smod":
        ghsl_path = os.path.join(ghsl_folder, config["ghsl_smod_file"])

    # Path to the downloaded zip file
    ghsl_zip = os.path.join(ghsl_folder, "ghsl.zip")

    # Download and extract the GHSL data if it doesn't already exist
    if not os.path.exists(ghsl_path):
        if type == "built_c":
            command1 = f"wget {config['ghsl_built_c_url']} -O {ghsl_zip}"
        elif type == "smod":
            command1 = f"wget {config['ghsl_smod_url']} -O {ghsl_zip}"

        # Unzip the downloaded file into the GHSL folder
        command2 = f"unzip {ghsl_zip} -d {ghsl_folder}"
        subprocess.Popen(f"{command1} && {command2}", shell=True)
