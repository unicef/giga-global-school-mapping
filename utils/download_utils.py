import re
import os
import ast
import duckdb
import geojson
import overpass
import leafmap
import subprocess
import operator
import delta_sharing

import pandas as pd
import geopandas as gpd
from shapely import wkt
from rapidfuzz import fuzz

from country_bounding_boxes import country_subunits_by_iso_code
from utils import data_utils

import logging
import warnings
from typing import Union, List, Dict, Optional, Tuple, TYPE_CHECKING

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)


def _query_osm(iso_code: str, out_file: str, query: str) -> gpd.GeoDataFrame:
    """
    Execute an Overpass API query to download OpenStreetMap (OSM) data for a specific ISO code.

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


def download_osm(config: dict, category: str, source: str = "osm") -> gpd.GeoDataFrame:
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
        category (str): The category of OSM data to download (e.g., "school", "hospital").
        source (str): Source of the data. Defaults to "osm".

    Returns:
        GeoDataFrame: Combined GeoDataFrame containing the downloaded OSM data.
    """

    # Construct the output directory path
    out_dir = os.path.join(config["vectors_dir"], config["project"], category, source)
    out_dir = data_utils.makedir(out_dir)

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

    data = []  # List to store the downloaded data
    for iso_code in (pbar := data_utils.create_progress_bar(iso_codes)):
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
            subdata = data_utils.prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=config["columns"],
                out_file=out_subfile,
            )
            data.append(subdata)  # Add the processed data to the list

    # Combine all the downloaded and processed data into a single GeoDataFrame
    data = data_utils.concat_data(data, osm_file)
    return data


def _query_overture(
    config: dict, iso_code: str, out_file: str, query: str
) -> gpd.GeoDataFrame:
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


def download_overture(
    config: dict, category: str, exclude: str = None, source: str = "overture"
) -> gpd.GeoDataFrame:
    """
    Download geospatial data from the Overture dataset based on the provided configuration and category.
    For more info, see: https://til.simonwillison.net/overture-maps/overture-maps-parquet

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
    out_dir = data_utils.makedir(out_dir)

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

    data = []  # List to store the downloaded data
    for iso_code in (pbar := data_utils.create_progress_bar(iso_codes)):
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
                subdata["names"] = subdata["names"].astype(str)
                subdata["names"] = subdata["names"].apply(lambda x: ast.literal_eval(x))
                geoboundary = data_utils.get_geoboundaries(config, iso_code)
                subdata["name"] = subdata["names"].apply(
                    lambda x: x["common"][0]["value"]
                )
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")

                # Prepare and process the data
                subdata = data_utils.prepare_data(
                    config=config,
                    data=subdata,
                    iso_code=iso_code,
                    category=category,
                    source=source,
                    columns=config["columns"],
                    out_file=out_subfile,
                )
            subdata = gpd.read_file(out_subfile).reset_index(drop=True)
            data.append(subdata)  # Add the processed data to the list

    # Combine all the downloaded and processed data into a single GeoDataFrame
    data = data_utils.concat_data(data, overture_file)
    return data


def download_unicef(
    config: dict,
    profile_file: str,
    category: str = "school",
    source: str = "unicef",
    in_file: str = None,
) -> gpd.GeoDataFrame:
    """
    Downloads and processes UNICEF data for specified ISO codes.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
            - vectors_dir (str): Directory where vector data is stored.
            - project (str): Name of the project.
            - iso_codes (list of str): List of ISO codes to process.
            - columns (list of str): List of columns to retain in the data.
        profile_file (str): Path to the Delta Sharing profile file.
        category (str, optional): Data category. Defaults to "school".
        source (str, optional): Data source. Defaults to "unicef".

    Returns:
        GeoDataFrame: Combined GeoDataFrame of all processed data.
    """

    # Generate output directory
    data_dir = os.path.join(config["vectors_dir"], config["project"], category, source)
    data_dir = data_utils.makedir(data_dir)

    # List of ISO codes and columns to process
    iso_codes = config["iso_codes"]
    columns = config["columns"]

    data = []
    # Create a progress bar for ISO codes processing
    for iso_code in (pbar := data_utils.create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")

        filename = f"{iso_code}_{source}.geojson"
        out_subfile = os.path.join(data_dir, filename)

        # Check if the file already exists to avoid redundant processing
        if not os.path.exists(out_subfile):
            if in_file:
                subdata = pd.read_csv(os.path.join(os.getcwd(), in_file))
            else:
                # Initialize Delta Sharing client
                try:
                    delta_sharing.SharingClient(profile_file)
                    table_url = f"{profile_file}#gold.school-master.{iso_code}"
                    subdata = delta_sharing.load_as_pandas(table_url)
                except Exception as e:
                    logging.info(e)
                    continue

            # Convert latitude and longitude to geometry
            subdata["geometry"] = gpd.GeoSeries.from_xy(
                subdata["longitude"], subdata["latitude"]
            )
            subdata["name"] = subdata["school_name"]
            # Prepare the data with additional processing
            subdata = data_utils.prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=columns,
            )
            # Convert to GeoDataFrame with specified CRS
            subdata = gpd.GeoDataFrame(subdata, geometry="geometry", crs="EPSG:4326")
            # Save the processed data to file
            subdata.to_file(out_subfile)

        # Read the processed data from file and reset the index
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)

    # Combine all the processed data into a single GeoDataFrame and save to file
    out_dir = os.path.dirname(data_dir)
    out_file = os.path.join(out_dir, f"{source}.geojson")
    data = data_utils.concat_data(data, out_file)
    return data


def get_country(config: dict, source: str = "ms"):
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

    matches = dict()  # Dictionary to store the matches
    for iso_code in iso_codes:
        # Get the country name and other region info based on the ISO code
        country, _, _ = data_utils.get_iso_regions(config, iso_code)

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


def download_buildings(config: dict, source: str = "ms", verbose: bool = False) -> None:
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
    matches = get_country(config, source)

    for iso_code in (pbar := data_utils.create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")

        # Create output directories
        out_dir = data_utils.makedir(
            os.path.join(config["vectors_dir"], f"{source}_buildings")
        )
        temp_dir = data_utils.makedir(os.path.join(out_dir, iso_code))
        country = matches[iso_code]

        # Download building data if the output file doesn't already exist
        out_file = str(os.path.join(out_dir, f"{iso_code}_{source}_EPSG4326.geojson"))
        if not os.path.exists(out_file):
            quiet = operator.not_(verbose)
            try:
                if source == "ms":
                    download_ms_buildings(
                        country,
                        config["microsoft_url"],
                        out_dir=temp_dir,
                        merge_output=out_file,
                        quiet=quiet,
                        overwrite=True,
                    )
                elif source == "google":
                    download_google_buildings(
                        country,
                        config["google_url"],
                        out_dir=temp_dir,
                        merge_output=out_file,
                        quiet=quiet,
                        overwrite=True,
                    )
            except Exception as e:
                logging.info(e)
                continue

        # Define paths for reprojected GeoJSON file and rasterized TIFF file
        out_file_epsg3857 = str(
            os.path.join(out_dir, f"{iso_code}_{source}_EPSG3857.geojson")
        )
        tif_dir = data_utils.makedir(
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


def download_ghsl(config: dict, type: str = "built_c") -> None:
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
    ghsl_folder = data_utils.makedir(ghsl_folder)

    # Determine the GHSL file path based on the type
    if type == "built_c":
        ghsl_path = os.path.join(ghsl_folder, config["ghsl_built_c_file"])
    elif type == "smod":
        ghsl_path = os.path.join(ghsl_folder, config["ghsl_smod_file"])

    # Download and extract the GHSL data if it doesn't already exist
    if not os.path.exists(ghsl_path):
        if type == "built_c":
            ghsl_zip = os.path.join(ghsl_folder, "ghsl_built_c.zip")
            command1 = f"wget {config['ghsl_built_c_url']} -O {ghsl_zip}"
        elif type == "smod":
            ghsl_zip = os.path.join(ghsl_folder, "ghsl_smod.zip")
            command1 = f"wget {config['ghsl_smod_url']} -O {ghsl_zip}"

        # Unzip the downloaded file into the GHSL folder
        command2 = f"unzip {ghsl_zip} -d {ghsl_folder}"
        subprocess.Popen(f"{command1} && {command2}", shell=True)


def download_ms_buildings(
    location: str,
    building_url: str,
    out_dir: Optional[str] = None,
    merge_output: Optional[str] = None,
    head=None,
    quiet: bool = False,
    **kwargs,
) -> List[str]:
    """
    Download Microsoft Buildings dataset for a specific location. Check the dataset links from
        https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv.

    Args:
        location: The location name for which to download the dataset.
        building_url: Building URL for Microsoft Buildings.
        out_dir: The output directory to save the downloaded files.
            If not provided, the current working directory is used.
        merge_output: Optional. The output file path for merging the downloaded files into a single GeoDataFrame.
        head: Optional. The number of files to download. If not provided, all files will be downloaded.
        quiet: Optional. If True, suppresses the download progress messages.
        **kwargs: Additional keyword arguments to be passed to the `gpd.to_file` function.

    Returns:
        A list of file paths of the downloaded files.

    """

    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import shape

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_links = pd.read_csv(building_url)
    country_links = dataset_links[dataset_links.Location == location]

    if not quiet:
        print(f"Found {len(country_links)} links for {location}")
    if head is not None:
        country_links = country_links.head(head)

    filenames = []
    i = 1

    for _, row in country_links.iterrows():
        if not quiet:
            print(f"Downloading {i} of {len(country_links)}: {row.QuadKey}.geojson")
        i += 1
        filename = os.path.join(out_dir, f"{row.QuadKey}.geojson")
        filenames.append(filename)
        if os.path.exists(filename):
            print(f"File {filename} already exists, skipping...")
            continue
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        gdf.to_file(filename, driver="GeoJSON", **kwargs)

    if merge_output is not None:
        if os.path.exists(merge_output):
            print(f"File {merge_output} already exists, skip merging...")
            return filenames
        leafmap.merge_vector(filenames, merge_output, quiet=quiet)

    return filenames


def download_google_buildings(
    location: str,
    building_url: str,
    out_dir: Optional[str] = None,
    merge_output: Optional[str] = None,
    head: Optional[int] = None,
    keep_geojson: bool = False,
    overwrite: bool = False,
    quiet: bool = False,
    **kwargs,
) -> List[str]:
    """
    Download Google Open Building dataset for a specific location. Check the dataset links from
        https://sites.research.google/open-buildings.

    Args:
        location: The location name for which to download the dataset.
        building_url: Building URL for Google Open Buildings.
        out_dir: The output directory to save the downloaded files.
            If not provided, the current working directory is used.
        merge_output: Optional. The output file path for merging the downloaded files into a single GeoDataFrame.
        head: Optional. The number of files to download. If not provided, all files will be downloaded.
        keep_geojson: Optional. If True, the GeoJSON files will be kept after converting them to CSV files.
        overwrite: Optional. If True, overwrite the existing files.
        quiet: Optional. If True, suppresses the download progress messages.
        **kwargs: Additional keyword arguments to be passed to the `gpd.to_file` function.

    Returns:
        A list of file paths of the downloaded files.

    """
    country_url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    building_gdf = gpd.read_file(building_url)
    country_gdf = gpd.read_file(country_url)

    country = country_gdf[country_gdf["NAME"] == location]

    if len(country) == 0:
        country = country_gdf[country_gdf["NAME_LONG"] == location]
        if len(country) == 0:
            raise ValueError(f"Could not find {location} in the Natural Earth dataset.")

    gdf = building_gdf[building_gdf.intersects(country.geometry.iloc[0])]
    gdf.sort_values(by="size_mb", inplace=True)

    print(f"Found {len(gdf)} links for {location}.")
    if head is not None:
        gdf = gdf.head(head)

    if len(gdf) > 0:
        links = gdf["tile_url"].tolist()
        leafmap.download_files(links, out_dir=out_dir, quiet=quiet, **kwargs)
        filenames = [os.path.join(out_dir, os.path.basename(link)) for link in links]

        gdfs = []
        for filename in filenames:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filename)

            # Create a geometry column from the "geometry" column in the DataFrame
            df["geometry"] = df["geometry"].apply(wkt.loads)

            # Convert the pandas DataFrame to a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry="geometry")
            gdf.crs = "EPSG:4326"
            if keep_geojson:
                gdf.to_file(
                    filename.replace(".csv.gz", ".geojson"), driver="GeoJSON", **kwargs
                )
            gdfs.append(gdf)

        if merge_output:
            if os.path.exists(merge_output) and not overwrite:
                print(f"File {merge_output} already exists, skip merging...")
            else:
                if not quiet:
                    print("Merging GeoDataFrames ...")
                gdf = gpd.GeoDataFrame(
                    pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
                )
                gdf.to_file(merge_output, **kwargs)

    else:
        print(f"No buildings found for {location}.")
