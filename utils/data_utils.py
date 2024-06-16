import re
import os
import uuid
import requests
import logging

import geojson
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from pyproj import Proj, Transformer
from scipy.sparse.csgraph import connected_components

pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)


def clean_text(text):
    if text:
        return re.sub(r"[^\w\s]", "", text).upper()
    return text


def create_progress_bar(items):
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(items, total=len(items), bar_format=bar_format)
    return pbar


def makedir(out_dir):
    cwd = os.path.dirname(os.getcwd())
    out_dir = os.path.join(cwd, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def get_iso_regions(config, iso_code):
    # Load ISO codes of countries and regions/subregions
    codes = pd.read_csv(config["iso_codes_url"])
    subcode = codes.query(f"`alpha-3` == '{iso_code}'")
    country = subcode["name"].values[0]
    subregion = subcode["sub-region"].values[0]
    region = subcode["region"].values[0]

    return country, subregion, region


def get_image_filepaths(config, data, in_dir=None, ext=".tiff"):    
    filepaths = []
    cwd = os.path.dirname(os.getcwd())
    for index, row in data.iterrows():
        file = f"{row['UID']}{ext}"
        if not in_dir:
            filepath = os.path.join(
                cwd,
                config["rasters_dir"],
                config["maxar_dir"],
                config["project"],
                row["iso"],
                row["class"],
                file,
            )
        else:
            filepath = os.path.join(in_dir, file)
        filepaths.append(filepath)
    return filepaths


def convert_crs(data, src_crs="EPSG:4326", target_crs="EPSG:3857"):
    # Convert lat long to the target CRS
    if ("lat" not in data.columns) or ("lon" not in data.columns):
        data["lon"], data["lat"] = data.geometry.x, data.geometry.y
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    data["lon"], data["lat"] = transformer.transform(
        data["lon"].values, data["lat"].values
    )
    geometry = gpd.GeoSeries.from_xy(data["lon"], data["lat"])

    # Convert the data to GeoDataFrame
    data = pd.DataFrame(data.drop("geometry", axis=1))
    data = gpd.GeoDataFrame(data, geometry=geometry, crs=target_crs)
    return data


def concat_data(data, out_file=None, verbose=False):
    data = pd.concat(data).reset_index(drop=True)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    data = data.drop_duplicates()

    if out_file:
        data.to_file(out_file, driver="GeoJSON")
    if verbose:
        logging.info(f"Generated {out_file}")
        logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    return data


def generate_uid(data, category):
    data["index"] = data.index.to_series().apply(lambda x: str(x).zfill(8))
    data["category"] = category
    uids = data[["source", "iso", "category", "index"]].agg("-".join, axis=1)
    data = data.drop(["index", "category"], axis=1)
    data["UID"] = uids.str.upper()
    return data


def prepare_data(config, data, iso_code, category, source, columns, out_file=None):
    if "giga_id_school" not in data.columns:
        data["giga_id_school"] = data.reset_index().index

    for column in columns:
        if column not in data.columns:
            data[column] = None

    country, region, subregion = _get_iso_regions(config, iso_code)
    data["source"] = source.upper()
    data["iso"] = iso_code
    data["country"] = country
    data["subregion"] = region
    data["region"] = subregion

    if len(data) > 0:
        data = _generate_uid(data, category)
    data = data[columns]
    data = data.drop_duplicates(columns)

    if out_file:
        data.to_file(out_file, driver="GeoJSON")
    return data


def get_geoboundaries(config, iso_code, out_dir=None, adm_level="ADM0"):
    # Query geoBoundaries
    if not out_dir:
        cwd = os.path.dirname(os.getcwd())
        out_dir = os.path.join(
            cwd, config["vectors_dir"], config["project"], "geoboundaries"
        )
    if not os.path.exists(out_dir):
        out_dir = makedir(out_dir)

    # Save the result as a GeoJSON
    filename = f"{iso_code}_{adm_level}_geoboundary.geojson"
    out_file = os.path.join(out_dir, filename)

    if not os.path.exists(out_file):
        try:
            url = f"{config['gbhumanitarian_url']}{iso_code}/{adm_level}/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]
        except:
            url = f"{config['gbopen_url']}{iso_code}/ADM0/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]
        geoboundary = requests.get(download_path).json()
        with open(out_file, "w") as file:
            geojson.dump(geoboundary, file)

    # Read data using GeoPandas
    geoboundary = gpd.read_file(out_file).fillna("")
    if "shapeName" in geoboundary.columns:
        geoboundary["shapeName"] = geoboundary["shapeName"].apply(
            lambda x: "".join(
                [char if char.isalnum() or char == "-" else " " for char in x]
            )
        )
    return geoboundary


def read_data(data_dir, sources=[], exclude=[]):
    data_dir = makedir(data_dir)
    if len(sources) > 0:
        files = [f"{source}.geojson" for source in sources]
    else:
        files = next(os.walk(data_dir), (None, None, []))[2]
    files = [file for file in files if file not in exclude]
    logging.info(files)

    data = []
    for file in (pbar := create_progress_bar(files)):
        pbar.set_description(f"Reading {file}")
        filename = os.path.join(data_dir, file)
        subdata = gpd.read_file(filename)
        data.append(subdata)

    # Concatenate files in data_dir
    data = gpd.GeoDataFrame(pd.concat(data).copy(), crs="EPSG:4326")
    data = data.drop_duplicates()
    return data


def connect_components(data, buffer_size):
    temp = data.copy()
    if data.crs != "EPSG:3857":
        temp = _convert_crs(data, target_crs="EPSG:3857")
    geometry = temp["geometry"].buffer(buffer_size, cap_style=3)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups
    return data


def drop_duplicates(data, priority):
    data["temp_source"] = pd.Categorical(
        data["source"], categories=priority, ordered=True
    )
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    return data


def generate_samples(
    config, iso_code, buffer_size, spacing, adm_level="ADM0", shapename=None
):
    # Get geographical boundaries for the ISO code at the specified administrative level
    bounds = _get_geoboundaries(config, iso_code, adm_level=adm_level)
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
    points = gpd.GeoDataFrame(geometry=geometries, crs=bounds.crs).reset_index(
        drop=True
    )
    points = gpd.sjoin(points, bounds, predicate="within")
    points = points.drop(["index_right"], axis=1)

    return points


def get_counts(config, column="iso", layer="clean"):
    cwd = os.path.dirname(os.getcwd())
    categories = [config["pos_class"], config["neg_class"]]
    data = {category: [] for category in categories}

    iso_codes = config["iso_codes"]
    for iso_code in (pbar := create_progress_bar(iso_codes)):
        pbar.set_description(f"Reading counts for {iso_code}")
        for category in categories:
            dir = os.path.join(cwd, config["vectors_dir"], config["project"], category)
            filepath = os.path.join(dir, layer, f"{iso_code}_{layer}.geojson")
            subdata = gpd.read_file(filepath)
            if "clean" in subdata.columns:
                subdata = subdata[subdata["clean"] == 0]
            if "validated" in subdata.columns:
                subdata = subdata[subdata["validated"] == 0]
            data[category].append(subdata)

    for key, values in data.items():
        data[key] = pd.concat(values)

    counts = pd.merge(
        data[categories[0]][column].value_counts(),
        data[categories[1]][column].value_counts(),
        left_index=True,
        right_index=True,
    )
    counts.columns = categories

    return counts
