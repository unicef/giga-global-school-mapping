import os
import pandas as pd
import geopandas as gpd
import rasterio as rio

import collections
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import clf_utils
import eval_utils
import data_utils

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42


def _get_scalers(scalers):
    scalers_list = [None]

    for scaler in scalers:
        scalers_list.append(clf_utils.get_scaler(scaler))

    return scalers_list


def _get_pipeline(model, selector):
    if model in clf_utils.MODELS:
        model = clf_utils.get_model(model)

    if selector in clf_utils.SELECTORS:
        selector = clf_utils.get_selector(selector)

    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )


def _get_params(scalers, model_params, selector_params):
    def _get_range(param):
        if param[0] == "np.linspace":
            return list(np.linspace(*param[1:]).astype(int))
        elif param[0] == "range":
            return list(range(*param[1:]))
        return param

    scalers = {"scaler": _get_scalers(scalers)}

    if model_params:
        model_params = {
            "model__" + name: _get_range(param) for name, param in model_params.items()
        }
    else:
        model_params = {}

    if selector_params:
        selector_params = {
            "selector__" + name: _get_range(param)
            for name, param in selector_params.items()
        }
    else:
        selector_params = {}

    params = [model_params, selector_params, scalers]

    return dict(collections.ChainMap(*params))


def get_cv(c):
    pipe = _get_pipeline(c["model"], c["selector"])
    params = _get_params(c["scalers"], c["model_params"], c["selector_params"])
    cv, cv_params = c["cv"], c["cv_params"]

    assert cv in [
        "RandomizedSearchCV",
        "GridSearchCV",
    ]

    scoring = eval_utils.get_scoring(c["pos_class"])
    if cv == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipe, params, scoring=scoring, random_state=SEED, **cv_params
        )
    elif cv == "GridSearchCV":
        return GridSearchCV(pipe, params, scoring=scoring, **cv_params)


def model_trainer(c, data, features, target):
    logging.info("Features: {}, Target: {}".format(features, target))

    X = data[features]
    y = data[target].values

    cv = get_cv(c)
    logging.info(cv)
    cv.fit(X, y)

    logging.info("Best estimator: {}".format(cv.best_estimator_))
    return cv


def load_data(
    config, attributes=["rurban"], in_dir="clean", out_dir="train", verbose=True
):
    cwd = os.path.dirname(os.getcwd())
    vector_dir = os.path.join(cwd, config["vectors_dir"], config["project"])
    iso_codes = config["iso_codes"]
    name = iso_codes[0]
    if "name" in config:
        name = config["name"] if config["name"] else name
    test_size = config["test_size"]

    filename = f"{name}_{out_dir}.geojson"
    out_file = os.path.join(cwd, vector_dir, out_dir, filename)
    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        if verbose:
            logging.info(f"Reading file {out_file}")
            _print_stats(data, attributes, test_size)
        return data

    data = []
    data_utils._makedir(os.path.dirname(out_file))
    for iso_code in iso_codes:
        in_file = f"{iso_code}_{in_dir}.geojson"
        pos_file = os.path.join(vector_dir, config["pos_class"], in_dir, in_file)
        neg_file = os.path.join(vector_dir, config["neg_class"], in_dir, in_file)

        pos = gpd.read_file(pos_file)
        pos["class"] = config["pos_class"]
        if "validated" in pos.columns:
            pos = pos[pos["validated"] == 0]

        neg = gpd.read_file(neg_file)
        neg["class"] = config["neg_class"]
        if "validated" in neg.columns:
            neg = neg[neg["validated"] == 0]
        data.append(pd.concat([pos, neg]))

    data = gpd.GeoDataFrame(pd.concat(data))
    data = data[(data["clean"] == 0)]
    data = _get_rurban_classification(config, data)
    data = _train_test_split(
        data, test_size=test_size, attributes=attributes, verbose=verbose
    )
    data.to_file(out_file, driver="GeoJSON")
    if verbose:
        _print_stats(data, attributes, test_size)

    return data


def _print_stats(data, attributes, test_size):
    total_size = len(data)
    test_size = int((total_size * test_size))
    attributes = attributes + ["class"]
    value_counts = data.groupby(attributes)[attributes[-1]].value_counts()
    value_counts = pd.DataFrame(value_counts).reset_index()
    value_counts["percentage"] = value_counts["count"] / total_size
    logging.info(f"\n{value_counts}")

    subcounts = pd.DataFrame(
        data.groupby(attributes + ["dataset"]).size().reset_index()
    )
    subcounts.columns = attributes + ["dataset", "count"]
    subcounts["percentage"] = (
        subcounts[subcounts.dataset != "train"]["count"] / test_size
    )
    subcounts = subcounts.set_index(attributes + ["dataset"])
    logging.info(f"\n{subcounts.to_string()}")

    if len(data.iso.unique()) > 1:
        subcounts = pd.DataFrame(
            data.groupby(["iso", "dataset", "class"]).size().reset_index()
        )
        subcounts.columns = ["iso", "dataset", "class", "count"]
        subcounts = subcounts.set_index(["iso", "dataset", "class"])
        logging.info(f"\n{subcounts.to_string()}")

        subcounts = pd.DataFrame(data.groupby(["iso", "dataset"]).size().reset_index())
        subcounts.columns = ["iso", "dataset", "count"]
        subcounts = subcounts.set_index(["iso", "dataset"])
        logging.info(f"\n{subcounts.to_string()}")

    subcounts = pd.DataFrame(data.groupby(["dataset", "class"]).size().reset_index())
    subcounts.columns = ["dataset", "class", "count"]
    subcounts["percentage"] = subcounts["count"] / total_size
    subcounts = subcounts.set_index(["dataset", "class"])

    logging.info(f"\n{subcounts.to_string()}")
    logging.info(f"\n{data.dataset.value_counts()}")
    logging.info(f"\n{data.dataset.value_counts(normalize=True)}")
    for attribute in attributes:
        if attribute != "iso":
            logging.info(f"\n{data[attribute].value_counts()}")
            logging.info(f"\n{data[attribute].value_counts(normalize=True)}")


def _get_rurban_classification(config, data):
    data = data.to_crs("ESRI:54009")
    coord_list = [(x, y) for x, y in zip(data["geometry"].x, data["geometry"].y)]

    cwd = os.path.dirname(os.getcwd())
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_smod_file"])
    with rio.open(ghsl_path) as src:
        data["ghsl_smod"] = [x[0] for x in src.sample(coord_list)]

    rural = [10, 11, 12, 13]
    data["rurban"] = "urban"
    data.loc[data["ghsl_smod"].isin(rural), "rurban"] = "rural"

    return data


def _train_test_split(data, test_size=0.2, attributes=["rurban"], verbose=True):
    if "dataset" in data.columns:
        return data

    data["dataset"] = None
    total_size = len(data)
    test_size = int((total_size * test_size))
    logging.info(f"Data dimensions: {total_size}")

    test = data.copy()
    value_counts = data.groupby(attributes)[attributes[-1]].value_counts()
    value_counts = pd.DataFrame(value_counts).reset_index()

    for _, row in value_counts.iterrows():
        subtest = test.copy()
        for i in range(len(attributes)):
            subtest = subtest[subtest[attributes[i]] == row[attributes[i]]]
        subtest_size = int(test_size * (row["count"] / total_size))
        if subtest_size > len(subtest):
            subtest_size = len(subtest)
        subtest_files = subtest.sample(subtest_size, random_state=SEED).UID.values
        in_test = data["UID"].isin(subtest_files)
        data.loc[in_test, "dataset"] = "test"

        subval_files = (
            data[data.dataset != "test"]
            .sample(subtest_size, random_state=SEED)
            .UID.values
        )
        in_val = data["UID"].isin(subval_files)
        data.loc[in_val, "dataset"] = "val"

    data.dataset = data.dataset.fillna("train")

    return data
