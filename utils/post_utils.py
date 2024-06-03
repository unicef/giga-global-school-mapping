import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import joblib
import torch
import sys

sys.path.insert(0, "../src")
import sat_download

sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import pred_utils
import embed_utils
import model_utils
import eval_utils
from matplotlib import colors

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import matplotlib as mpl

import logging

logging.basicConfig(level=logging.INFO)

LIGHT_YELLOW = "#ffe3c9"
YELLOW = "#ff9f40"
DARK_YELLOW = "#f47f20"
LIGHT_BLUE = "#d6e4fd"
BLUE = "#277aff"
DARK_BLUE = "#0530ad"
RED = "#f94b4b"

def get_evaluation(output, beta=2, optim_threshold=None):
    results = eval_utils.evaluate(
        y_true=output["y_true"], 
        y_pred=output["y_preds"], 
        y_prob=output["y_probs"], 
        pos_label=1, 
        neg_label=0,
        beta=beta,
        optim_threshold=optim_threshold
    )
    return results

def get_results(iso_code, model_config, phase="test", calibration=None):
    cwd = os.path.dirname(os.getcwd())
    exp_name = f"{iso_code}_{model_config['config_name']}"
    
    if calibration:
        output_path = os.path.join(
            cwd,  
            model_config["exp_dir"], 
            model_config["project"], 
            exp_name,
            f"{exp_name}_{phase}_{calibration}.csv"
        )
        output = pd.read_csv(output_path)
        return output

    output_path = os.path.join(
        cwd,  
        model_config["exp_dir"], 
        model_config["project"], 
        exp_name, 
        f"{exp_name}_{phase}.csv"
    )
    output = pd.read_csv(output_path)
    return output


def join_with_geoboundaries(
    master, preds, data_config, iso_code, threshold_dist=250, adm_level="ADM2"
):
    geoboundaries = data_utils._get_geoboundaries(
        data_config, iso_code, adm_level="ADM2"
    )
    intersection, master_filtered, preds_filtered = get_intersection(
        master, preds, threshold_dist=threshold_dist
    )
    sjoin = gpd.sjoin(master_filtered, geoboundaries)
    master_plot = geoboundaries.merge(
        sjoin.groupby("shapeName").count().geometry.rename("n_points").reset_index()
    )
    geoboundaries = geoboundaries.to_crs(preds_filtered.crs)
    sjoin = gpd.sjoin(preds_filtered, geoboundaries)
    preds_plot = geoboundaries.merge(
        sjoin.groupby("shapeName").count().geometry.rename("n_points").reset_index()
    )
    geoboundaries["diff"] = preds_plot["n_points"] - master_plot["n_points"]
    return master_plot, preds_plot, geoboundaries


def plot_choropleth(
    map, column="n_points", cmap="Blues", shrink=0.6, vmax=None, vcenter=None
):
    if vcenter:
        norm = colors.TwoSlopeNorm(
            vmin=np.min(map[column].values), vcenter=vcenter, vmax=vmax
        )
    ax = map.plot(
        column,
        legend=True,
        aspect=1,
        vmax=vmax,
        cmap=cmap,
        norm=None,
        legend_kwds={
            "shrink": shrink,
        },
    )
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def show_barplot(master, preds, threshold_dist=250, pad=50):
    plt.figure(dpi=1200)
    master_unconfirmed, preds_unvalidated = get_statistics(
        master, preds, threshold_dist
    )
    bins = (
        pd.cut(preds_unvalidated["prob"], [i*0.1 for i in range(1, 11)], 10)
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    y = [i * 0.9 for i in range(len(bins))]
    colors = [BLUE, BLUE, BLUE, DARK_BLUE, DARK_BLUE]
    labels = [f"{int(iv.left*100)}-{int(iv.right*100)}%" for iv in bins.index]

    bars = ax.barh(y, list(bins), height=0.5, align="edge", color=colors)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#A8BAC4", lw=1.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_capstyle("butt")

    ax.set_yticks(np.arange(0, len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel("Confidence Scores")
    ax.set_title(
        f"(Total: {len(preds_unvalidated):,})", loc="left", fontdict={"fontsize": 10}
    )
    fig.suptitle("Number of Model-identified Schools for Validation")
    fig._suptitle.set_size(12)

    for name, count, y_pos in zip(list(bins), list(bins), y):
        ax.text(
            pad,
            y_pos + 0.5 / 2,
            f"{name:,}",
            color="white",
            fontsize=11,
            va="center",
            path_effects=None,
        )


def get_intersection(
    master, preds, threshold_dist=150, distance_col="distance_to_nearest_master"
):
    master_filtered = master[master["clean"] == 0].drop_duplicates("geometry")
    # Filter predictions: if multiple predictions match to the same master
    #  data point, retain one and discard the other
    preds_filtered = preds[
        ~(
            (preds.sort_values(distance_col)["MUID"].duplicated(keep="first"))
            & (preds[distance_col] < threshold_dist)
        )
    ]
    intersection = preds_filtered[(preds_filtered[distance_col] < threshold_dist)]
    return intersection, master_filtered, preds_filtered


def get_statistics(master, preds, threshold_dist=150):
    intersection, master_filtered, preds_filtered = get_intersection(
        master, preds, threshold_dist
    )
    master_unconfirmed = master_filtered[
        ~master_filtered["MUID"].isin(intersection["MUID"])
    ]
    master_confirmed = master_filtered[
        master_filtered["MUID"].isin(intersection["MUID"])
    ]
    preds_unvalidated = preds_filtered[
        ~preds_filtered["PUID"].isin(intersection["PUID"])
    ]
    logging.info(f"{master_confirmed.shape}, {intersection.shape}")
    return master_unconfirmed, preds_unvalidated


def venn_diagram(master, preds, threshold_dist=150):
    plt.figure(dpi=1200)
    font = {"family": "serif", "weight": "normal", "size": 12}
    mpl.rc("font", **font)

    intersection, master_filtered, preds_filtered = get_intersection(
        master, preds, threshold_dist
    )
    master_unconfirmed, preds_unvalidated = get_statistics(
        master, preds, threshold_dist
    )
    subsets = (
        master_unconfirmed.shape[0],
        preds_unvalidated.shape[0],
        intersection.shape[0],
    )

    v = venn2(
        subsets=subsets,
        set_labels=("A", "B", "C"),
        set_colors=[LIGHT_BLUE, LIGHT_YELLOW],
        subset_label_formatter=lambda x: f"{x:,}",
    )
    c = venn2_circles(subsets=subsets)
    c[0].set_lw(1)
    c[0].set_alpha(0.5)
    c[1].set_lw(1)
    c[1].set_alpha(0.5)
    c[1].set_facecolor(LIGHT_YELLOW)
    c[1].set_edgecolor(YELLOW)
    c[0].set_facecolor(LIGHT_BLUE)
    c[0].set_edgecolor(BLUE)

    for text in v.set_labels:
        text.set_fontsize(8)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(13)

    v.get_label_by_id("A").set_text(
        f"Government-mapped \nschools locations \n({master_filtered.shape[0]:,} total)",
    )
    v.get_label_by_id("A").set_color(BLUE)
    v.get_label_by_id("B").set_text(
        f"Model-identified school \nlocations ({preds_filtered.shape[0]:,} total)"
    )
    v.get_label_by_id("B").set_color(DARK_YELLOW)
    v.get_label_by_id("11").set_text(f"{intersection.shape[0]:,}")

    plt.annotate(
        "Schools identified by both",
        fontsize=8,
        xy=v.get_label_by_id("11").get_position() - np.array([0, 0.05]),
        xytext=(0, -30),
        ha="center",
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6, edgecolor="white"),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.", color="gray"),
    )

    plt.show()


def donut_chart(master, source="master", source_col="clean"):
    plt.figure(dpi=1200)
    counts = master[source_col].value_counts()
    logging.info(counts)

    code = {
        0: "Unique school \nlocations",
        1: "Unpopulated \narea",
        2: "Non-unique school \nlocations",
    }
    sizes = list(counts.values)
    names = [f"{code[val]} ({size:,})" for val, size in zip(list(counts.index), sizes)]

    colors = {0: BLUE, 2: YELLOW, 1: RED}
    donut = plt.Circle((0, 0), 0.7, color="white")
    plt.rcParams["text.color"] = "#595959"
    plt.pie(sizes, labels=names, colors=[colors[v] for v in counts.keys()])
    p = plt.gcf()
    p.gca().add_artist(donut)
    plt.show()


def load_data(
    iso_code,
    data_config,
    model_config=None,
    sum_threshold=5,
    adm_level="ADM2",
    source="pred",
    source_col="clean",
    cam_model_config=False,
    buffer_size=0
):

    def join_with_geoboundary(data, adm_level):
        cols = list(data.columns)
        admin = data_utils._get_geoboundaries(
            data_config, iso_code, adm_level=adm_level
        )
        admin.rename(columns={"shapeName": adm_level}, inplace=True)
        data = gpd.sjoin(data, admin[["geometry", adm_level]], how="left")
        data = data[cols + [adm_level]]
        return data

    def clean_data(data, source_col):
        data.loc[(data[source_col] == 3), source_col] = 1
        data.loc[data.duplicated("geometry") & (data[source_col] == 1), source_col] = 1
        data.loc[data.duplicated("geometry") & (data[source_col] != 1), source_col] = 2
        data = join_with_geoboundary(data, adm_level="ADM1")
        data = join_with_geoboundary(data, adm_level="ADM2")
        data = join_with_geoboundary(data, adm_level="ADM3")
        return data

    cwd = os.path.dirname(os.getcwd())
    if source == "master":
        filename = os.path.join(
            cwd,
            data_config["vectors_dir"],
            data_config["project"],
            "sources",
            f"{iso_code}_{source}.geojson",
        )
        data = gpd.read_file(filename)
        data.loc[(data[source_col] == 1), source_col] = 0
        data = clean_data(data, source_col)
        data = data.rename({"UID": "MUID"}, axis=1)
        logging.info(data[source_col].value_counts())
        logging.info(f"Data dimensions: {data.shape}")
        return data

    elif "osm" in source or "overture" in source:
        filename = os.path.join(
            cwd,
            data_config["vectors_dir"],
            data_config["project"],
            "sources",
            f"{iso_code}_{source}.geojson",
        )
        data = gpd.read_file(filename)
        data = data[data[source_col] == 0]
        data = data.rename({"UID": "SUID"}, axis=1)
        data = data.drop_duplicates("SUID").reset_index(drop=True)
        logging.info(data[source_col].value_counts())
        logging.info(f"Data dimensions: {data.shape}")
        return data

    else:
        if not cam_model_config:
            cam_model_config = model_config
            
        model_config["iso_codes"] = [iso_code]
        out_dir = os.path.join(
            cwd,
            "output",
            iso_code,
            "results",
            model_config["project"],
            "cams",
            model_config["config_name"],
            cam_model_config["config_name"],
        )
        geoboundary = data_utils._get_geoboundaries(
            data_config, iso_code, adm_level=adm_level
        )

        data = []
        filenames = next(os.walk(out_dir), (None, None, []))[2]
        filenames = [filename for filename in filenames if "cam.gpkg" in filename]
        for filename in filenames:
            subdata = gpd.read_file(os.path.join(out_dir, filename))
            subdata = subdata[subdata["sum"] > sum_threshold]
            data.append(subdata)

        data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry", crs="EPSG:3857")
        data = data.reset_index(drop=True)

        data = data_utils._connect_components(data, buffer_size=0)
        data = data.sort_values("prob", ascending=False).drop_duplicates(["group"])

        data = data.to_crs("EPSG:4326")
        data = join_with_geoboundary(data, adm_level="ADM1")
        data = join_with_geoboundary(data, adm_level="ADM2")
        data = join_with_geoboundary(data, adm_level="ADM3")
        data = data.to_crs("EPSG:3857")

        data = data.drop_duplicates("geometry").reset_index(drop=True)
        data = data.reset_index(drop=True)
        
        data["PUID"] = data["ADM2"] + "_" + data["UID"].astype(str)
        data = data[~data.duplicated("PUID")]

        if buffer_size > 0:
            data = data_utils._connect_components(data, buffer_size=buffer_size)
            data = data.sort_values("prob", ascending=False).drop_duplicates(["group"])
        
        logging.info(f"Data dimensions: {data.shape}")

    return data
