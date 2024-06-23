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
from utils import config_utils
from utils import pred_utils
from utils import post_utils
from utils import model_utils
from utils import eval_utils

import matplotlib.backends
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib_venn import venn2, venn2_circles
import matplotlib.patches as mpatches
import matplotlib as mpl

logging.basicConfig(level=logging.INFO)

LIGHT_YELLOW = "#ffe3c9"
YELLOW = "#ff9f40"
DARK_YELLOW = "#f47f20"
LIGHT_BLUE = "#d6e4fd"
BLUE = "#277aff"
DARK_BLUE = "#0530ad"
RED = "#f94b4b"


def plot_choropleth(
    master, preds, config, iso_code, threshold_dist=250, adm_level="ADM2"
):
    def subplot_choropleth(
        map, column, ax, cmap="inferno_r", shrink=0.5, vmin=None, vmax=None
    ):
        legend_kwds = {"shrink": shrink, "orientation": "horizontal", "loc": "botom"}
        map.plot(
            column,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            legend=True,
            legend_kwds=legend_kwds,
            ax=ax,
        )
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    map = calculate_stats_per_geoboundary(
        master, preds, config, iso_code, threshold_dist
    )
    vmax = np.max(
        [np.nanmax(map.master_counts.values), np.nanmax(map.preds_counts.values)]
    )
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 16))
    ax1 = subplot_choropleth(map, column="master_counts", vmin=0, vmax=vmax, ax=ax1)
    ax2 = subplot_choropleth(map, column="preds_counts", vmin=0, vmax=vmax, ax=ax2)
    ax3 = subplot_choropleth(map, column="diff", vmin=0, vmax=vmax, ax=ax3)


def plot_venn_diagram(master, preds, threshold_dist=250, labels=True):
    plt.figure(dpi=1200)
    font = {"family": "serif", "weight": "normal", "size": 12}
    mpl.rc("font", **font)

    preds = post_utils.calculate_nearest_distance(
        preds, master, source_name="master", source_uid="MUID"
    )
    master = post_utils.calculate_nearest_distance(
        master, preds, source_name="pred", source_uid="PUID"
    )
    (
        master_filtered,
        preds_filtered,
        intersection,
        master_unconfirmed,
        preds_unconfirmed,
    ) = calculate_stats(master, preds, threshold_dist=threshold_dist)

    subsets = (
        master_unconfirmed.shape[0],
        preds_unconfirmed.shape[0],
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

    if labels:
        v.get_label_by_id("A").set_text(
            f"Government-mapped \nschools locations \n({master_filtered.shape[0]:,} total)",
        )
        v.get_label_by_id("B").set_text(
            f"Model-identified school \nlocations ({preds_filtered.shape[0]:,} total)"
        )
        v.get_label_by_id("11").set_text(f"{intersection.shape[0]:,}")

        plt.annotate(
            "Schools identified by both",
            fontsize=8,
            xy=v.get_label_by_id("11").get_position() - np.array([0, 0.05]),
            xytext=(0, -30),
            ha="center",
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5", fc="white", alpha=0.6, edgecolor="white"
            ),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0.", color="gray"
            ),
        )
    else:
        v.get_label_by_id("A").set_text(f"{master_filtered.shape[0]:,} total")
        v.get_label_by_id("B").set_text(f"{preds_filtered.shape[0]:,} total")

    v.get_label_by_id("A").set_color(BLUE)
    v.get_label_by_id("B").set_color(DARK_YELLOW)

    plt.show()


def plot_pie_chart(master, source="master", source_col="clean", legend=False):
    fig = plt.figure(dpi=1200, layout="constrained")
    counts = master[source_col].value_counts()
    logging.info(counts)

    code = {
        0: "Valid school \npoints",
        1: "Uninhabited \narea",
        2: "Duplicate school \npoints",
    }
    sizes = list(counts.values)
    if legend:
        fig.legend(
            handles=[
                mpatches.Patch(color=colors[i], label=code[i]) for i in range(0, 3)
            ],
            loc="outside right",
        )

    donut = plt.Circle((0, 0), 0.7, color="white")
    plt.rcParams["text.color"] = "#595959"
    if legend:
        names = [f"{size:,}" for val, size in zip(list(counts.index), sizes)]
        plt.pie(
            sizes,
            labels=names,
            colors=[colors[v] for v in counts.keys()],
            textprops={"fontsize": 15},
        )
    else:
        names = [
            f"{code[val]} ({size:,})" for val, size in zip(list(counts.index), sizes)
        ]
        plt.pie(
            sizes,
            labels=names,
            colors=[colors[v] for v in counts.keys()],
            textprops={"fontsize": 12},
        )

    p = plt.gcf()
    p.gca().add_artist(donut)
    plt.show()


def calculate_stats_per_geoboundary(
    master, preds, config, iso_code, threshold_dist=250, adm_level="ADM2"
):
    def get_counts(geoboundaries, data, name):
        geoboundaries = geoboundaries.to_crs(data.crs)
        sjoin = gpd.sjoin(data, geoboundaries)
        counts = (
            sjoin.groupby("shapeName").count().geometry.rename("counts").reset_index()
        )
        counts = geoboundaries[["shapeName"]].merge(
            counts,
            on="shapeName",
            how="left",
        )["counts"]
        geoboundaries[f"{name}_counts"] = counts.fillna(0)
        return geoboundaries

    geoboundaries = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)
    statistics = calculate_stats(master, preds, threshold_dist=threshold_dist)
    master_filtered, preds_filtered = statistics[0], statistics[1]

    geoboundaries = get_counts(geoboundaries, master_filtered, name="master")
    geoboundaries = get_counts(geoboundaries, preds_filtered, name="preds")
    geoboundaries["diff"] = (
        geoboundaries["preds_counts"] - geoboundaries["master_counts"]
    )
    return geoboundaries


def calculate_stats(
    master, preds, distance_col="distance_to_nearest_master", threshold_dist=250
):
    master_filtered = master[master["clean"] == 0].drop_duplicates("geometry")
    preds_filtered = preds[
        ~(
            (preds.sort_values(distance_col)["MUID"].duplicated(keep="first"))
            & (preds[distance_col] < threshold_dist)
        )
    ]
    intersection = preds_filtered[(preds_filtered[distance_col] < threshold_dist)]
    master_unconfirmed = master_filtered[
        ~master_filtered["MUID"].isin(intersection["MUID"])
    ]
    preds_unconfirmed = preds_filtered[
        ~preds_filtered["PUID"].isin(intersection["PUID"])
    ]
    return (
        master_filtered,
        preds_filtered,
        intersection,
        master_unconfirmed,
        preds_unconfirmed,
    )