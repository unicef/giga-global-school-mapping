import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import logging
import joblib
import torch
import sys

from src import sat_download
from utils import data_utils
from utils import post_utils
from utils import model_utils
from utils import eval_utils

import matplotlib.backends
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2, venn2_circles
import matplotlib.patches as mpatches
import matplotlib as mpl

logging.basicConfig(level=logging.INFO)

LIGHT_YELLOW = "#ffe3c9"
DARK_YELLOW = "#f47f20"
YELLOW = "#ff9f40"
LIGHT_BLUE = "#d6e4fd"
DARK_BLUE = "#0530ad"
BLUE = "#277aff"
RED = "#f94b4b"

font = {"family": "serif", "weight": "normal", "size": 12}
mpl.rc("font", **font)


def plot_heatmap(config: dict, exp_dir: str, project: str = "GIGAv1") -> None:
    """
    Plots a heatmap of AUPRC scores for model performance across different ISO codes.

    Args:
        config (dict): Configuration dictionary containing ISO codes and other settings.
        exp_dir (str): Directory where experiment results are stored.
    """
    data = dict()

    # Iterate over each ISO code in the configuration
    for iso_code in config:
        iso_dir = os.path.join(exp_dir, project, iso_code)
        iso_code_name = "Regional" if len(iso_code) < 3 else iso_code
        data[iso_code_name] = dict()

        # Compare performance with all other ISO codes
        for iso_code_test in list(config)[:-1]:
            if len(iso_code) < 3:
                # For regional codes, use ensemble models for test results
                results = model_utils.ensemble_models(iso_code, config, phase="test")
                results = results[results["iso"] == iso_code_test]
                results = eval_utils.evaluate(
                    y_true=results["y_true"],
                    y_pred=results["y_preds"],
                    y_prob=results["y_probs"],
                )
                data[iso_code_name][iso_code_test] = results["auprc"]
            elif iso_code == iso_code_test:
                # For the same ISO code, evaluate using ensemble models
                results = model_utils.ensemble_models(iso_code, config, phase="test")
                results = eval_utils.evaluate(
                    y_true=results["y_true"],
                    y_pred=results["y_preds"],
                    y_prob=results["y_probs"],
                )
                data[iso_code_name][iso_code_test] = results["auprc"]
            else:
                # Load results from file for different ISO codes
                results_name = f"{iso_code_test}_ensemble"
                results_file = os.path.join(iso_dir, results_name, "results.json")
                with open(results_file) as results:
                    results = json.load(results)
                    data[iso_code_name][iso_code_test] = results["test_auprc"]

    # Convert the data dictionary to a DataFrame
    data = pd.DataFrame(data).T

    # Create and save the heatmap
    s = sns.heatmap(data, cmap="viridis", annot=True, annot_kws={"fontsize": 8})
    for i in range(len(list(config)[:-1])):
        s.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=2))
    s.set(ylabel="Train country")
    s.set(xlabel="Test country")
    figure = s.get_figure()
    figure.savefig("assets/heatmap.pdf", dpi=500, bbox_inches="tight")


def plot_regional_vs_country(config: dict) -> dict:
    """
    Plots AUPRC scores for regional versus country-specific models.

    Args:
        config (dict): Configuration dictionary containing ISO codes and other settings.

    Returns:
        dict: A dictionary with AUPRC scores for regional and country-specific models.
    """
    # List of ISO codes excluding the last one (usually regional)
    iso_codes = list(config)[:-1]
    country_auprc = {"regional": [], "country-specific": []}

    # Iterate over each ISO code to compute AUPRC scores
    for iso_code in iso_codes:
        # Evaluate country-specific model performance
        country_test = model_utils.ensemble_models(iso_code, config, phase="test")
        auprc = eval_utils.evaluate(
            y_true=country_test["y_true"],
            y_pred=country_test["y_preds"],
            y_prob=country_test["y_probs"],
        )["auprc"]
        country_auprc["country-specific"].append(auprc)

        # Evaluate regional model performance for the current ISO code
        regional_test = model_utils.ensemble_models(
            list(config)[-1], config, phase="test"
        )
        regional_test = regional_test[regional_test["iso"] == iso_code]
        auprc = eval_utils.evaluate(
            y_true=regional_test["y_true"],
            y_pred=regional_test["y_preds"],
            y_prob=regional_test["y_probs"],
        )["auprc"]
        country_auprc["regional"].append(auprc)

    # Prepare data for plotting
    x = np.arange(len(iso_codes))
    width = 0.3
    multiplier = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 1.5), layout="constrained", dpi=300)
    colors = {"regional": "#277aff", "country-specific": "#ff9f40"}

    # Plot bars for regional and country-specific AUPRC scores
    for label, auprc in country_auprc.items():
        offset = width * multiplier
        ax.bar(x + offset, auprc, width, label=label.title(), color=colors[label])
        multiplier += 1

    # Customize plot
    ax.set_ylabel("AUPRC")
    ax.set_xticks(x + width, iso_codes)
    ax.set_ylim(0, 1.75)
    ax.legend(loc="upper left", ncols=2)

    # Display and save the plot
    plt.show()
    fig.savefig("assets/regional_auprc.pdf", bbox_inches="tight")

    return country_auprc


def plot_rurban(config: dict) -> None:
    """
    Plots AUPRC scores for rural versus urban classifications across different ISO codes.

    Args:
        config (dict): Configuration dictionary containing ISO codes and other settings.

    Returns:
        None
    """
    # List of ISO codes excluding the last one (usually regional)
    iso_codes = list(config)[:-1]
    rurban_auprc = {"rural": [], "urban": []}

    # Iterate over each ISO code to compute AUPRC scores for rural and urban classifications
    for iso_code in iso_codes:
        # model_config = config_utils.load_config(os.path.join(os.getcwd(), config[iso_code]))
        # test_output = calib_utils.get_model_output(iso_code, model_config, phase="test")

        # Load model outputs for the current ISO code
        test_output = model_utils.ensemble_models(iso_code, config, phase="test")

        # Compute AUPRC scores for rural and urban classifications
        for rurban in ["rural", "urban"]:
            subtest = test_output[test_output.rurban == rurban]
            auprc = eval_utils.evaluate(
                y_true=subtest["y_true"],
                y_pred=subtest["y_preds"],
                y_prob=subtest["y_probs"],
            )["auprc"]
            rurban_auprc[rurban].append(auprc)

    # Prepare data for plotting
    x = np.arange(len(iso_codes))
    width = 0.3
    multiplier = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 1.5), layout="constrained", dpi=300)
    colors = {"rural": "#277aff", "urban": "#ff9f40"}

    # Plot bars for rural and urban AUPRC scores
    for rurban, auprc in rurban_auprc.items():
        offset = width * multiplier
        ax.bar(x + offset, auprc, width, label=rurban.title(), color=colors[rurban])
        multiplier += 1

    # Customize plot
    ax.set_ylabel("AUPRC")
    ax.set_xticks(x + width, iso_codes)
    ax.set_ylim(0, 1.75)

    # Display and save the plot
    ax.legend(loc="upper left", ncols=2)
    plt.show()
    fig.savefig("assets/rurban_auprc.pdf", bbox_inches="tight")


def plot_choropleth(
    master: gpd.GeoDataFrame,
    preds: gpd.GeoDataFrame,
    config: dict,
    iso_code: str,
    threshold_dist: int = 250,
    adm_level: str = "ADM2",
) -> None:
    """
    Plots choropleth maps for master counts, predictions, and their differences.

    Args:
        master (gpd.GeoDataFrame): GeoDataFrame containing master counts.
        preds (gpd.GeoDataFrame): GeoDataFrame containing prediction counts.
        config (dict): Configuration dictionary containing parameters for map plotting.
        iso_code (str): ISO code for the geographical area.
        threshold_dist (int, optional): Distance threshold for calculating statistics. Defaults to 250.
        adm_level (str, optional): Administrative level for the map. Defaults to "ADM2".

    Returns:
        None
    """

    def subplot_choropleth(
        map: gpd.GeoDataFrame,
        column: str,
        cmap: str = "inferno_r",
        shrink: float = 0.5,
        vmin: float = None,
        vmax: float = None,
        title: str = "",
        label: str = "",
    ):
        """
        Creates and saves a choropleth map for a given column of the GeoDataFrame.

        Args:
            map (gpd.GeoDataFrame): GeoDataFrame to plot.
            column (str): Column name in the GeoDataFrame to plot.
            cmap (str, optional): Colormap to use for the map. Defaults to "inferno_r".
            shrink (float, optional): Shrink factor for the legend. Defaults to 0.5.
            vmin (float, optional): Minimum value for the color scale. Defaults to None.
            vmax (float, optional): Maximum value for the color scale. Defaults to None.
            title (str, optional): Title of the map. Defaults to an empty string.
            label (str, optional): Label for the map legend. Defaults to an empty string.

        Returns:
            None
        """
        fig, ax = plt.subplots(dpi=300)
        legend_kwds = {
            "shrink": shrink,
            "orientation": "horizontal",
            "pad": 0.015,
            "label": label,
        }
        map.plot(
            column,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            legend=True,
            legend_kwds=legend_kwds,
            ax=ax,
        )
        # Customize appearance
        # ax.set_title(title)
        ax.title.set_size(18)
        ax.labelcolor = "black"
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Save figure
        fig.savefig(f"assets/choropleth_{column}.pdf", bbox_inches="tight")

    # Calculate statistics per geoboundary
    map = calculate_stats_per_geoboundary(
        master, preds, config, iso_code, threshold_dist, adm_level=adm_level
    )
    vmax = np.max(
        [np.nanmax(map.master_counts.values), np.nanmax(map.preds_counts.values)]
    )

    # Plot choropleth maps for different data columns
    subplot_choropleth(
        map,
        column="master_counts",
        vmin=0,
        vmax=vmax,
        title="Government-registered Schools",
        label="Number of schools",
    )
    subplot_choropleth(
        map,
        column="preds_counts",
        vmin=0,
        vmax=vmax,
        title="Model Predictions",
        label="Number of model predictions",
    )
    subplot_choropleth(
        map,
        column="diff",
        vmin=0,
        vmax=vmax,
        title="Difference",
        label="Number of model predictions",
    )


def plot_venn_diagram(
    master: gpd.GeoDataFrame,
    preds: gpd.GeoDataFrame,
    threshold_dist: int = 250,
    labels: bool = True,
) -> None:
    """
    Plots a Venn diagram to visualize the overlap between government-mapped school locations and model predictions.

    Args:
        master (gpd.GeoDataFrame): GeoDataFrame containing master school locations.
        preds (gpd.GeoDataFrame): GeoDataFrame containing model predictions.
        threshold_dist (int, optional): Distance threshold for calculating statistics. Defaults to 250.
        labels (bool, optional): Whether to include detailed labels in the Venn diagram. Defaults to True.

    Returns:
        None
    """
    fig = plt.figure(dpi=300)

    master = master[master.clean == 0]
    matches = post_utils.match_dataframes(
        left_df=preds.to_crs("EPSG:3857"),
        right_df=master.to_crs("EPSG:3857"),
        distance_threshold=250,
    )

    master_unconfirmed = master.shape[0] - matches.shape[0]
    intersection = matches.shape[0]
    preds_unconfirmed = preds.shape[0] - matches.shape[0]
    subsets = (master_unconfirmed, preds_unconfirmed, intersection)

    # Plot Venn diagram
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

    # Customize text labels
    for text in v.set_labels:
        text.set_fontsize(10)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(13)

    if labels:
        v.get_label_by_id("A").set_text(
            f"Government-mapped \nschools locations \n({master.shape[0]:,} total)",
        )
        v.get_label_by_id("B").set_text(
            f"Model predictions \n({preds.shape[0]:,} total)"
        )
        v.get_label_by_id("11").set_text(f"{intersection:,}")

        # Annotate the intersection
        plt.annotate(
            "Schools identified by both",
            fontsize=10,
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
        v.get_label_by_id("A").set_text(f"{master.shape[0]:,} total")
        v.get_label_by_id("B").set_text(f"{preds.shape[0]:,} total")

    plt.show()
    fig.savefig("assets/venn_diagram.pdf", bbox_inches="tight")


def plot_pie_chart(
    master: pd.DataFrame,
    source: str = "master",
    source_col: str = "clean",
    legend: bool = False,
) -> None:
    """
    Plots a pie chart of the distribution of different types of school data points.

    Args:
        master (pd.DataFrame): DataFrame containing the school data.
        source (str, optional): Name of the source column to be used. Defaults to "master".
        source_col (str, optional): Column name in the DataFrame that contains the data to plot. Defaults to "clean".
        legend (bool, optional): Whether to include a legend in the pie chart. Defaults to False.

    Returns:
        None
    """
    fig = plt.figure(dpi=1200)
    font = {"family": "serif", "weight": "normal", "size": 12}
    mpl.rc("font", **font)

    plt.title("Government Data")
    colors = {0: BLUE, 2: YELLOW, 1: RED}
    code = {
        0: "Valid school \npoints",
        1: "Uninhabited \narea",
        2: "Duplicate school \npoints",
    }

    # Count the occurrences of each category
    counts = master[source_col].value_counts()
    sizes = list(counts.values)
    if legend:
        fig.legend(
            handles=[
                mpatches.Patch(color=colors[i], label=code[i]) for i in range(0, 3)
            ],
            loc="outside right",
        )

    # Create a donut-shaped pie chart
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
    master: gpd.GeoDataFrame,
    preds: gpd.GeoDataFrame,
    config: dict,
    iso_code: str,
    threshold_dist: int = 250,
    adm_level: str = "ADM2",
) -> gpd.GeoDataFrame:
    """
    Calculates statistics of master and prediction data per geographical boundary.

    Args:
        master (gpd.GeoDataFrame): GeoDataFrame containing the master (ground truth) data.
        preds (gpd.GeoDataFrame): GeoDataFrame containing the prediction data.
        config (dict): Configuration dictionary containing parameters for geoboundaries.
        iso_code (str): ISO code used to select specific geographical boundaries.
        threshold_dist (int, optional): Distance threshold for filtering. Defaults to 250.
        adm_level (str, optional): Administrative level for geographical boundaries. Defaults to "ADM2".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with additional columns for counts of master and prediction data,
            and their differences.
    """

    def get_counts(
        geoboundaries: gpd.GeoDataFrame, data: gpd.GeoDataFrame, name: str
    ) -> gpd.GeoDataFrame:
        """
        Calculates counts of data points within each geographical boundary.

        Args:
            geoboundaries (gpd.GeoDataFrame): GeoDataFrame containing the geographical boundaries.
            data (gpd.GeoDataFrame): GeoDataFrame with the data points to count.
            name (str): Prefix for the count column name.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with counts of data points.
        """
        # Ensure geoboundaries are in the same coordinate reference system as the data
        geoboundaries = geoboundaries.to_crs(data.crs)

        # Spatial join to count how many points fall into each boundary
        sjoin = gpd.sjoin(data, geoboundaries)

        # Count the number of points per boundary
        counts = (
            sjoin.groupby("shapeName").count().geometry.rename("counts").reset_index()
        )

        # Merge counts back into the original geoboundaries
        counts = geoboundaries[["shapeName"]].merge(
            counts,
            on="shapeName",
            how="left",
        )["counts"]

        # Add counts to the geoboundaries GeoDataFrame
        geoboundaries[f"{name}_counts"] = counts.fillna(0)

        return geoboundaries

    # Get geographical boundaries from the configuration
    geoboundaries = data_utils.get_geoboundaries(config, iso_code, adm_level=adm_level)

    # Calculate statistics for the master and prediction data
    statistics = calculate_stats(master, preds, threshold_dist=threshold_dist)
    master_filtered, preds_filtered = statistics[0], statistics[1]

    # Get counts of master and prediction data points within each geographical boundary
    geoboundaries = get_counts(geoboundaries, master_filtered, name="master")
    geoboundaries = get_counts(geoboundaries, preds_filtered, name="preds")

    # Calculate the difference between prediction and master counts
    geoboundaries["diff"] = (
        geoboundaries["preds_counts"] - geoboundaries["master_counts"]
    )
    return geoboundaries


def calculate_stats(
    master: pd.DataFrame,
    preds: pd.DataFrame,
    distance_col: str = "distance_to_nearest_master",
    threshold_dist: int = 250,
) -> tuple:
    """
    Calculates various statistics from master and prediction datasets based on distance thresholds.

    Args:
        master (pd.DataFrame): DataFrame containing master (ground truth) data with a column for cleaning status.
        preds (pd.DataFrame): DataFrame containing prediction data with columns for unique IDs and distance.
        distance_col (str, optional): Column name in `preds` DataFrame that contains distance to nearest master.
            Defaults to "distance_to_nearest_master".
        threshold_dist (int, optional): Distance threshold for filtering predictions. Defaults to 250.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - master_filtered (pd.DataFrame): Filtered master data where `clean` column is 0.
            - preds_filtered (pd.DataFrame): Filtered predictions based on distance threshold and duplicate removal.
            - intersection (pd.DataFrame): Predictions within the distance threshold.
            - master_unconfirmed (pd.DataFrame): Master data not found in the intersection.
            - preds_unconfirmed (pd.DataFrame): Predictions not found in the intersection.
    """
    # Filter master data to include only entries marked as "clean" == 0
    master_filtered = master[master["clean"] == 0]

    # Filter predictions based on distance threshold and remove duplicates
    preds_filtered = preds[
        ~(
            preds.sort_values(distance_col)["MUID"].duplicated(keep="first")
            & (preds[distance_col] < threshold_dist)
        )
    ]

    # Identify predictions within the distance threshold
    intersection = preds_filtered[(preds_filtered[distance_col] < threshold_dist)]

    # Identify master data not present in the intersection
    master_unconfirmed = master_filtered[
        ~master_filtered["MUID"].isin(intersection["MUID"])
    ]

    # Identify predictions not present in the intersection
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
