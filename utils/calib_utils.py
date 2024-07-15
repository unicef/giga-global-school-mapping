import os
import sys
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.calibration import CalibrationDisplay

from netcal.metrics import ACE, ECE, MCE, NLL
from netcal.binning import IsotonicRegression, HistogramBinning, BBQ, ENIR
from netcal.scaling import TemperatureScaling, LogisticCalibration, BetaCalibration

from utils import post_utils
from utils import pred_utils
from utils import cnn_utils
from utils import model_utils
from utils import config_utils
from utils import data_utils
from sklearn.metrics import log_loss, brier_score_loss, average_precision_score

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42

CALIBRATORS = {
    "IsotonicRegression": IsotonicRegression(),
    "TemperatureScaling": TemperatureScaling(),
    "LogisticCalibration": LogisticCalibration(),
    "BetaCalibration": BetaCalibration(),
}


def calculate_bins(preds: np.ndarray, labels: np.ndarray, n_bins: int):
    """
    Calculate the bins, accuracies, confidences, and sizes for calibration.

    Args:
        preds (array-like): Predicted probabilities.
        labels (array-like): Ground truth labels.
        n_bins (int): Number of bins for calibration.

    Returns:
        tuple: A tuple containing bins, bin accuracies, bin confidences, and bin sizes.
    """
    # Create evenly spaced bins from 0.1 to 1
    bins = np.linspace(0.1, 1, n_bins)
    # Digitize predictions into bins
    binned = np.digitize(preds, bins)

    # Initialize bin accuracies, confidences, and sizes
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)

    for bin in range(n_bins):
        # Calculate size of each bin
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            # Calculate accuracy of each bin
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            # Calculate confidence of each b
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes


def calculate_metrics(preds: np.ndarray, labels: np.ndarray, n_bins: int):
    """
    Calculate the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        preds (array-like): Predicted probabilities.
        labels (array-like): Ground truth labels.
        n_bins (int): Number of bins for calibration.

    Returns:
        tuple: A tuple containing ECE and MCE.
    """
    ece, mce = 0, 0

    # Calculate bins, accuracies, confidences, and sizes
    bins, bin_accs, bin_confs, bin_sizes = calculate_bins(preds, labels, n_bins)
    for i in range(len(bins)):
        # Calculate the absolute difference between accuracy and confidence
        abs_conf_diff = abs(bin_accs[i] - bin_confs[i])
        # Update ECE
        ece += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_diff
        # Update MCE
        mce = max(mce, abs_conf_diff)
    return ece, mce


def reliability_diagram(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Plot reliability diagram to visualize the calibration of a model.

    Args:
        preds (array-like): Predicted probabilities.
        labels (array-like): Ground truth labels.
        n_bins (int, optional): Number of bins for the diagram. Defaults to 10.

    Returns:
        None: The function generates a reliability diagram plot.
    """
    # Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    ece, mce = calculate_metrics(preds, labels, n_bins)

    # Calculate bins, accuracies, confidences, and sizes for the reliability diagram
    bins, bin_accs, bin_confs, bin_sizes = calculate_bins(preds, labels, n_bins)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    # Set axis limits and ticks
    ax.set_xlim(-0.05, 1)
    ax.set_xticks([x * 0.1 for x in range(0, 11)])
    ax.set_ylim(0, 1)
    ax.set_yticks([x * 0.1 for x in range(0, 11)])

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    # Enable grid
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")

    # Prepare bins for plotting
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0, n_bins + 1) - width / 2
    bin_accs = np.insert(bin_accs, 0, 0)

    # Calculate gaps between bins and accuracies
    gap = np.array(bins - bin_accs)

    # Plot bars for the reliability diagram
    plt.bar(bin_centers, bins, width=width, alpha=0.3, edgecolor="black", color="r")
    plt.bar(bin_centers, bin_accs, width=width, alpha=1, edgecolor="black", color="b")
    plt.bar(
        bin_centers,
        gap,
        bottom=bin_accs,
        color=[1, 0.7, 0.7],
        alpha=0.5,
        width=width,
        hatch="//",
        edgecolor="r",
    )

    # Plot the perfect calibration line
    plt.plot([-width / 2, 1], [0, 1], "--", color="gray", linewidth=2)

    # Add text box with ECE and MCE values
    plt.gca().set_aspect("equal", adjustable="box")
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5, alpha=0.8)
    plt.text(
        0.15,
        0.9,
        "ECE: {:.4f} \nMCE: {:.4f}".format(ece, mce),
        ha="center",
        va="center",
        size=10,
        weight="normal",
        bbox=bbox_props,
    )


def calibration_curves(
    iso_code: str,
    config: dict,
    highlight: str = None,
    phase: str = "test",
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curves for the uncalibrated model and various calibrators.

    Args:
        iso_code (str): ISO code for the dataset.
        config (dict): Configuration dictionary with experiment details.
        highlight (str, optional): Name of the calibrator to highlight in the plot. Defaults to None.
        phase (str, optional): Phase of the dataset (e.g., 'val', 'test'). Defaults to "test".
        n_bins (int, optional): Number of bins for calibration curves. Defaults to 10.

    Returns:
        None: The function generates a plot of calibration curves.
    """

    from matplotlib.backends.backend_pdf import PdfPages

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Get model output for the specified phase
    # output = get_model_output(iso_code, config, phase=phase)
    output = model_utils.ensemble_models(iso_code, config, phase=phase)

    # Plot uncalibrated calibration curve
    CalibrationDisplay.from_predictions(
        output["y_true"],
        output["y_probs"],
        name="Uncalibrated (baseline)",
        ax=ax,
        alpha=1.0,
        color="red",
        marker=None,
        n_bins=n_bins,
    )

    # Plot calibration curves for each calibrator
    for name, calibrator in CALIBRATORS.items():
        output = get_calibrator_outputs(
            iso_code, config, calibrator, phase=phase, n_bins=n_bins
        )
        kwargs = {"alpha": 0.25, "marker": None}

        # Highlight the specified calibrator
        if name == highlight:
            kwargs = {"alpha": 1.0, "color": "blue", "marker": None}

        if name != "BBQ" and name != "ENIR":
            name = "".join([" " + s if s.isupper() else s for s in name]).lstrip()

        CalibrationDisplay.from_predictions(
            output["y_true"],
            output["y_probs_cal"],
            name=name,
            ax=ax,
            n_bins=n_bins,
            **kwargs,
        )

    plt.legend(loc="lower right", fontsize=8)
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    fig.savefig("assets/calibration_curve.pdf", bbox_inches="tight")


def calibration_metrics(
    confidences: np.ndarray, ground_truth: np.ndarray, phase: str, n_bins: int
) -> dict:
    """
    Calculate calibration metrics for given confidences and ground truth labels.

    Args:
        confidences (np.ndarray): Array of predicted probabilities.
        ground_truth (np.ndarray): Array of true labels.
        phase (str): Phase of the dataset (e.g., 'val', 'test').
        n_bins (int): Number of bins for calibration metrics.

    Returns:
        dict: Dictionary containing calibration metrics.
    """
    # Initialize calibration metric objects
    ace = ACE(n_bins)
    ece = ECE(n_bins)
    mce = MCE(n_bins)

    # Calculate and return calibration metrics
    return {
        f"{phase}_ace": ace.measure(confidences, ground_truth),
        f"{phase}_ece": ece.measure(confidences, ground_truth),
        f"{phase}_mce": mce.measure(confidences, ground_truth),
        f"{phase}_nll": log_loss(ground_truth, confidences),
        f"{phase}_briers": brier_score_loss(ground_truth, confidences),
        f"{phase}_auprc": average_precision_score(
            ground_truth, confidences, pos_label=1
        ),
    }


def compare_calibrators(iso_code: str, config: dict, n_bins: int = 10) -> dict:
    """
    Compare different calibrators by evaluating their calibration metrics.

    Args:
        iso_code (str): ISO code of the dataset.
        config (dict): Configuration dictionary with keys:
            - config_name (str): Name of the configuration.
            - exp_dir (str): Directory where experiments are stored.
            - project (str): Name of the project.
        n_bins (int, optional): Number of bins for the calibration curve. Defaults to 10.

    Returns:
        dict: Dictionary containing calibration metrics for each calibrator and phase.
    """
    # Initialize the results dictionary
    results = dict()
    uncalibrated_name = "Uncalibrated (baseline)"
    results[uncalibrated_name] = dict()

    # Iterate through validation and test phases
    for phase in ["val", "test"]:
        # Get model output for the current phase
        # output = get_model_output(iso_code, config, phase=phase)
        output = model_utils.ensemble_models(iso_code, config, phase=phase)

        # Calculate and store calibration metrics for uncalibrated outputs
        results[uncalibrated_name].update(
            calibration_metrics(
                output["y_probs"].values, output["y_true"].values, phase, n_bins
            )
        )
        # Iterate through each calibrator defined in CALIBRATORS
        for calibrator_name, calibrator in CALIBRATORS.items():
            calibrator = load_calibrator(iso_code, config, calibrator)

            # Initialize results for the calibrator if not already present
            if calibrator_name not in results:
                results[calibrator_name] = dict()

            # Get calibrated outputs for the current phase
            output = get_calibrator_outputs(iso_code, config, calibrator, phase=phase)

            # Calculate and store calibration metrics for calibrated outputs
            results[calibrator_name].update(
                calibration_metrics(
                    output["y_probs_cal"].values, output["y_true"].values, phase, n_bins
                )
            )
    return results


def load_calibrator(iso_code: str, config: dict, calibrator, phase: str = "val"):
    """
    Load the calibrator model from a file or train and save it if it doesn't exist.

    Args:
        iso_code (str): ISO code of the dataset.
        config (dict): Configuration dictionary with keys:
            - config_name (str): Name of the configuration.
            - exp_dir (str): Directory where experiments are stored.
            - project (str): Name of the project.
        calibrator: Calibrator object or string representing its class name.
        phase (str, optional): Phase of the data ('val', 'train', 'test'). Defaults to "val".

    Returns:
        calibrator: Trained calibrator model.
    """
    # Determine the calibrator name
    calibrator_name = calibrator
    if not isinstance(calibrator, str):
        calibrator_name = calibrator.__class__.__name__

    # Construct the path to the calibrator file
    calibrator_file = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
        f"{iso_code}_{config['config_name']}_{calibrator_name}.pkl",
    )
    data_utils.makedir(os.path.dirname(calibrator_file))

    # Check if the calibrator file exists
    # if not os.path.exists(calibrator_file):
    # Load model output and fit the calibrator
    calibrator = CALIBRATORS[calibrator_name]
    output = model_utils.ensemble_models(iso_code, config, phase=phase)
    confidences = output["y_probs"].values
    ground_truth = output["y_true"].values
    calibrator.fit(confidences, ground_truth)
    joblib.dump(calibrator, calibrator_file)

    # Load and return the calibrator from the file
    calibrator = joblib.load(calibrator_file)
    return calibrator


def get_calibrator_outputs(
    iso_code: str, config: dict, calibrator, phase: str = "test", n_bins: int = 10
) -> pd.DataFrame:
    """
    Get the outputs from the calibrator and save or load from CSV.

    Args:
        iso_code (str): ISO code of the dataset.
        config (dict): Configuration dictionary with keys:
            - config_name (str): Name of the configuration.
            - exp_dir (str): Directory where experiments are stored.
            - project (str): Name of the project.
        calibrator: Calibrator object or string representing its class name.
        phase (str, optional): Phase of the data ('test', 'train', 'val'). Defaults to "test".
        n_bins (int, optional): Number of bins for calibration curve. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing the calibrator outputs.
    """
    # Determine the calibrator name
    calibrator_name = calibrator
    if not isinstance(calibrator, str):
        calibrator_name = calibrator.__class__.__name__

    # Construct the filename and output path
    filename = "{}_{}_{}_{}.csv".format(
        iso_code, config["config_name"], phase, calibrator_name
    )
    output_path = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
        filename,
    )

    # Read and return the output CSV as a DataFrame
    output = pd.read_csv(output_path)
    return output
