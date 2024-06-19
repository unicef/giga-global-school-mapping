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
from sklearn.metrics import log_loss, brier_score_loss

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42

CALIBRATORS = [
    IsotonicRegression(),
    TemperatureScaling(),
    HistogramBinning(),
    LogisticCalibration(),
    BetaCalibration(),
    BBQ(),
    ENIR(),
]


def calculate_bins(preds, labels, n_bins):
    bins = np.linspace(0.1, 1, n_bins)
    binned = np.digitize(preds, bins)

    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)

    for bin in range(n_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes


def calculate_metrics(preds, labels, n_bins):
    ECE, MCE = 0, 0
    bins, bin_accs, bin_confs, bin_sizes = calculate_bins(preds, labels, n_bins)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    return ECE, MCE


def reliability_diagram(preds, labels, n_bins=10):
    ECE, MCE = calculate_metrics(preds, labels, n_bins)
    bins, bin_accs, bin_confs, bin_sizes = calculate_bins(preds, labels, n_bins)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    ax.set_xlim(-0.05, 1)
    ax.set_xticks([x * 0.1 for x in range(0, 11)])
    ax.set_ylim(0, 1)
    ax.set_yticks([x * 0.1 for x in range(0, 11)])

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")

    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0, n_bins + 1) - width / 2
    bin_accs = np.insert(bin_accs, 0, 0)

    gap = np.array(bins - bin_accs)
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
    plt.plot([-width / 2, 1], [0, 1], "--", color="gray", linewidth=2)
    plt.gca().set_aspect("equal", adjustable="box")
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5, alpha=0.8)
    plt.text(
        0.15,
        0.9,
        "ECE: {:.4f} \nMCE: {:.4f}".format(ECE, MCE),
        ha="center",
        va="center",
        size=10,
        weight="normal",
        bbox=bbox_props,
    )


def calibration_curves(iso_code, config, highlight=None, phase="test", n_bins=10):
    fig, ax = plt.subplots(figsize=(6, 4))
    output = get_model_output(iso_code, config, phase=phase)
    CalibrationDisplay.from_predictions(
        output["y_true"],
        output["y_probs"],
        name="Uncalibrated",
        ax=ax,
        alpha=1.0,
        color="red",
        marker=None,
        n_bins=n_bins,
    )

    for calibrator in CALIBRATORS:
        output = get_calibrator_outputs(
            iso_code, config, calibrator, phase=phase, n_bins=n_bins
        )
        name = calibrator.__class__.__name__
        kwargs = {"alpha": 0.25, "marker": None}
        if name == highlight:
            kwargs = {"alpha": 1.0, "color": "blue", "marker": None}
        CalibrationDisplay.from_predictions(
            output["y_true"],
            output["y_probs_cal"],
            name=name,
            ax=ax,
            n_bins=n_bins,
            **kwargs,
        )
    plt.legend(loc="lower right", fontsize=7.5)


def calibration_metrics(confidences, ground_truth, phase, n_bins):
    ace = ACE(n_bins)
    ece = ECE(n_bins)
    mce = MCE(n_bins)

    return {
        f"{phase}_ace": ace.measure(confidences, ground_truth),
        f"{phase}_ece": ece.measure(confidences, ground_truth),
        f"{phase}_mce": mce.measure(confidences, ground_truth),
        f"{phase}_nll": log_loss(ground_truth, confidences),
        f"{phase}_briers": brier_score_loss(ground_truth, confidences),
    }


def compare_calibrators(iso_code, config, n_bins=10):
    results = dict()
    results["Uncalibrated"] = dict()
    for phase in ["val", "test"]:
        output = get_model_output(iso_code, config, phase=phase)
        results["Uncalibrated"].update(
            calibration_metrics(
                output["y_probs"].values, output["y_true"].values, phase, n_bins
            )
        )
        for calibrator in CALIBRATORS:
            calibrator = load_calibrator(iso_code, config, calibrator)
            calibrator_name = calibrator.__class__.__name__
            if calibrator_name not in results:
                results[calibrator_name] = dict()

            output = get_calibrator_outputs(iso_code, config, calibrator, phase=phase)
            results[calibrator_name].update(
                calibration_metrics(
                    output["y_probs_cal"].values, output["y_true"].values, phase, n_bins
                )
            )
    return results


def load_calibrator(iso_code, config, calibrator, phase="val"):
    calibrator_name = calibrator
    if isinstance(calibrator, str):
        calibrator_name = calibrator.__class__.__name__

    calibrator_file = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
        f"{iso_code}_{config['config_name']}_{calibrator_name}.pkl",
    )

    if not os.path.exists(calibrator_file):
        output = get_model_output(iso_code, config, phase=phase)
        confidences = output["y_probs"].values
        ground_truth = output["y_true"].values
        calibrator.fit(confidences, ground_truth)
        joblib.dump(calibrator, calibrator_file)

    calibrator = joblib.load(calibrator_file)
    return calibrator


def get_calibrator_outputs(iso_code, config, calibrator, phase="test", n_bins=10):
    calibrator_name = calibrator
    if isinstance(calibrator, str):
        calibrator_name = calibrator.__class__.__name__

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

    if not os.path.exists(output_path):
        output = get_model_output(iso_code, config, phase=phase)
        confidences = output["y_probs"].values
        calibrated = calibrator.transform(confidences)
        output["y_probs_cal"] = np.clip(calibrated, 0, 1)
        output.to_csv(output_path, index=False)

    output = pd.read_csv(output_path)
    return output


def get_model_output(iso_code, config, phase="test"):
    filename = f"{iso_code}_{config['config_name']}_{phase}.csv"
    output_path = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
        filename,
    )
    output = pd.read_csv(output_path)
    return output
