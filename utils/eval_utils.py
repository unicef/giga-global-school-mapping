import os
import json_fix
import json
import logging
import wandb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker

from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    auc,
    roc_auc_score,
    brier_score_loss,
    det_curve,
)
from functools import partial

json.fallback_table[np.ndarray] = lambda array: array.tolist()
json.fallback_table[np.integer] = lambda obj: int(obj)


def save_files(results: dict, cm: list[pd.DataFrame], exp_dir: str) -> None:
    """
    Saves the results and confusion matrix files to the specified directory.

    Args:
        results (dict): A dictionary containing the results to be saved in a JSON file.
        cm (list[pd.DataFrame]): A list of DataFrames where the first element is the confusion matrix,
                                 the second element contains confusion matrix metrics, and the third
                                 element is a string representing the confusion matrix report.
        exp_dir (str): The directory where the files will be saved.

    Returns:
        None
    """
    # Check if the directory exists; if not, create it
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Save the results dictionary to a JSON file
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f)

    # Save the confusion matrix DataFrame to a CSV file
    cm[0].to_csv(os.path.join(exp_dir, "confusion_matrix.csv"))
    # Save the confusion matrix metrics DataFrame to a CSV file
    cm[1].to_csv(os.path.join(exp_dir, "cm_metrics.csv"))

    # Append the confusion matrix report string to a log file
    open(os.path.join(exp_dir, "cm_report.log"), "a").write(cm[2])


def save_results(
    test: pd.DataFrame,
    target: str,
    pos_class: str,
    classes: list,
    results_dir: str,
    pred: str,
    prob: str,
    beta: float = 0.5,
    neg_class: int = 0,
    optim_threshold: float = None,
    prefix: str = None,
    log: bool = True,
) -> dict:
    """
    Evaluates results, saves confusion matrix and results to files, and logs results if required.

    Args:
        test (pd.DataFrame): The DataFrame containing the test data with actual and predicted values.
        target (str): The column name in `test` containing the actual target values.
        pos_class (str): The positive class label used in evaluation.
        classes (list): List of class labels for the confusion matrix.
        results_dir (str): The directory where the results and confusion matrix will be saved.
        pred (str): The column name in `test` containing the predicted values.
        prob (str): The column name in `test` containing the predicted probabilities.
        beta (float, optional): The beta parameter for the F-beta score. Defaults to 0.5.
        neg_class (str): The negative class label used in evaluation. Defaults to 0.
        optim_threshold (float): The threshold for optimization, if any. Defaults to None.
        prefix (str): A prefix to add to the result keys for identification. Defaults to None.
        log (bool): Whether to log results using `logging` and `wandb`. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluated results.
    """
    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Evaluate results based on the test data and specified parameters
    results = evaluate(
        test[target],
        test[pred],
        test[prob],
        pos_label=pos_class,
        beta=beta,
        optim_threshold=optim_threshold,
        neg_label=neg_class,
    )

    # If a prefix is provided, prepend it to each result key
    if prefix:
        results = {f"{prefix}_{key}": val for key, val in results.items()}

    # Filter out result keys that end with an underscore
    log_results = {key: val for key, val in results.items() if key[-1] != "_"}

    # Get confusion matrix and save the results and matrix to files
    cm = get_confusion_matrix(test[target], test[pred], classes)
    save_files(results, cm, results_dir)

    # Log results if specified
    if log:
        logging.info(log_results)
        wandb.log(log_results)

    return results


def get_confusion_matrix(y_true: list, y_pred: list, class_names: list) -> tuple:
    """
    Computes the confusion matrix, metrics from the matrix, and a classification report.

    Args:
        y_true (list): A list of true class labels.
        y_pred (list): A list of predicted class labels.
        class_names (list): A list of class names for the confusion matrix.

    Returns:
        tuple:
            - A DataFrame representing the confusion matrix.
            - A DataFrame containing metrics derived from the confusion matrix.
            - A string containing the classification report.
    """
    # Convert all inputs to strings for consistency
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]
    class_names = [str(x) for x in class_names]

    # Convert lists to pandas Series
    y_pred = pd.Series(y_pred, name="Predicted")
    y_true = pd.Series(y_true, name="Actual")

    # Compute confusion matrix and convert to DataFrame
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Compute metrics from confusion matrix
    cm_metrics = get_metrics(cm, list(cm.columns))

    # Generate classification report
    cm_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    return cm, cm_metrics, cm_report


def get_metrics(cm: pd.DataFrame, class_names: list) -> pd.DataFrame:
    """
    Computes precision, recall, and F1 score metrics from the confusion matrix for each class.

    Args:
        cm (pd.DataFrame): A DataFrame representing the confusion matrix with actual classes
            as rows and predicted classes as columns.
        class_names (list of str): A list of class names for which metrics are to be calculated.

    Returns:
        pd.DataFrame: A DataFrame containing precision, recall, and F1 score for each class.
    """
    # Dictionary to store metrics for each class
    metrics = {}

    for i in class_names:
        # True positives for class i
        tp = cm.loc[i, i]

        # False negatives for class i
        fn = cm.loc[i, :].drop(i).sum()

        # False positives for class i
        fp = cm.loc[:, i].drop(i).sum()

        # Calculate precision: tp / (tp + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate recall: tp / (tp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score: 2 / (1/recall + 1/precision)
        f1 = 2 / (recall**-1 + precision**-1) if precision + recall > 0 else 0

        # Store metrics in a dictionary
        scores = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100,
        }

        metrics[i] = scores

    # Convert the metrics dictionary to a DataFrame
    metrics = pd.DataFrame(metrics).T

    return metrics


def get_optimal_threshold(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, beta: float = 0.5
) -> tuple:
    """
    Calculates the optimal threshold for classification based on the F-score.

    Args:
        precision (np.ndarray): Array of precision scores for each threshold.
        recall (np.ndarray): Array of recall scores for each threshold.
        thresholds (np.ndarray): Array of threshold values.
        beta (float, optional): Weight of recall in the F-score. Default is 0.5.

    Returns:
        tuple:
            - threshold (float): The optimal threshold that maximizes the F-score.
            - fscores (numpy.ndarray): Array of F-scores corresponding to each threshold.
    """
    # Calculate the numerator of the F-score formula
    numerator = (1 + beta**2) * precision * recall

    # Calculate the denominator of the F-score formula
    denom = ((beta**2) * precision) + recall

    # Compute the F-scores while avoiding division by zero
    fscores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

    # Find the threshold with the maximum F-score
    threshold = thresholds[np.argmax(fscores)]

    return threshold, fscores


def auprc(recall: np.ndarray, precision: np.ndarray, min_precision: float = 0) -> float:
    """
    Computes the Area Under the Precision-Recall Curve (AUPRC).

    Args:
        recall (np.ndarray): Array of recall values.
        precision (np.ndarray): Array of precision values corresponding to each recall value.
        min_precision (float, optional): Minimum precision value to account for the baseline. Default is 0.

    Returns:
        float: The computed AUPRC value.
    """
    # Calculate the area under the precision-recall curve by summing the areas of trapezoids
    auc_ = -np.sum(np.diff(recall) * (np.array(precision)[:-1]))

    # Compute the maximum possible area with the minimum precision value
    max_area = min_precision * np.max(recall)

    return auc_ - max_area


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    pos_label: int = 1,
    neg_label: int = 0,
    beta: float = 2,
    optim_threshold: float = None,
    min_precision: float = 0.9,
) -> float:
    """
    Evaluates various performance metrics for a classification model.

    Args:
        y_true (np.ndarray or list): True binary labels.
        y_pred (np.ndarray or list): Predicted binary labels.
        y_prob (np.ndarray or list): Predicted probabilities for the positive class.
        pos_label (int, optional): Label for the positive class. Default is 1.
        neg_label (int, optional): Label for the negative class. Default is 0.
        beta (float, optional): Weight of recall in the F-beta score. Default is 2.
        optim_threshold (float, optional): Threshold to use for binary classification. If None, it will be computed.
        min_precision (float, optional): Minimum precision value to consider in AUPRC calculation. Default is 0.9.

    Returns:
        dict: A dictionary containing various performance metrics, including:
            - "auprc": Area Under the Precision-Recall Curve.
            - "ap": Average Precision score.
            - "roc_auc": Area Under the ROC Curve.
            - "brier_score": Brier score loss.
            - "precision_scores_": Array of precision scores for different thresholds.
            - "recall_scores_": Array of recall scores for different thresholds.
            - "thresholds_": Array of thresholds.
            - "optim_threshold": Optimal threshold that maximizes the F-beta score.
            - "fbeta_score": F-beta score at the optimal threshold.
            - "precision_score": Precision score at the optimal threshold.
            - "recall_score": Recall score at the optimal threshold.
            - "overall_accuracy": Accuracy score at the optimal threshold.
            - "balanced_accuracy": Balanced accuracy score at the optimal threshold.
    """
    # Calculate precision, recall, and thresholds for different probability thresholds
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_prob, pos_label=pos_label
    )

    # Compute the optimal threshold if not provided
    if not optim_threshold:
        optim_threshold, _ = get_optimal_threshold(
            precision[:-1], recall[:-1], pr_thresholds, beta=beta
        )

    # Generate predictions based on the optimal threshold
    y_pred_optim = [pos_label if val > optim_threshold else neg_label for val in y_prob]
    fpr, fnr, det_thresholds = det_curve(y_true, y_prob)

    return {
        # Performance metrics for the full range of thresholds
        "auprc": auprc(recall, precision),
        "ap": average_precision_score(y_true, y_prob, pos_label=pos_label),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob, pos_label=pos_label),
        "precision_scores_": precision,
        "recall_scores_": recall,
        "pr_thresholds": pr_thresholds,
        "det_thresholds": det_thresholds,
        "fpr": fpr,
        "fnr": fnr,
        # Performance metrics at the optimal threshold
        "optim_threshold": optim_threshold,
        "fbeta_score": fbeta_score(
            y_true,
            y_pred_optim,
            beta=beta,
            pos_label=pos_label,
            average="binary",
            zero_division=0,
        )
        * 100,
        "precision_score": precision_score(
            y_true, y_pred_optim, pos_label=pos_label, average="binary", zero_division=0
        )
        * 100,
        "recall_score": recall_score(
            y_true, y_pred_optim, pos_label=pos_label, average="binary", zero_division=0
        )
        * 100,
        "overall_accuracy": accuracy_score(y_true, y_pred_optim) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_optim) * 100,
    }


def plot_results(results, plot="det"):
    if plot == "pr":
        thresholds = results["pr_thresholds"]
        right = results["precision_scores_"][:-1]
        right_label = "Precision"
        left = results["recall_scores_"][:-1]
        left_label = "Recall"

    elif plot == "det":
        thresholds = results["det_thresholds"]
        right = results["fnr"]
        right_label = "False Negative Rate (FNR)"
        left = results["fpr"]
        left_label = "False Positive Rate (FPR)"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, right, "b--")
    ax.plot(thresholds, left, "g--")
    ax.set_ylabel(left_label)
    ax.yaxis.label.set_color("green")
    ax2 = ax.twinx()
    l1 = ax.get_ylim()
    l2 = ax2.get_ylim()

    def func(x):
        return l2[0] + (x - l1[0]) / (l1[1] - l1[0]) * (l2[1] - l2[0])

    ticks = func(ax.get_yticks())
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax2.set_yticklabels(ax.get_yticklabels())
    ax2.set_ylabel(right_label, rotation=270, labelpad=15)
    ax2.yaxis.label.set_color("blue")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax.set_xlabel("Probability Threshold")
    loc = plticker.MultipleLocator(
        base=0.1
    )  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
