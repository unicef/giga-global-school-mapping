import os
import json_fix
import json
import logging
import wandb
import pandas as pd
import numpy as np
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
    brier_score_loss
)

json.fallback_table[np.ndarray] = lambda array: array.tolist()


def _save_files(results, cm, exp_dir):
    """
    Save evaluation results and confusion matrix to the specified directory.
    Args:
    - results (dict): Evaluation results to be saved as JSON.
    - cm (tuple): Tuple containing confusion matrix components (DataFrame, DataFrame, str).
    - exp_dir (str): Directory path to save the results.
    Saves:
    - "results.json": JSON file containing the evaluation results.
    - "confusion_matrix.csv": CSV file containing the confusion matrix data.
    - "cm_metrics.csv": CSV file containing metrics derived from the confusion matrix.
    - "cm_report.log": Log file containing the detailed confusion matrix report.
    """
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f)
    cm[0].to_csv(os.path.join(exp_dir, "confusion_matrix.csv"))
    cm[1].to_csv(os.path.join(exp_dir, "cm_metrics.csv"))
    open(os.path.join(exp_dir, "cm_report.log"), "a").write(cm[2])


def save_results(test, target, pos_class, classes, results_dir, pred, prob, beta=0.5, prefix=None, log=True):
    """
    Save evaluation results and confusion matrix to the specified directory.

    Args:
    - results (dict): Evaluation results to be saved as JSON.
    - cm (tuple): Tuple containing confusion matrix components (DataFrame, DataFrame, str).
    - exp_dir (str): Directory path to save the results.

    Saves:
    - "results.json": JSON file containing the evaluation results.
    - "confusion_matrix.csv": CSV file containing the confusion matrix data.
    - "cm_metrics.csv": CSV file containing metrics derived from the confusion matrix.
    - "cm_report.log": Log file containing the detailed confusion matrix report.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results = evaluate(test[target], test[pred], test[prob], pos_class, beta)
    log_results = {key: val for key, val in results.items() if key[-1] != '_'}
    cm = get_confusion_matrix(test[target], test[pred], classes)
    _save_files(results, cm, results_dir)
    
    if prefix: 
        log_results = {f"{prefix}_{key}": val for key, val in log_results.items()}
    if log: 
        logging.info(log_results)
        wandb.log(log_results)
    return results


def get_confusion_matrix(y_true, y_pred, class_names):
    """Generates the confusion matrix given the predictions
    and ground truth values.

    Args:
    - y_test (list or numpy array): A list of ground truth values.
    - y_pred (list of numpy array): A list of prediction values.
    - class_names (list): A list of string labels or class names.

    Returns:
    - pandas DataFrame: The confusion matrix.
    - pandas DataFrame: A dataframe containing the precision,
            recall, and F1 score per class.
    """
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]
    class_names = [str(x) for x in class_names]

    y_pred = pd.Series(y_pred, name="Predicted")
    y_true = pd.Series(y_true, name="Actual")
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_metrics = _get_metrics(cm, list(cm.columns))
    cm_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return cm, cm_metrics, cm_report


def _get_metrics(cm, class_names):
    """Return the precision, recall, and F1 score per class.

    Args:
    - cm (pandas DataFrame or numpy array): The confusion matrix.
    - class_names (list): A list of string labels or class names.

    Returns:
    - pandas DataFrame: A dataframe containing the precision,
    recall, and F1 score per class.
    """

    metrics = {}
    for i in class_names:
        tp = cm.loc[i, i]
        fn = cm.loc[i, :].drop(i).sum()
        fp = cm.loc[:, i].drop(i).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 / (recall**-1 + precision**-1) if precision + recall > 0 else 0

        scores = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100,
        }

        metrics[i] = scores
    metrics = pd.DataFrame(metrics).T

    return metrics


def evaluate(y_true, y_pred, y_prob, pos_label, beta=0.5):
    """Returns a dictionary of performance metrics.

    Args:
    - y_true (list or numpy array): A list of ground truth values.
    - y_pred (list of numpy array): A list of prediction values.

    Returns:
    - dict: A dictionary of performance metrics.
    """
    y_prob_50 = [val if val > 0.50 else 0 for val in y_prob]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    precision_50, recall_50, thresholds_50 = precision_recall_curve(y_true, y_prob_50, pos_label=pos_label)

    return {
        "fbeta_score": fbeta_score(
            y_true, y_pred, beta=beta, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "f1_score": f1_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "precision_score": precision_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "recall_score": recall_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "overall_accuracy": accuracy_score(
            y_true, y_pred
        ) * 100,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ) * 100,
        "ap": average_precision_score(y_true, y_prob, pos_label=pos_label),
        "auprc": auc(recall, precision),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob, pos_label=pos_label),
        "precision_scores_": precision,
        "recall_scores_": recall,
        "thresholds_": thresholds,
        "ap_50": average_precision_score(y_true, y_prob_50, pos_label=pos_label),
        "auprc_50": auc(recall_50, precision_50),
        "roc_auc_50": roc_auc_score(y_true, y_prob_50),
        "brier_score_50": brier_score_loss(y_true, y_prob_50, pos_label=pos_label),
        "precision_scores_50_": precision,
        "recall_scores_50_": recall,
        "thresholds_50_": thresholds
    }


def get_scoring(pos_label, beta=0.5):
    """Returns the dictionary of scorer objects."""
    return {"fbeta_score": make_scorer(fbeta_score, beta=beta, pos_label=pos_label, average="binary")}