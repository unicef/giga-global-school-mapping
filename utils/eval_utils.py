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
json.fallback_table[np.integer] = lambda obj: int(obj)


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


def save_results(
    test, 
    target, 
    pos_class, 
    classes, 
    results_dir, 
    pred, 
    prob, 
    beta=0.5, 
    neg_class=0,
    optim_threshold=None, 
    prefix=None, 
    log=True
):
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
    results = evaluate(
        test[target], 
        test[pred], 
        test[prob], 
        pos_label=pos_class, 
        beta=beta, 
        optim_threshold=optim_threshold, 
        neg_label=neg_class
    )
    if prefix: 
        results = {f"{prefix}_{key}": val for key, val in results.items()}
    
    log_results = {key: val for key, val in results.items() if key[-1] != '_'}
    cm = get_confusion_matrix(test[target], test[pred], classes)
    _save_files(results, cm, results_dir)
    
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

def get_optimal_threshold(precision, recall, thresholds, beta=0.5):
    numerator = (1 + beta**2) * precision * recall
    denom = ((beta**2) * precision) + recall
    fscores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    threshold = thresholds[np.argmax(fscores)]
    return threshold, fscores


def evaluate(
    y_true, 
    y_pred, 
    y_prob, 
    pos_label=1, 
    neg_label=0, 
    beta=0.5, 
    optim_threshold=None, 
    default_threshold=0.5
):
    """Returns a dictionary of performance metrics.

    Args:
    - y_true (list or numpy array): A list of ground truth values.
    - y_pred (list of numpy array): A list of prediction values.

    Returns:
    - dict: A dictionary of performance metrics.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)

    idx_threshold = np.where(np.array(thresholds) > default_threshold)
    precision_partial = precision[idx_threshold]
    recall_partial = recall[idx_threshold]
    thresholds_partial = thresholds[idx_threshold]

    y_pred_optim = default_threshold
    if not optim_threshold and len(thresholds_partial) > 0:
        optim_threshold, _ = get_optimal_threshold(
            precision_partial, recall_partial, thresholds_partial, beta=beta
        )
        y_pred_optim = [pos_label if val > optim_threshold else neg_label for val in y_prob]
    
    return {
        # Performance metrics for probabilities > 0.5 threshold
        "p_auprc": auc(recall_partial, precision_partial),
        "p_precision_scores_": precision_partial,
        "p_recall_scores_": recall_partial,
        "p_thresholds_": thresholds_partial,
        # Performance metrics for the full range of thresholds
        "ap": average_precision_score(y_true, y_prob, pos_label=pos_label),
        "auprc": auc(recall, precision),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob, pos_label=pos_label),
        "precision_scores_": precision,
        "recall_scores_": recall,
        "thresholds_": thresholds,
        # Performance metrics at the optimal threshold
        "optim_threshold": optim_threshold,
        "fbeta_score_optim": fbeta_score(
            y_true, y_pred_optim, beta=beta, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "precision_score_optim": precision_score(
            y_true, y_pred_optim, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "recall_score_optim": recall_score(
            y_true, y_pred_optim, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "f1_score_optim": f1_score(
            y_true, y_pred_optim, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "overall_accuracy_optim": accuracy_score(
            y_true, y_pred_optim
        ) * 100,
        "balanced_accuracy_optim": balanced_accuracy_score(
            y_true, y_pred_optim
        ) * 100,
        # Performance metrics for hard predictions
        "fbeta_score": fbeta_score(
            y_true, y_pred, beta=beta, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "precision_score": precision_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "recall_score": recall_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "f1_score": f1_score(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        ) * 100,
        "overall_accuracy": accuracy_score(
            y_true, y_pred
        ) * 100,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ) * 100,
    }


def get_scoring(pos_label, beta=0.5):
    """Returns the dictionary of scorer objects."""
    return {"ap_50": make_scorer(average_precision_score, needs_proba=True, pos_label=pos_label)}