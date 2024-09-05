import os
import shutil
import time
import argparse
import logging
import pandas as pd

import torch
import wandb
import torch.nn as nn

from utils import config_utils
from utils import cnn_utils
from utils import eval_utils
from utils import model_utils
from utils import pred_utils


# Get device
cwd = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")


def main(config, project):
    # Create experiment folder
    exp_name = "cv"
    exp_dir = os.path.join(cwd, exp_name, project)
    logging.info(f"Experiment directory: {exp_dir}")

    dataloaders = dict()
    for iso_code in config:
        model_config = config_utils.load_config(config[iso_code][0])
        model_config["iso_codes"] = [iso_code]
        data, data_loader, classes = cnn_utils.load_dataset(
            config=model_config, phases=["test"], verbose=False
        )
        dataloaders[iso_code] = data_loader["test"]

        iso_dir = os.path.join(exp_dir, iso_code)
        if os.path.exists(iso_dir):
            shutil.rmtree(iso_dir)
        os.makedirs(iso_dir)

        model_configs = model_utils.get_ensemble_configs(iso_code, config)
        for iso_code_test in config:
            results_name = f"{iso_code_test}_ensemble"
            results_dir = os.path.join(iso_dir, results_name)
            os.makedirs(results_dir)

            print(f"Test country: {iso_code_test}")
            probs = 0
            for model_config in model_configs:
                print(f"Model: {iso_code} {model_config['model']}")
                model = pred_utils.load_model(iso_code, config=model_config)
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=model_config["label_smoothing"]
                )
                model.eval()

                model_config["iso_codes"] = [iso_code_test]
                results, _, preds = cnn_utils.evaluate(
                    dataloaders[iso_code_test],
                    model=model,
                    criterion=criterion,
                    device=device,
                    pos_label=1,
                    class_names=[1, 0],
                    beta=2,
                    phase="test",
                    wandb=None,
                    logging=logging,
                )
                dataset = model_utils.load_data(
                    config=model_config, attributes=["rurban", "iso"], verbose=False
                )
                dataset = dataset[dataset.dataset == "test"]
                preds = pd.merge(dataset, preds, on="UID", how="inner")
                results_name = f"{iso_code_test}_{model_config['model']}"
                preds.to_csv(
                    os.path.join(results_dir, f"{results_name}.csv"), index=False
                )
                probs = probs + preds["y_probs"].to_numpy()

            preds["y_probs"] = probs / len(model_configs)
            preds.to_csv(os.path.join(results_dir, f"{results_name}.csv"))

            pred_results = eval_utils.save_results(
                preds,
                target="y_true",
                pred="y_preds",
                prob="y_probs",
                pos_class=1,
                classes=[1, 0],
                results_dir=results_dir,
                prefix="test",
            )
            print(pred_results["test_ap"], pred_results["test_auprc"])


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument(
        "--project", default="GIGAv1", help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.config)
    config = config_utils.create_config(config_file)
    main(config, args.project)
