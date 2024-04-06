import os
import time
import argparse
import pandas as pd
from collections import Counter
import torch

import sys
sys.path.insert(0, "../utils/")
import config_utils
import cnn_utils
import eval_utils
import model_utils
import wandb
import logging

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")


def main(c):    
    # Create experiment folder
    exp_name = f"{c['iso_code']}_{c['config_name']}"
    exp_dir = os.path.join(cwd,  c["exp_dir"], c["project"], exp_name)
    logging.info(f"Experiment directory: {exp_dir}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    logname = os.path.join(exp_dir, f"{exp_name}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logging.info(exp_name)
    wandb.run.name = exp_name
    
    # Load dataset
    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases)
    logging.info(f"Train/test sizes: {len(data['train'])}/{len(data['test'])}")

    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        label_smoothing=c["label_smoothing"],
        lr=c["lr"],
        momentum=c["momentum"],
        gamma=c["gamma"],
        step_size=c["step_size"],
        patience=c["patience"],
        dropout=c["dropout"],
        device=device,
    )
    logging.info(model)

    # Instantiate wandb tracker
    wandb.watch(model)

    # Commence model training
    n_epochs = c["n_epochs"]
    beta = c["beta"]
    since = time.time()
    best_score = -1

    for epoch in range(1, n_epochs + 1):
        logging.info("\nEpoch {}/{}".format(epoch, n_epochs))

        # Train model
        cnn_utils.train(
            data_loader["train"],
            model,
            criterion,
            optimizer,
            device,
            pos_label=1,
            beta=beta,
            wandb=wandb,
            logging=logging
        )
        # Evauate model
        val_results, val_cm, val_preds = cnn_utils.evaluate(
            data_loader["test"], 
            classes=[1, 0], 
            model, 
            criterion, 
            device, 
            pos_label=1,
            beta=beta,
            wandb=wandb, 
            logging=logging
        )
        scheduler.step(val_results[f"fbeta_score_{beta}"])

        # Save best model so far
        if val_results[f"fbeta_score_{beta}"] > best_score:
            best_score = val_results[f"fbeta_score_{beta}"]
            precision = val_results[f"precision_score"]
            recall = val_results[f"recall_score"]
            best_weights = model.state_dict()

            eval_utils._save_files(val_results, val_cm, exp_dir)
            model_file = os.path.join(exp_dir, f"{exp_name}.pth")
            torch.save(model.state_dict(), model_file)
            
        logging.info(f"Best Fbeta score: {best_score}")
        logging.info(f"Best Precision: {precision}")
        logging.info(f"Best Recall: {recall}")

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    # Terminate trackers
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best model
    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    # Calculate test performance using best model
    logging.info("\nTest Results")
    test_results, test_cm, test_preds = cnn_utils.evaluate(
        data_loader["test"], 
        classes, 
        model, 
        criterion, 
        device, 
        pos_label=1, 
        beta=beta, 
        wandb=wandb, 
        logging=logging
    )
    dataset = model_utils.load_data(config=c, attributes=["rurban", "iso"], verbose=False)
    dataset = dataset[dataset.dataset == "test"]
    test_preds = pd.merge(dataset, test_preds, on='UID', how='inner')
    test_preds.to_csv(os.path.join(exp_dir, f"{exp_name}.csv"), index=False)

    # Save results in experiment directory
    eval_utils.save_results(
        test_preds, 
        target="y_true", 
        pred="y_preds", 
        pos_class=1, 
        classes=[1, 0], 
        results_dir=exp_dir
    )

    for rurban in ["urban", "rural"]:
        subresults_dir = os.path.join(exp_dir, rurban)
        subtest_preds = test_preds[test_preds.rurban == rurban]
        results = eval_utils.save_results(
            subtest_preds, 
            target="y_true",  
            pred="y_preds", 
            pos_class=1, 
            classes=[1, 0], 
            subresults_dir=subresults_dir, 
            rurban=rurban
        )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--cnn_config", help="Config file")
    parser.add_argument("--iso", help="ISO code", default=[], nargs='+')
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.cnn_config)
    c = config_utils.load_config(config_file)
    c["iso_codes"] = args.iso
    iso = args.iso[0]
    if "name" in c: iso = c["name"]
    c["iso_code"] = iso
    log_c = {
        key: val for key, val in c.items() 
        if ('url' not in key) 
        and ('dir' not in key)
        and ('file' not in key)
    }
    logging.info(log_c)

    # Set wandb configs
    wandb.init(project=c["project"], config=log_c)

    main(c)