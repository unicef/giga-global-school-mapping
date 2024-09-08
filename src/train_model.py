import os
import shutil
import time
import argparse
import logging
import pandas as pd

import torch
import wandb
from utils import config_utils
from utils import cnn_utils
from utils import eval_utils
from utils import model_utils


# Get device
cwd = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")


def main(c, wandb):
    # Create experiment folder
    exp_name = f"{c['iso_code']}_{c['config_name']}"
    if c["pretrained"]:
        exp_name = f"{c['iso_code']}_{c['config_name']}_{c['pretrained']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], c["project"], exp_name)
    logging.info(f"Experiment directory: {exp_dir}")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Initialize logging
    logname = os.path.join(exp_dir, f"{exp_name}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logging.info(exp_name)
    wandb.run.name = exp_name

    # Load dataset
    phases = ["train", "val", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases)
    logging.info(
        f"Train/val/test sizes: {len(data['train'])}/{len(data['val'])}/{len(data['test'])}"
    )
    wandb.log({f"{phase}_size": len(data[phase]) for phase in phases})

    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        label_smoothing=c["label_smoothing"],
        lr=c["lr"],
        patience=c["patience"],
        data_loader=data_loader,
        device=device,
        lr_finder=c["lr_finder"],
        model_file=c["model_file"],
    )
    logging.info(model)

    lr = optimizer.param_groups[0]["lr"]
    logging.info(f"LR: {lr}")
    wandb.log({"lr": lr})

    # Instantiate wandb tracker
    wandb.watch(model)

    # Commence model training
    n_epochs = c["n_epochs"]
    beta = c["beta"]
    scorer = c["scorer"]
    since = time.time()
    best_score = -1
    best_results = None

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
            logging=logging,
        )
        # Evauate model
        val_results, val_cm, val_preds = cnn_utils.evaluate(
            data_loader["val"],
            model=model,
            criterion=criterion,
            device=device,
            pos_label=1,
            class_names=[1, 0],
            beta=beta,
            phase="val",
            wandb=wandb,
            logging=logging,
        )
        scheduler.step(val_results["val_loss"])

        # Save best model so far
        if val_results[f"val_{scorer}"] > best_score:
            best_score = val_results[f"val_{scorer}"]
            best_results = val_results

            eval_utils.save_files(val_results, val_cm, exp_dir)
            model_file = os.path.join(exp_dir, f"{exp_name}.pth")
            torch.save(model.state_dict(), model_file)

        logging.info(f"Best val_{scorer}: {best_score}")
        log_results = {key: val for key, val in best_results.items() if key[-1] != "_"}
        logging.info(f"Best scores: {log_results}")

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < c["lr_min"]:
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
    optim_threshold = best_results["val_optim_threshold"]

    # Calculate test performance using best model
    final_results = {}
    for phase in ["val", "test"]:
        logging.info(f"\n{phase.capitalize()} Results")
        test_results, test_cm, test_preds = cnn_utils.evaluate(
            data_loader[phase],
            model=model,
            criterion=criterion,
            device=device,
            pos_label=1,
            class_names=[1, 0],
            beta=beta,
            phase=phase,
            wandb=wandb,
            optim_threshold=optim_threshold,
            logging=logging,
        )
        final_results.update(test_results)

        dataset = model_utils.load_data(
            config=c, attributes=["rurban", "iso"], verbose=False
        )
        test_dataset = dataset[dataset.dataset == phase]
        test_preds = pd.merge(test_dataset, test_preds, on="UID", how="inner")
        test_preds.to_csv(os.path.join(exp_dir, f"{exp_name}_{phase}.csv"), index=False)
        eval_utils.save_results(
            test_preds,
            target="y_true",
            pred="y_preds",
            prob="y_probs",
            pos_class=1,
            classes=[1, 0],
            optim_threshold=optim_threshold,
            results_dir=os.path.join(exp_dir, phase),
            prefix=phase,
        )

        for rurban in ["urban", "rural"]:
            subtest_preds = test_preds[test_preds.rurban == rurban]
            eval_utils.save_results(
                subtest_preds,
                target="y_true",
                pred="y_preds",
                prob="y_probs",
                pos_class=1,
                classes=[1, 0],
                optim_threshold=optim_threshold,
                results_dir=os.path.join(exp_dir, phase, rurban),
                prefix=f"{phase}_{rurban}",
            )

    return final_results


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument(
        "--lr_finder", help="Learning rate finder (boolean indicator)", default=None
    )
    parser.add_argument("--pretrained", help="Pretrained model file", default=None)
    parser.add_argument("--iso", help="ISO 3166-1 alpha-3 code", default=[], nargs="+")
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.config)
    c = config_utils.load_config(config_file)
    if "iso_codes" not in c:
        c["iso_codes"] = args.iso
        iso = args.iso[0]
    if "name" in c:
        iso = c["name"]
    c["iso_code"] = iso

    if args.lr_finder:
        args.lr_finder = bool(eval(args.lr_finder))
        c["lr_finder"] = args.lr_finder

    c["model_file"] = None
    c["pretrained"] = None
    if args.pretrained:
        model_file = os.path.join(
            os.getcwd(),
            c["exp_dir"],
            c["project"],
            f"{args.pretrained}_{c['config_name']}",
            f"{args.pretrained}_{c['config_name']}.pth",
        )
        c["pretrained"] = args.pretrained
        c["model_file"] = model_file

    log_c = {
        key: val
        for key, val in c.items()
        if ("url" not in key)
        and ("dir" not in key)
        and ("file" not in key)
        and ("school" not in key)
        and ("exclude" not in key)
        and ("ms_dict" not in key)
    }
    logging.info(log_c)

    # Set wandb configs
    wandb.init(project=c["project"], config=log_c)

    main(c, wandb)
