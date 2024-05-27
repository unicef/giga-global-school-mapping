import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

import post_utils
import pred_utils
import cnn_utils
import model_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_results(
    model, 
    iso_code, 
    config, 
    exp_dir,
    calibration,
    beta=2, 
):
    if calibration == "tempscaling":
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        _, data_loader, _ = cnn_utils.load_dataset(
            config, phases=["train", "val", "test"], verbose=False
        )
        for phase in ['val', 'test']:   
            _, _, preds = cnn_utils.evaluate(
                data_loader[phase],  
                model=model, 
                criterion=criterion, 
                device=device, 
                pos_label=1,
                class_names=[1, 0],
                beta=beta,
                phase=phase,
                logging=logging
            )
                
            dataset = model_utils.load_data(
                config=config, attributes=["rurban", "iso"], verbose=False
            )
            dataset = dataset[dataset.dataset == phase]
            preds = pd.merge(dataset, preds, on='UID', how='inner')
            filename = f"{iso_code}_{config['config_name']}_{phase}_{calibration}.csv"
            preds.to_csv(os.path.join(exp_dir, filename), index=False)

    elif calibration == "isoreg":
        for phase in ['val', 'test']:
            output = post_utils.get_results(iso_code, config, phase=phase)
            preds = model.predict(output["y_probs"])
            output["y_probs"] = preds
            filename = f"{iso_code}_{config['config_name']}_{phase}_{calibration}.csv"
            output.to_csv(os.path.join(exp_dir, filename), index=False)
        

def temperature_check(iso_code, config, exp_dir, lr=0.01, save_model=False):
    calibration = "tempscaling"
    cwd = os.path.dirname(os.getcwd())
    config["iso_codes"] = [iso_code]

    classes = {1: config["pos_class"], 0: config["neg_class"]}
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = pred_utils.load_cnn(
        config, 
        classes, 
        model_file,  
        temp_lr=lr, 
        verbose=True,
        calibration=calibration,
        save=save_model
    ).eval()
                
    return model