import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import rasterio as rio
import pandas as pd
import numpy as np

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch_lr_finder import LRFinder
from torchgeo.models import ResNet50_Weights

from torchvision import models, transforms
import torchvision.transforms.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
)
import torch.nn.functional as nnf
import satlaspretrain_models

import sys

sys.path.insert(0, "../utils/")
import eval_utils
import clf_utils
import data_utils
import model_utils

SEED = 42

# Add temporary fix for hash error: https://github.com/pytorch/vision/issues/7744
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

import logging

logging.basicConfig(level=logging.INFO)

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SchoolDataset(Dataset):
    def __init__(self, dataset, classes, transform=None, normalize="imagenet"):
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.normalize = normalize

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        uid = item["UID"]
        filepath = item["filepath"]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            x = self.transform(image)

        y = self.classes[item["class"]]
        image.close()
        return x, y, uid

    def __len__(self):
        return len(self.dataset)


def visualize_data(data, data_loader, phase="test", n=4, normalize="imagenet"):
    inputs, classes, uids = next(iter(data_loader[phase]))
    fig, axes = plt.subplots(n, n, figsize=(6, 6))

    key_list = list(data[phase].classes.keys())
    val_list = list(data[phase].classes.values())

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            title = key_list[val_list.index(classes[i * n + j])]
            if normalize == "imagenet":
                image = np.clip(
                    np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1
                )
            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")


def load_dataset(config, phases, verbose=True):
    dataset = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=verbose)
    dataset["filepath"] = data_utils.get_image_filepaths(config, dataset)
    classes_dict = {config["pos_class"]: 1, config["neg_class"]: 0}

    normalize = config["normalize"]
    transforms = get_transforms(size=config["img_size"], normalize=normalize)
    classes = list(dataset["class"].unique())
    logging.info(f" Classes: {classes}")

    data = {
        phase: SchoolDataset(
            dataset[dataset.dataset == phase]
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True),
            classes_dict,
            transforms[phase],
            normalize=normalize,
        )
        for phase in phases
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            data[phase],
            batch_size=config["batch_size"],
            num_workers=config["n_workers"],
            shuffle=True,
            drop_last=True,
        )
        for phase in phases
    }

    return data, data_loader, classes


def train(
    data_loader,
    model,
    criterion,
    optimizer,
    device,
    logging,
    pos_label,
    beta,
    optim_threshold=None,
    wandb=None,
):
    model.train()

    y_actuals, y_preds, y_probs = [], [], []
    running_loss = 0.0
    for inputs, labels, _ in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            soft_outputs = nnf.softmax(outputs, dim=1)
            probs = soft_outputs[:, 1]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())
            y_probs.extend(probs.data.cpu().numpy().tolist())

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(
        y_actuals,
        y_preds,
        y_probs,
        pos_label,
        beta=beta,
        optim_threshold=optim_threshold,
    )
    epoch_results["loss"] = epoch_loss
    epoch_results = {f"train_{key}": val for key, val in epoch_results.items()}

    learning_rate = optimizer.param_groups[0]["lr"]
    log_results = {key: val for key, val in epoch_results.items() if key[-1] != "_"}
    logging.info(f"Train: {log_results} LR: {learning_rate}")

    if wandb is not None:
        wandb.log(log_results)

    return epoch_results


def evaluate(
    data_loader,
    class_names,
    model,
    criterion,
    device,
    logging,
    pos_label,
    beta,
    phase,
    optim_threshold=None,
    wandb=None,
):
    model.eval()

    y_uids, y_actuals, y_preds, y_probs = [], [], [], []
    running_loss = 0.0
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for inputs, labels, uids in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            soft_outputs = nnf.softmax(outputs, dim=1)
            probs = soft_outputs[:, 1]
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.data.cpu().numpy().tolist())
        y_probs.extend(probs.data.cpu().numpy().tolist())
        y_uids.extend(uids)

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(
        y_actuals,
        y_preds,
        y_probs,
        pos_label,
        beta=beta,
        optim_threshold=optim_threshold,
    )
    epoch_results["loss"] = epoch_loss
    epoch_results = {f"{phase}_{key}": val for key, val in epoch_results.items()}

    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    preds = pd.DataFrame(
        {"UID": y_uids, "y_true": y_actuals, "y_preds": y_preds, "y_probs": y_probs}
    )

    log_results = {key: val for key, val in epoch_results.items() if key[-1] != "_"}
    logging.info(f"{phase.capitalize()} Loss: {epoch_loss} {log_results}")
    if wandb is not None:
        wandb.log(log_results)

    return epoch_results, (confusion_matrix, cm_metrics, cm_report), preds


def get_transforms(size, normalize="imagenet"):
    transformations = {
        "train": [
            transforms.Resize(size),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ],
        "val": [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ],
        "test": [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ],
    }

    if normalize == "imagenet":
        for k, v in transformations.items():
            transformations[k].append(transforms.Normalize(imagenet_mean, imagenet_std))

    transformations = {k: transforms.Compose(v) for k, v in transformations.items()}
    return transformations


def get_model(model_type, n_classes, dropout=None):
    if "resnet" in model_type:
        if model_type == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet34":
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_type == "resnet50_fmow_rgb_gassl":
            weights = ResNet50_Weights.FMOW_RGB_GASSL
            model = timm.create_model(
                "resnet50", in_chans=weights.meta["in_chans"], num_classes=n_classes
            )
            model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    if "inception" in model_type:
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    if "vgg" in model_type:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)

    if "efficientnet" in model_type:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)

    if "xception" in model_type:
        model = timm.create_model("xception", pretrained=True, num_classes=n_classes)

    if "convnext" in model_type:
        if "small" in model_type:
            model = models.convnext_small(weights="IMAGENET1K_V1")
        elif "base" in model_type:
            model = models.convnext_base(weights="IMAGENET1K_V1")
        elif "large" in model_type:
            model = models.convnext_large(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, n_classes)

    if "satlas" in model_type:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_manager = satlaspretrain_models.Weights()
        model_identifier = model_type.split("-")[-1]
        model = weights_manager.get_pretrained_model(
            model_identifier=model_identifier,
            num_categories=n_classes,
            fpn=True,
            head=satlaspretrain_models.Head.CLASSIFY,
            device=device,
        )

        class ModelModified(nn.Module):
            def __init__(self, model):
                super(ModelModified, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)[0]

        model = ModelModified(model)

    return model


def load_model(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    data_loader=None,
    label_smoothing=0.0,
    lr=0.001,
    momentum=0.9,
    gamma=0.1,
    step_size=7,
    patience=7,
    dropout=0,
    device="cpu",
    start_lr=1e-6,
    end_lr=1e-3,
    num_iter=1000,
    lr_finder=True,
):
    model = get_model(model_type, n_classes, dropout)
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if lr_finder:
        lr = run_lr_finder(
            data_loader,
            model,
            optimizer,
            criterion,
            device,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
        )
        for param in optimizer.param_groups:
            param["lr"] = lr

    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=patience, mode="min"
        )

    return model, criterion, optimizer, scheduler


def run_lr_finder(
    data_loader,
    model,
    optimizer,
    criterion,
    device,
    start_lr,
    end_lr,
    num_iter,
    plot=False,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        data_loader["val"],
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
    )
    if plot:
        lr_finder.plot()

    lrs = np.array(lr_finder.history["lr"])
    losses = np.array(lr_finder.history["loss"])
    min_grad_idx = None
    min_grad_idx = (np.gradient(np.array(losses))).argmin()
    if min_grad_idx is not None:
        best_lr = lrs[min_grad_idx]

    logging.info(f"Best lr: {best_lr}")
    lr_finder.reset()
    return best_lr
