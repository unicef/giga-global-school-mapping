import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch_lr_finder import LRFinder

from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ViT_H_14_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)
from torch.utils.data import DataLoader
import torch.nn.functional as nnf
import satlaspretrain_models

from utils import eval_utils
from utils import data_utils
from utils import model_utils

# Add temporary fix for hash error:
# https://github.com/pytorch/vision/issues/7744
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from typing import Union, Any, Tuple, Dict, Optional, Callable

import logging

SEED = 42
logging.basicConfig(level=logging.INFO)
ImageFile.LOAD_TRUNCATED_IMAGES = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SchoolDataset(Dataset):
    """
    A custom dataset class for handling school images.

    Attributes:
        dataset (DataFrame): DataFrame containing dataset information.
        classes (dict): Dictionary mapping class labels to numerical values.
        transform (callable, optional): Transformation to be applied on the images.
        normalize (str, optional): Normalization method. Defaults to "imagenet".
        return_uid (bool, optional): Flag to determine if UID should be returned. Defaults to True.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        classes: dict,
        transform: Callable = None,
        normalize: str = "imagenet",
        return_uid: bool = True,
    ):
        """
        Initializes the SchoolDataset instance.

        Args:
            dataset (DataFrame): DataFrame containing dataset information.
            classes (dict): Dictionary mapping class labels to numerical values.
            transform (callable, optional): Transformation to be applied on the images. Defaults to None.
            normalize (str, optional): Normalization method. Defaults to "imagenet".
            return_uid (bool, optional): Flag to determine if UID should be returned. Defaults to True.
        """
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.normalize = normalize
        self.return_uid = return_uid

    def set_return_uid(self, return_uid: bool):
        """
        Sets the flag to determine if UID should be returned.

        Args:
            return_uid (bool): Flag to determine if UID should be returned.
        """
        self.return_uid = return_uid

    def __getitem__(self, index: int):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image tensor, class label,
                and optionally the UID.
        """
        # Get the row at the specified index
        item = self.dataset.iloc[index]
        # Extract UID
        uid = item["UID"]
        # Extract file path to the image
        filepath = item["filepath"]

        # Open the image and convert to RGB
        if os.path.exists(filepath):
            while True:
                try:
                    image = Image.open(filepath).convert("RGB")
                except:
                    time.sleep(1)
                else:
                    break

        # Apply transformations if any
        if self.transform:
            x = self.transform(image)

        # Get the class label
        y = self.classes[item["class"]]
        image.close()

        # Return image tensor, label, and UID
        if self.return_uid:
            return x, y, uid
        else:
            return x, y

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)


def visualize_data(
    data: dict,
    data_loader: dict,
    phase: str = "test",
    n: int = 4,
    normalize: str = "imagenet",
) -> None:
    """
    Visualizes a grid of images from the data loader.

    Args:
        data (dict): Dictionary containing dataset objects for different phases.
        data_loader (dict): Dictionary containing data loaders for different phases.
        phase (str, optional): Phase of the data loader to visualize. Defaults to "test".
        n (int, optional): Number of images per row and column in the grid. Defaults to 4.
        normalize (str, optional): Normalization method. Defaults to "imagenet".

    Returns:
        None
    """
    # Get a batch of inputs, classes, and UIDs from the data loader
    inputs, classes, uids = next(iter(data_loader[phase]))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n, n, figsize=(6, 6), dpi=300)

    # Extract class labels and values
    key_list = list(data[phase].classes.keys())
    val_list = list(data[phase].classes.values())

    for i in range(n):
        for j in range(n):
            # Get the image and transpose it to (H, W, C)
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            # Get the title for the image based on its class
            title = key_list[val_list.index(classes[i * n + j])]
            # Normalize the image if required
            if normalize == "imagenet":
                image = np.clip(
                    np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1
                )
            # Display the image in the subplot
            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")

    # Display the figure
    plt.show()


def load_dataset(config: dict, phases: list, verbose: bool = True) -> tuple:
    """
    Loads the dataset and prepares data loaders for specified phases.

    Args:
        config (dict): Configuration dictionary containing various settings.
            - "pos_class" (str): Positive class label.
            - "neg_class" (str): Negative class label.
            - "normalize" (str): Normalization method.
            - "img_size" (int): Size of the images.
            - "batch_size" (int): Batch size for the data loader.
            - "n_workers" (int): Number of workers for the data loader.
        phases (list): List of dataset phases to load (e.g., ["train", "val", "test"]).
        verbose (bool, optional): If True, prints additional information. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - data (dict): Dictionary of SchoolDataset objects for each phase.
            - data_loader (dict): Dictionary of DataLoader objects for each phase.
            - classes (list): List of unique class labels.
    """
    # Load the dataset with specified attributes and verbose option
    dataset = model_utils.load_data(
        config, attributes=["rurban", "iso"], verbose=verbose
    )

    # Generate file paths for the images in the dataset
    dataset["filepath"] = data_utils.get_image_filepaths(config, dataset)

    # Create a dictionary for class labels
    classes_dict = {config["pos_class"]: 1, config["neg_class"]: 0}

    # Get normalization method and image transformations
    transforms = get_transforms(size=config["img_size"], normalize=config["normalize"])

    # List unique class labels in the dataset
    classes = list(dataset["class"].unique())
    if verbose:
        logging.info(f" Classes: {classes}")

    # Create a dictionary of SchoolDataset objects for each phase
    data = {
        phase: SchoolDataset(
            dataset[dataset.dataset == phase]
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True),
            classes_dict,
            transforms[phase],
            normalize=config["normalize"],
        )
        for phase in phases
    }

    # Create a dictionary of DataLoader objects for each phase
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
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logging: Any,
    pos_label: int,
    beta: float,
    optim_threshold: Optional[float] = None,
    wandb: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Train the model for one epoch.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.
        logging (Any): Logging object for logging training progress.
        pos_label (int): Positive class label.
        beta (float): Weight of precision in the F-beta score.
        optim_threshold (float, optional): Threshold for optimizing predictions. Defaults to None.
        wandb (Any, optional): Weights & Biases object for logging. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing training loss and evaluation metrics.
    """
    # Set the model to training mode
    model.train()

    # Initialize lists to store actual labels, predicted labels, and prediction probabilities
    y_actuals, y_preds, y_probs = [], [], []
    running_loss = 0.0

    # Iterate over batches of data from the data loader
    for inputs, labels, _ in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)  # Move inputs to the specified device
        labels = labels.to(device)  # Move labels to the specified device

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            soft_outputs = nnf.softmax(outputs, dim=1)
            probs = soft_outputs[:, 1]
            loss = criterion(outputs, labels)

            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters

            running_loss += loss.item() * inputs.size(0)
            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())
            y_probs.extend(probs.data.cpu().numpy().tolist())

    # Calculate epoch loss
    epoch_loss = running_loss / len(data_loader)

    # Evaluate the model's performance
    epoch_results = eval_utils.evaluate(
        y_actuals,
        y_preds,
        y_probs,
        pos_label,
        beta=beta,
        optim_threshold=optim_threshold,
    )

    # Add loss to the results
    epoch_results["loss"] = epoch_loss
    epoch_results = {f"train_{key}": val for key, val in epoch_results.items()}

    # Get the current learning rate
    learning_rate = optimizer.param_groups[0]["lr"]

    # Log the results
    log_results = {key: val for key, val in epoch_results.items() if key[-1] != "_"}
    logging.info(f"Train: {log_results} LR: {learning_rate}")

    # Log results to Weights & Biases if available
    if wandb is not None:
        wandb.log(log_results)
    return epoch_results


def evaluate(
    data_loader: DataLoader,
    class_names: list,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    logging: Any,
    pos_label: int,
    beta: float,
    phase: str,
    optim_threshold: Optional[float] = None,
    wandb: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Tuple[torch.Tensor, Dict[str, Any], str], pd.DataFrame]:
    """
    Evaluate the model on the given data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        class_names (list): List of class names.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        logging (Any): Logging object for logging evaluation progress.
        pos_label (int): Positive class label.
        beta (float): Weight of precision in the F-beta score.
        phase (str): Phase of the evaluation (e.g., "test", "validation").
        optim_threshold (float, optional): Threshold for optimizing predictions. Defaults to None.
        wandb (Any, optional): Weights & Biases object for logging. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Tuple[torch.Tensor, Dict[str, Any], str], pd.DataFrame]:
            Evaluation results, confusion matrix, and predictions DataFrame.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store UIDs, actual labels, predicted labels, and prediction probabilities
    y_uids, y_actuals, y_preds, y_probs = [], [], [], []
    running_loss = 0.0  # Initialize running loss
    confusion_matrix = torch.zeros(
        len(class_names), len(class_names)
    )  # Initialize confusion matrix

    # Iterate over batches of data from the data loader
    for inputs, labels, uids in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)  # Move inputs to the specified device
        labels = labels.to(device)  # Move labels to the specified device

        # Disable gradient calculation for evaluation
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            soft_outputs = nnf.softmax(outputs, dim=1)
            probs = soft_outputs[:, 1]
            loss = criterion(outputs, labels)

        # Update running loss
        running_loss += loss.item() * inputs.size(0)
        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.data.cpu().numpy().tolist())
        y_probs.extend(probs.data.cpu().numpy().tolist())
        y_uids.extend(uids)

    # Calculate epoch loss
    epoch_loss = running_loss / len(data_loader)

    # Evaluate the model's performance
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

    # Generate confusion matrix and related metrics
    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    preds = pd.DataFrame(
        {"UID": y_uids, "y_true": y_actuals, "y_preds": y_preds, "y_probs": y_probs}
    )

    # Log the results
    log_results = {key: val for key, val in epoch_results.items() if key[-1] != "_"}
    logging.info(f"{phase.capitalize()} Loss: {epoch_loss} {log_results}")

    # Log results to Weights & Biases if available
    if wandb is not None:
        wandb.log(log_results)

    return epoch_results, (confusion_matrix, cm_metrics, cm_report), preds


def get_transforms(
    size: Union[int, tuple], normalize: str = "imagenet"
) -> Dict[str, transforms.Compose]:
    """
    Get data transformations for training, validation, and testing.

    Args:
        size (Union[int, tuple]): Size for the image transformations.
        normalize (str, optional): Normalization type. Defaults to "imagenet".

    Returns:
        Dict[str, transforms.Compose]: Dictionary containing transformations for 'train', 'val', and 'test'.
    """
    # Define transformations for different phases
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

    # Add normalization if specified
    if normalize == "imagenet":
        for k, v in transformations.items():
            transformations[k].append(transforms.Normalize(imagenet_mean, imagenet_std))

    # Compose the transformations
    transformations = {k: transforms.Compose(v) for k, v in transformations.items()}
    return transformations


def get_model(model_type: str, n_classes: int) -> nn.Module:
    """
    Get a pretrained model with the specified architecture and modify the
    final layer to match the number of classes.

    Args:
        model_type (str): The type of model to load (e.g., "resnet18", "inception_v3").
        n_classes (int): The number of output classes for the final layer.

    Returns:
        nn.Module: The modified model with the final layer adjusted for the specified number of classes.
    """

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

    elif "inception" in model_type:
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    elif "vgg" in model_type:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)

    elif "efficientnet" in model_type:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)

    elif "xception" in model_type:
        model = timm.create_model("xception", pretrained=True, num_classes=n_classes)

    elif "convnext" in model_type:
        if "small" in model_type:
            model = models.convnext_small(weights="IMAGENET1K_V1")
        elif "base" in model_type:
            model = models.convnext_base(weights="IMAGENET1K_V1")
        elif "large" in model_type:
            model = models.convnext_large(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, n_classes)

    elif "satlas" in model_type:
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

    elif "vit" in model_type:
        if model_type == "vit_b_16":
            model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
        elif model_type == "vit_l_16":
            model = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
        elif model_type == "vit_h_14":
            model = models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)

    elif "swin" in model_type:
        if model_type == "swin_v2_t":
            model = models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
            model.head = nn.Linear(model.head.in_features, n_classes)
        elif model_type == "swin_v2_s":
            model = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
            model.head = nn.Linear(model.head.in_features, n_classes)
        elif model_type == "swin_v2_b":
            model = models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            model.head = nn.Linear(model.head.in_features, n_classes)

    return model


def load_model(
    model_type: str,
    n_classes: int,
    pretrained: str,
    scheduler_type: str,
    optimizer_type: str,
    data_loader: Optional[DataLoader] = None,
    label_smoothing: float = 0.0,
    lr: float = 0.001,
    patience: int = 7,
    device: str = "cpu",
    start_lr: float = 1e-6,
    end_lr: float = 1e-3,
    num_iter: int = 1000,
    lr_finder: bool = True,
    model_file: str = None,
):
    """
    Load a model, set up the optimizer, loss function, and learning rate scheduler.

    Args:
        model_type (str): The type of model to load (e.g., "resnet18", "inception_v3").
        n_classes (int): The number of output classes for the final layer.
        pretrained (bool): If True, use pretrained weights.
        scheduler_type (str): The type of scheduler to use ("StepLR" or "ReduceLROnPlateau").
        optimizer_type (str): The type of optimizer to use ("SGD" or "Adam").
        data_loader (Optional[DataLoader], optional): DataLoader for the training data.
            Required if lr_finder is True. Defaults to None.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
        lr (float, optional): Learning rate. Defaults to 0.001.
        momentum (float, optional): Momentum for the SGD optimizer. Defaults to 0.9.
        patience (int, optional): Number of epochs with no improvement for ReduceLROnPlateau. Defaults to 7.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Defaults to "cpu".
        start_lr (float, optional): Initial learning rate for learning rate finder. Defaults to 1e-6.
        end_lr (float, optional): Maximum learning rate for learning rate finder. Defaults to 1e-3.
        num_iter (int, optional): Number of iterations for learning rate finder. Defaults to 1000.
        lr_finder (bool, optional): If True, use learning rate finder. Defaults to True.

    Returns:
        Tuple[nn.Module, nn.CrossEntropyLoss, optim.Optimizer,
        Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]]:
            - model (nn.Module): The loaded model.
            - criterion (nn.CrossEntropyLoss): The loss function.
            - optimizer (optim.Optimizer): The optimizer.
            - scheduler (Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]):
                The learning rate scheduler.
    """
    # Get the model based on the specified type and number of classes
    model = get_model(model_type, n_classes)
    model = nn.DataParallel(model)  # Wrap the model for multi-GPU training
    if model_file:
        logging.info(f"Loading {model_file}...")
        model.load_state_dict(torch.load(model_file, map_location=device))
        logging.info(f"{model_file} loaded")
    model = model.to(device)  # Move the model to the specified device

    # Define the loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Set up the optimizer based on the specified type
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optionally find the optimal learning rate
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

    # Set up the learning rate scheduler based on the specified type
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=patience, mode="min"
    )
    return model, criterion, optimizer, scheduler


def run_lr_finder(
    data_loader: Dict[str, DataLoader],
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    start_lr: float,
    end_lr: float,
    num_iter: int,
    plot: bool = False,
) -> float:
    """
    Run a learning rate finder to determine the optimal learning rate.

    Args:
        data_loader (Dict[str, DataLoader]): Dictionary containing DataLoaders for different phases.
        model (nn.Module): The model to test.
        optimizer (optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        device (str): Device to run the model on ("cpu" or "cuda").
        start_lr (float): Starting learning rate for the range test.
        end_lr (float): Ending learning rate for the range test.
        num_iter (int): Number of iterations to run the range test.
        plot (bool, optional): If True, plot the learning rate vs. loss. Defaults to False.

    Returns:
        float: The best learning rate found.
    """
    # Initialize the learning rate finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # Perform the range test
    lr_finder.range_test(
        data_loader["val"],  # Use validation data loader
        start_lr=start_lr,  # Starting learning rate
        end_lr=end_lr,  # Ending learning rate
        num_iter=num_iter,  # Number of iterations
        step_mode="exp",  # Use exponential step mode
    )

    # Optionally plot the learning rate vs. loss
    if plot:
        lr_finder.plot()

    # Extract learning rates and corresponding losses from the history
    lrs = np.array(lr_finder.history["lr"])
    losses = np.array(lr_finder.history["loss"])

    # Calculate the gradient of the loss curve
    min_grad_idx = (np.gradient(np.array(losses))).argmin()

    # Determine the best learning rate as the one where the gradient is minimal
    best_lr = start_lr
    if min_grad_idx is not None:
        best_lr = lrs[min_grad_idx]

    # Log the best learning rate found
    logging.info(f"Best lr: {best_lr}")

    # Reset the model and optimizer to their initial states
    lr_finder.reset()
    return best_lr
