import os
import logging
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm

from shapely import geometry
import geopandas as gpd
import rasterio as rio
import rasterio.plot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)

import torch
import torchvision
from torchcam.utils import overlay_mask
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from utils import data_utils
from utils import pred_utils
from utils import cnn_utils

from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    GradCAMElementWise,
    RandomCAM,
)
import PIL

import cv2
import skimage.feature

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


cams = {
    "randomcam": RandomCAM,
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "gradcamplusplus": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "gradcamelementwise": GradCAMElementWise,
}


def get_best_cam_method(iso_code: str, model_config: dict) -> str:
    """
    Determines the best Class Activation Map (CAM) method for a specified model 
    configuration and country, based on the minimum score in the CAM results file.

    Args:
        iso_code (str): The ISO code for the country of interest.
        model_config (dict): Configuration dictionary for the model, which includes:
            - model (str): The name of the model.
            - project (str): The name of the project folder where results are stored.

    Returns:
        str: The name of the CAM method with the lowest score, indicating the 
             best-performing method for the specified model and country.

    Raises:
        FileNotFoundError: If the CAM results file does not exist in the specified path.
    """
    model_name = model_config["model"]
    cam_results_file = os.path.join(
        os.getcwd(),
        "exp",
        model_config["project"],
        f"{iso_code}_{model_name}",
        "cam_results.csv"
    )
    if os.path.exists(cam_results_file):
        cam_results = pd.read_csv(cam_results_file)
        cam_results["method"] = cam_results["method"].replace(
            {"gradcam++" : "gradcamplusplus"}
        )
        cam_method = cam_results[
            cam_results["score"] == cam_results["score"].min()
        ]["method"].values[0]
        print(f"Best cam method: {cam_method}")
    else:
        raise FileNotFoundError(f"CAM results file not found at {cam_results_file}")
    return cam_method


def get_cam_extractor(config: dict, model: torch.nn.Module, cam_extractor):
    """
    Retrieves the appropriate Class Activation Map (CAM) extractor based on the model configuration.

    Args:
        config (dict): Configuration dictionary containing model details.
        model (torch.nn.Module): The neural network model from which CAMs will be extracted.
        cam_extractor (callable): A callable that initializes the CAM extractor with the given parameters.

    Returns:
        cam_extractor: An instance of the CAM extractor configured with the
            appropriate target layers and reshape transformation.
    """
    reshape_transform = None
    cam_extractor = cams[cam_extractor]

    class ReshapeTransform:
        def __init__(self, model_type="vit", height=16, width=16):
            """
            Initializes the ReshapeTransform class with model type and dimensions.

            Args:
                model_type (str): The type of model (e.g., "vit").
                height (int): The height of the transformed tensor.
                width (int): The width of the transformed tensor.
            """
            self.model_type = model_type
            self.height = height
            self.width = width

        def reshape_transform(self, tensor):
            """
            Reshapes the input tensor based on the model type.

            Args:
                tensor (torch.Tensor): The tensor to be reshaped.

            Returns:
                torch.Tensor: The reshaped tensor.
            """
            if "vit" in self.model_type:
                tensor = tensor[:, 1:, :].reshape(
                    tensor.size(0), self.height, self.width, tensor.size(2)
                )
            tensor = tensor.transpose(2, 3).transpose(1, 2)
            return tensor

    # Determine the target layers and reshape transform based on the model type
    if "vgg" in config["model"].lower():
        target_layers = [model.module.features[-1]]

    elif "convnext" in config["model"].lower():
        target_layers = [model.module.features[-1][-1].block[2]]
        reshape_transform = ReshapeTransform(config["model"]).reshape_transform

    elif "vit" in config["model"].lower():
        target_layers = [model.module.encoder.layers[-1].ln_1]
        if config["model"] == "vit_h_14":
            reshape_transform = ReshapeTransform(
                config["model"], 16, 16
            ).reshape_transform
        elif (config["model"] == "vit_l_16") or (config["model"] == "vit_b_16"):
            reshape_transform = ReshapeTransform(
                config["model"], 14, 14
            ).reshape_transform

    elif "swin" in config["model"].lower():
        target_layers = [model.module.features[-1][-1].norm2]
        reshape_transform = ReshapeTransform(config["model"], 14, 14).reshape_transform

    return cam_extractor(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )


def cam_predict(
    iso_code: str,
    config: dict,
    data: gpd.GeoDataFrame,
    geotiff_dir: str,
    shapename: str,
    buffer_size: int = 50,
    verbose: bool = False,
) -> gpd.GeoDataFrame:

    """
    Runs the Class Activation Map (CAM) prediction process for a given region, model configuration, 
    and dataset, and saves the results as a GeoJSON file. If results already exist, 
    loads them directly.

    Args:
        iso_code (str): The ISO code representing the country.
        config (dict): Configuration dictionary for the model, including:
            - model (str): The model name used for CAM predictions.
            - project (str): The project directory where output is stored.
            - config_name (str): The configuration name used for naming output files.
        data (gpd.GeoDataFrame): The geospatial data to be processed.
        geotiff_dir (str): Directory containing GeoTIFF files needed for model predictions.
        shapename (str): Identifier for the shapefile, used in naming the output file.
        buffer_size (int, optional): Size of the buffer around points for CAM extraction. Default is 50.
        verbose (bool, optional): Flag for verbosity in the prediction process. Default is False.

    Returns:
        gpd.GeoDataFrame: The results of the CAM prediction as a GeoDataFrame.
    """
    # Determine the best CAM method for the given iso_code and model config
    cam_method = get_best_cam_method(iso_code, config)

    # Create the output directory for storing results if it doesn't already exist
    out_dir = data_utils.makedir(
        os.path.join(
            os.getcwd(),
            "output",
            iso_code,
            "results",
            config["project"],
            "cams",
            "ensemble",
            config["config_name"],
            cam_method,
        )
    )

    # Define the path to the output file
    out_file = os.path.join(
        out_dir, f"{iso_code}_{shapename}_{config['config_name']}_{cam_method}.geojson"
    )

    # If the output file already exists, load and return it
    if os.path.exists(out_file):
        return gpd.read_file(out_file)
    
    # Load the specified model for the country and configuration
    model = pred_utils.load_model(iso_code, config)
    model.eval() # Set model to evaluation mode for inference

    # Initialize the CAM extractor with the selected CAM method and model
    cam_extractor = get_cam_extractor(config, model, cam_method)

    # Generate CAM points
    results = generate_cam_points(
        data, config, geotiff_dir, model, cam_extractor, buffer_size
    )

    # Assign building pixel sum to CAM points 
    print(f"Filtering buildings for {len(results)}")
    temp_file = os.path.join(
        out_dir, f"{iso_code}_{shapename}_{config['config_name']}_{cam_method}_temp.geojson"
    )
    results.to_file(temp_file, driver="GeoJSON")
    results = pred_utils.filter_by_buildings(iso_code, config, results, in_vector=temp_file)

    # Save the resulting CAM points to a GeoJSON file
    results.to_file(out_file, driver="GeoJSON")
    
    return results


def generate_cam_points(
    data: pd.DataFrame,
    config: dict,
    in_dir: str,
    model: torch.nn.Module,
    cam_extractor,
    buffer_size: int = 50,
    show: bool = False,
) -> gpd.GeoDataFrame:
    """
    Generates Class Activation Map (CAM) points for each image in the dataset 
    and returns them as a GeoDataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the metadata and information about the images.
        config (dict): Configuration dictionary for image processing.
        in_dir (str): Directory containing the input images.
        model (torch.nn.Module): The neural network model used for CAM extraction.
        cam_extractor (callable): Function or class used for extracting CAMs.
        buffer_size (int, optional): Buffer size around each CAM point in meters. Defaults to 50.
        show (bool, optional): If True, displays the CAM points on top of the raster images. 
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the CAM points with their probabilities and UIDs.
    """
    # List to store the generated CAM points
    results = []

    # Reset index of the input DataFrame
    data = data.reset_index(drop=True)

    # Get filepaths for the images
    filepaths = data_utils.get_image_filepaths(config, data, in_dir, ext=".tif")

    # Store the coordinate reference system (CRS) of the input data
    crs = data.crs

    # Iterate over each row in the DataFrame
    print(f"Generating CAM points for {len(data)}")
    for index in tqdm(list(data.index), total=len(data)):
        # Generate CAM for the current image
        if os.path.exists(filepaths[index]):
            _, point, _ = generate_cam(
                config, filepaths[index], model, cam_extractor, metrics=False, show=False
            )
            # Open the image file and extract coordinates for the CAM point
            with rio.open(filepaths[index]) as map_layer:
                coord = [map_layer.xy(point[1], point[0])]
                coord = geometry.Point(coord)
                crs = map_layer.crs
                results.append(coord)

                # Optionally show the CAM point on the image
                if show:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    rasterio.plot.show(map_layer, ax=ax)
                    geom = gpd.GeoDataFrame(geometry=[coord], crs=crs)
                    geom.plot(facecolor="none", edgecolor="blue", ax=ax)

    # Convert the results list to a GeoDataFrame
    results = gpd.GeoDataFrame(geometry=results, crs=crs)

    # Create a square buffer around each point
    results = results.to_crs("EPSG:3857")
    results["geometry"] = results["geometry"].buffer(buffer_size, cap_style=3)
    results = results.to_crs(crs)

    # Assign UID and probabilties
    results["prob"] = data.prob
    results["UID"] = data.UID

    return results


def georeference_images(
    data: pd.DataFrame, config: dict, in_dir: str, out_dir: str
) -> None:
    """
    Georeferences and saves images from the specified directory using the given metadata.

    Args:
        data (pd.DataFrame): DataFrame containing image metadata and geometries.
        config (dict): Configuration dictionary specifying image dimensions and other settings.
        in_dir (str): Directory where input images are located.
        out_dir (str): Directory where georeferenced images will be saved.

    Returns:
        None
    """
    filepaths = data_utils.get_image_filepaths(config, data, in_dir=in_dir)
    data = data.reset_index(drop=True)

    # Iterate over each row in the DataFrame
    for index in tqdm(range(len(data)), total=len(data)):
        # Define the output filename for the georeferenced image
        filename = os.path.join(out_dir, f"{data.iloc[index].UID}.tif")

        # Check if the file already exists to avoid reprocessing
        if not os.path.exists(filename):
            # Open the input image file
            dataset = rio.open(filepaths[index], "r")
            if dataset.read().shape[0] < 3:
                continue

            # Read specific bands from the image
            bands = [1, 2, 3]
            #logging.info(filepaths[index])
            dataset = dataset.read(bands)

            # Get bounding box from the DataFrame for georeferencing
            bounds = data.iloc[index].geometry.bounds

            # Create a transformation matrix from the bounding box
            transform = rio.transform.from_bounds(
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                config["width"],
                config["height"],
            )
            # Define the CRS (Coordinate Reference System)
            crs = {"init": "EPSG:3857"}

            # Write the georeferenced image to the output file
            with rio.open(
                filename,
                "w",
                driver="GTiff",
                width=config["width"],
                height=config["height"],
                count=len(bands),
                dtype=dataset.dtype,
                nodata=0,
                transform=transform,
                crs=crs,
            ) as dst:
                dst.write(dataset, indexes=bands)


def compare_cams(
    iso_code: str,
    config: dict,
    filepaths: list,
    percentile: int = 90,
    show: bool = True,
    metrics: bool = True,
    verbose: bool = False,
) -> tuple:
    """
    Compares different CAM (Class Activation Map) methods on a set of images and computes their scores.

    Args:
        iso_code (str): ISO code for the region or dataset.
        config (dict): Configuration dictionary containing model and CAM settings.
        filepaths (list of str): List of file paths to the images to be processed.
        percentile (int, optional): Percentile for the CAM score threshold. Defaults to 90.
        show (bool, optional): Whether to display CAM visualizations. Defaults to True.
        metrics (bool, optional): Whether to calculate metrics for CAMs. Defaults to True.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.

    Returns:
        tuple: A dictionary of CAM scores for each method, and a dictionary of mean CAM scores for each method.
    """
    # Dictionary to store CAM scores for each method
    cam_scores = dict()

    # Iterate over each CAM method
    for cam_name, cam in cams.items():
        # Load the model and prepare it for evaluation
        model = pred_utils.load_model(iso_code, config, verbose=verbose)
        model.eval()
        model = model.to(device)

        # Initialize a list to store scores for the current CAM method
        cam_scores[cam_name] = []

        # Get the CAM extractor for the current method
        with get_cam_extractor(config, model, cam_name) as cam_extractor:
            cam_extractor.batch_size = config["batch_size"]

            # Create a progress bar for processing the file paths
            pbar = data_utils.create_progress_bar(filepaths) if not show else filepaths

            # Process each file path
            for filepath in pbar:
                if not show:
                    pbar.set_description(f"Processing {cam_name}")

                # Generate CAM for the current image and method
                cam_map, point, score = generate_cam(
                    config,
                    filepath,
                    model,
                    cam_extractor,
                    percentile,
                    title=cam_name,
                    show=show,
                    metrics=metrics,
                )

                # Append the CAM score to the list for the current method
                cam_scores[cam_name].append(score)

    # Calculate mean CAM scores for each method
    cam_scores_mean = dict()
    for cam in cam_scores:
        cam_scores_mean[cam] = np.mean(cam_scores[cam])

    return cam_scores, cam_scores_mean


def compare_random(
    iso_code: str,
    data: pd.DataFrame,
    config: dict,
    filepaths: list,
    percentile: int = 90,
    cam: str = "layercam",
    n_samples: int = 3,
    show: bool = True,
    verbose: bool = False,
) -> None:
    """
    Compares a random sample of images using a specified CAM extractor and generates CAM visualizations.

    Args:
        iso_code (str): ISO code for the specific dataset.
        data (pd.DataFrame): DataFrame containing image metadata, including class labels and other info.
        config (dict): Configuration dictionary containing model settings and other parameters.
        filepaths (list of str): List of file paths to the images.
        percentile (int, optional): Percentile for thresholding the CAM map. Defaults to 90.
        cam (str, optional): The CAM extractor method to use. Defaults to "layercam".
        n_samples (int, optional): Number of random samples to process. Defaults to 3.
        show (bool, optional): Whether to display the CAM visualizations. Defaults to True.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
    """
    if "class" in data.columns:
        # Filter data to include only positive class samples if specified
        data = data[(data["class"] == config["pos_class"])]

    # Load and prepare the model
    model = pred_utils.load_model(iso_code, config, verbose=verbose)
    model.eval()

    # Get the CAM extractor based on the specified method
    cam_extractor = get_cam_extractor(config, model, cam)

    # Process a random sample of images
    for index in list(data.sample(n_samples).index):
        title = str(cam_extractor.__class__.__name__)

        # Generate and visualize the CAM for the sampled image
        generate_cam(
            config,
            filepaths[index],
            model,
            cam_extractor,
            percentile,
            title=title,
            show=show,
        )


def generate_cam(
    config: dict,
    filepath: str,
    model: torch.nn.Module,
    cam_extractor,
    percentile: int = 90,
    metrics: bool = True,
    show: bool = True,
    title: str = "",
    save: str = None,
    figsize: tuple = (9, 9),
) -> tuple:
    """
    Generates a Class Activation Map (CAM) for a given image using a specified CAM extractor.

    Args:
        config (dict): Configuration dictionary containing model and image settings.
        filepath (str): Path to the image file.
        model (torch.nn.Module): The trained model to use for generating CAM.
        cam_extractor (callable): The CAM extractor function or object.
        percentile (int, optional): Percentile for thresholding the CAM. Defaults to 90.
        metrics (bool, optional): Whether to calculate metrics for the CAM. Defaults to True.
        show (bool, optional): Whether to display the CAM results. Defaults to True.
        title (str, optional): Title for the CAM plot. Defaults to an empty string.
        save (str or None, optional): If provided, saves the CAM plot to a file with this name. Defaults to None.
        figsize (tuple, optional): Size of the figure for plotting. Defaults to (9, 9).

    Returns:
        tuple: A tuple containing:
            - cam_map (numpy.ndarray): The generated CAM map.
            - point (tuple): Coordinates of the generated point from the CAM.
            - score (float): The CAM score, if metrics are calculated.
    """
    logger = logging.getLogger()
    logger.disabled = True

    # Load and preprocess the image
    image = Image.open(filepath).convert("RGB")
    transforms = cnn_utils.get_transforms(config["img_size"])
    input_tensor = transforms["test"](image).to(device).unsqueeze(0)
    input_image = input_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0))
    input_image = np.clip(
        np.array(cnn_utils.imagenet_std) * input_image
        + np.array(cnn_utils.imagenet_mean),
        0,
        1,
    )

    # Generate the CAM map
    targets = [ClassifierOutputTarget(1)]
    cam_map = cam_extractor(input_tensor=input_tensor, targets=targets)
    result = show_cam_on_image(input_image, cam_map[0, :], use_rgb=True)
    point = generate_point_from_cam(config, cam_map[0, :], image)

    score = 0
    if metrics:
        # Calculate CAM metrics
        cam_metric = ROADMostRelevantFirst(percentile=percentile)
        scores, road_visualizations = cam_metric(
            input_tensor, cam_map, targets, model, return_visualization=True
        )
        score = scores[0]

        # Process the road visualization
        road_visualization = road_visualizations[0].cpu().numpy().transpose((1, 2, 0))
        road_visualization = deprocess_image(road_visualization)
        edges = skimage.feature.canny(
            image=cv2.cvtColor(road_visualization, cv2.COLOR_BGR2GRAY), sigma=3
        )
        if sum(edges.flatten()) == 0:
            score = 0

    if show:
        # Display the CAM results
        road_visualization = Image.fromarray(road_visualization)
        thresh_cam = cam_map < np.percentile(cam_map, percentile)
        thresh_cam = thresh_cam.transpose((0, 1, 2))[0, :, :]

        n_axis = 3
        fig, ax = plt.subplots(1, n_axis, figsize=figsize, dpi=300)
        ax[0].imshow(image)
        ax[1].imshow(result)

        rect = patches.Rectangle(
            (point[0] - 75, point[1] - 75),
            150,
            150,
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        ax[0].add_patch(rect)

        ax[2].imshow(road_visualization)
        ax[2].text(5, 20, f"ROAD: {score:.3f}", size=10, color="white")
        ax[1].title.set_text(title)

        for i in range(n_axis):
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
        plt.show()

        if save:
            fig.savefig(f"assets/{save}.pdf", bbox_inches="tight")

    return cam_map, point, score


def generate_point_from_cam(
    config: dict, cam_map: np.ndarray, image: PIL.Image
) -> tuple:
    """
    Generates a point of interest from the Class Activation Map (CAM)
        by finding the location of the highest activation.

    Args:
        config (dict): Configuration dictionary containing model and image settings.
        cam_map (torch.Tensor or np.ndarray): The CAM map as a tensor or NumPy array.
        image (PIL.Image): The input image used for CAM generation.

    Returns:
        tuple: The (x, y) coordinates of the point with the highest activation in the CAM map.
    """
    if torch.is_tensor(cam_map):
        cam_map = np.array(cam_map.cpu())

    # Convert CAM map to tensor
    ten_map = torch.tensor(cam_map)
    transform = transforms.Resize(size=(image.size[0], image.size[1]), antialias=None)
    ten_map = transform(ten_map.unsqueeze(0)).squeeze()

    values = []
    for i in range(0, ten_map.shape[0]):
        # Find the maximum value and its index in each channel
        index, value = max(enumerate(ten_map[i]), key=operator.itemgetter(1))
        values.append(value)

    # Find the channel with the maximum activation
    y_index, y_value = max(enumerate(values), key=operator.itemgetter(1))
    x_index, x_value = max(enumerate(ten_map[y_index]), key=operator.itemgetter(1))

    # Return the coordinates of the maximum activation point
    return (x_index, y_index)
