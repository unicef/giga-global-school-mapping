import os
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import logging
import torch
from PIL import Image
import logging
import joblib
import torch

import cnn_utils
import data_utils
import config_utils
import embed_utils

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as nn
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask

import numpy as np
import operator
from PIL import Image
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms 
import rasterio as rio
from rasterio.mask import mask
from shapely import geometry
import rasterio.plot
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as nnf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

def reshape_transform(tensor):
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result

def cam_predict(iso_code, config, data, geotiff_dir, out_file):
    cwd = os.path.dirname(os.getcwd())
    classes = {1: config["pos_class"], 0: config["neg_class"]}

    out_dir = os.path.join(cwd, "output", iso_code, "results", config["project"])
    out_file = os.path.join(out_dir, out_file)
    if os.path.exists(out_file):
        return gpd.read_file(out_file)
    
    exp_dir = os.path.join(
        cwd, config["exp_dir"], config["project"], f"{iso_code}_{config['config_name']}"
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = load_cnn(config, classes, model_file, verbose=False).eval()

    if config["type"] == "cnn":
        from torchcam.methods import LayerCAM
        cam_extractor = LayerCAM(model)
    elif config["type"] == "vit":
        from pytorch_grad_cam import  LayerCAM
        target_layers = [model.module.model.backbone.backbone.features[-1][-2].norm1]
        cam_extractor = LayerCAM(
            model=model, target_layers=target_layers, reshape_transform=reshape_transform
        )
    results = generate_cam_bboxes(
        data.reset_index(drop=True), 
        config,
        geotiff_dir, 
        model, 
        cam_extractor
    )
    
    results = filter_by_buildings(iso_code, config, results)
    if len(results) > 0:
        results = data_utils._connect_components(results, buffer_size=0)
        results = results.sort_values("prob", ascending=False).drop_duplicates(["group"])
        results.to_file(out_file, driver="GPKG")
    return results


def generate_cam_bboxes(data, config, in_dir, model, cam_extractor, show=False):    
    results = []
    data = data.reset_index(drop=True)
    filepaths = data_utils.get_image_filepaths(config, data, in_dir, ext=".tif")
    crs = data.crs
    for index in tqdm(list(data.index), total=len(data)):
        _, bbox = generate_cam(config, filepaths[index], model, cam_extractor, show=False)
        with rio.open(filepaths[index]) as map_layer:
            coords1 = [x[0] for x in list(map_layer.xy(bbox[0][1], bbox[0][0]))]
            coords2 = [x[0] for x in list(map_layer.xy(bbox[0][3], bbox[0][0]))]  
            coords3 = [x[0] for x in list(map_layer.xy(bbox[0][3], bbox[0][2]))]  
            coords4 = [x[0] for x in list(map_layer.xy(bbox[0][1], bbox[0][2]))]  
            coords = [coords1, coords2, coords3, coords4]
            polygon = geometry.Polygon(coords)
            crs = map_layer.crs
            results.append(polygon)
            if show:
                fig, ax = plt.subplots(figsize=(6, 6))
                rasterio.plot.show(map_layer, ax=ax)
                geom = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
                geom.plot(facecolor='none', edgecolor='blue', ax=ax)
                
    results = gpd.GeoDataFrame(geometry=results, crs=crs)
    results["prob"] = data.prob
    results["UID"] = data.UID
    return results


def georeference_images(data, config, in_dir, out_dir):
    filepaths = data_utils.get_image_filepaths(config, data, in_dir=in_dir)
    data = data.reset_index(drop=True)
    for index in tqdm(range(len(data)), total=len(data)):
        filename = os.path.join(out_dir, f"{data.iloc[index].UID}.tif")
        if not os.path.exists(filename):
            dataset = rio.open(filepaths[index], 'r')
            bands = [1, 2, 3]
            dataset = dataset.read(bands)
            bounds = data.iloc[index].geometry.bounds
            transform = rio.transform.from_bounds(
                bounds[0], 
                bounds[1], 
                bounds[2], 
                bounds[3], 
                config["width"], 
                config["height"]
            )
            crs = {'init': 'EPSG:3857'}
            
            with rio.open(
                filename, 
                'w', 
                driver='GTiff',
                width=config["width"], 
                height=config["height"],
                count=len(bands), 
                dtype=dataset.dtype, 
                nodata=0,
                transform=transform, 
                crs=crs
            ) as dst:
                dst.write(dataset, indexes=bands)


def compare_cams(filepath, model, model_config, classes, model_file):
    if model_config["type"] == "cnn":
        from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, LayerCAM
        for cam_extractor in GradCAM, GradCAMpp, SmoothGradCAMpp, LayerCAM:
            model = load_cnn(model_config, classes, model_file, verbose=False).eval()
            cam_extractor = cam_extractor(model)
            title = str(cam_extractor.__class__.__name__)
            generate_cam(model_config, filepath, model, cam_extractor, title=title)
    elif model_config["type"] == "vit":
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
        for cam_extractor in GradCAM, GradCAMPlusPlus, LayerCAM:
            model = load_cnn(model_config, classes, model_file, verbose=False).eval()
            target_layers = [model.module.model.backbone.backbone.features[-1][-2].norm1]
            cam_extractor = cam_extractor(
                model=model, target_layers=target_layers, reshape_transform=reshape_transform
            )
            title = str(cam_extractor.__class__.__name__)
            generate_cam(model_config, filepath, model, cam_extractor, title=title);


def generate_bbox_from_cam_cnn(cam_map, image, buffer=75):
    cam_arr = np.array(cam_map.cpu())
    ten_map = torch.tensor(cam_arr)
    values = []
    for i in range(0, ten_map.shape[0]):
        index, value = max(enumerate(ten_map[i]), key=operator.itemgetter(1))
        values.append(value)
    y_index, y_value = max(enumerate(values), key=operator.itemgetter(1))
    x_index, x_value = max(enumerate(ten_map[y_index]), key=operator.itemgetter(1))
    
    cms = cam_map.shape[0]
    x = x_index * (image.size[1] // cms)
    y = y_index * (image.size[0] // cms)
    
    boxes = torch.tensor([[
        x-buffer, 
        y-buffer, 
        (x + (image.size[0]//cms)) + buffer, 
        (y + (image.size[1]//cms)) + buffer
    ]])
    draw_boxes = torchvision.utils.draw_bounding_boxes(
        image=transforms.PILToTensor()(image), 
        boxes=boxes,
        width=5,
        colors="blue"
    )
    return boxes, F.to_pil_image(draw_boxes)

        
def generate_bbox_from_cam_vit(cam_map, image, buffer=75):
    def convert(img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()
    
        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

    if torch.is_tensor(cam_map):
        cam_map = np.array(cam_map.cpu())
    cam_arr = np.array(cam_map)
    ten_map = torch.tensor(cam_arr)
    values = []
    for i in range(0, ten_map.shape[0]):
        index, value = max(enumerate(ten_map[i]), key=operator.itemgetter(1))
        values.append(value)
    y_index, y_value = max(enumerate(values), key=operator.itemgetter(1))
    x_index, x_value = max(enumerate(ten_map[y_index]), key=operator.itemgetter(1))
    
    cms = cam_map.shape[0]
    x = x_index * (image.shape[1] // cms)
    y = y_index * (image.shape[0] // cms)
    
    boxes = torch.tensor([[
        x-buffer, 
        y-buffer, 
        (x + (image.shape[0]//cms)) + buffer, 
        (y + (image.shape[1]//cms)) + buffer
    ]])
    image = convert(image, 0, 255, np.uint8).transpose((2, 0, 1))
    draw_boxes = torchvision.utils.draw_bounding_boxes(
        image=torch.from_numpy(image),
        boxes=boxes,
        width=5,
        colors="blue"
    )
    return boxes, F.to_pil_image(draw_boxes)


def generate_cam(config, filepath, model, cam_extractor, show=True, title="", figsize=(7, 7)):
    logger = logging.getLogger()
    logger.disabled = True

    image = Image.open(filepath).convert("RGB")
    transforms = cnn_utils.get_transforms(config["img_size"])
    input = transforms["test"](image).to(device).unsqueeze(0)
    input_image = input.detach().cpu().numpy()[0].transpose((1, 2, 0))
    input_image = np.clip(
        np.array(cnn_utils.imagenet_std) * input_image + np.array(cnn_utils.imagenet_mean), 0, 1
    )
    output = model(input)

    if config["type"] == "cnn":
        cams = cam_extractor(output.squeeze(0).argmax().item(), output)
        for name, cam in zip(cam_extractor.target_names, cams):
            cam_map = cam.squeeze(0)
            result = overlay_mask(image, to_pil_image(cam_map, mode='F'), alpha=0.5)
        bbox, draw_bbox = generate_bbox_from_cam_cnn(cam_map, image, 75)
    elif config["type"] == "vit":
        cam_map= cam_extractor(input_tensor=input, targets=None)[0, :]
        result = show_cam_on_image(input_image, cam_map, use_rgb=True)
        bbox, draw_bbox = generate_bbox_from_cam_vit(cam_map, input_image)

    if show:
        fig, ax = plt.subplots(1,3, figsize=figsize)
        ax[0].imshow(image)
        ax[1].imshow(result)
        ax[2].imshow(draw_bbox)
        ax[1].title.set_text(title)
        plt.show()
        
    return cam_map, bbox
        

def cnn_predict_images(data, model, config, in_dir, classes, threshold=0.5):
    """
    Predicts labels and probabilities for the given dataset using the provided model.

    Args:
    - data (Pandas DataFrame): Input dataset.
    - model (torch.nn.Module): Trained neural network model.
    - c (dict): Configuration dictionary.
    - in_file (str): Path to the input file.
    - out_dir (str): Directory path to save output files.
    - classes (dict): Dictionary containing class labels.
    - scale (float, optional): Scale factor for cropping shapes (default: 1.5).

    Returns:
    - Results as a GeoDataFrame containing predicted labels and probabilities for the dataset.
    """
    
    files = data_utils.get_image_filepaths(config, data, in_dir)
    preds, probs = [], []
    pbar = data_utils._create_progress_bar(files)

    for file in pbar:
        image = Image.open(file).convert("RGB")
        transforms = cnn_utils.get_transforms(config["img_size"])
        output = model(transforms["test"](image).to(device).unsqueeze(0))
        soft_outputs = nnf.softmax(output, dim=1).detach().cpu().numpy()
        probs.append(soft_outputs[:, 1][0])
        
    preds = np.array(probs) > threshold
    preds = [str(classes[int(pred)]) for pred in preds]
    data["pred"] = preds
    data["prob"] = probs
    return data


def cnn_predict(
    data, 
    iso_code, 
    shapename, 
    config, 
    in_dir=None, 
    out_dir=None, 
    n_classes=None, 
    threshold=0.5
):
    cwd = os.path.dirname(os.getcwd())
    if not out_dir:
        out_dir = os.path.join("output", iso_code, "results", config["project"])
        out_dir = data_utils._makedir(out_dir)
    
    name = f"{iso_code}_{shapename}"
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")

    if os.path.exists(out_file):
        return gpd.read_file(out_file)
    
    classes = {1: config["pos_class"], 0: config["neg_class"]}
    exp_dir = os.path.join(
        cwd, config["exp_dir"], 
        config["project"], 
        f"{iso_code}_{config['config_name']}"
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = load_cnn(config, classes, model_file)

    results = cnn_predict_images(data, model, config, in_dir, classes, threshold)
    results = results[["UID", "geometry", "pred", "prob"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")
    results.to_file(out_file, driver="GPKG")
    return results


def load_cnn(c, classes, model_file=None, verbose=True):
    """
    Loads a pre-trained model based on the provided configuration.

    Args:
    - c (dict): Configuration dictionary containing model details.
    - classes (dict): Dictionary containing class labels.
    - model_file (str, optional): Path to the pre-trained model file.

    Returns:
    - Loaded pre-trained model based on the provided configuration.
    """
    
    n_classes = len(classes)
    model = cnn_utils.get_model(c["model"], n_classes, c["dropout"])
    model= torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    if verbose:
        logging.info(f"Device: {device}")
        logging.info("Model file {} successfully loaded.".format(model_file))
    return model


def load_vit(config):
    model = torch.hub.load("facebookresearch/dinov2", config["embed_model"])
    model.name = config["embed_model"]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logging.info(f"Device: {device}")
    return model


def vit_pred(data, config, iso_code, shapename, sat_dir, id_col="UID"):
    cwd = os.path.dirname(os.getcwd())
    model = load_vit(config)

    # Generate Embeddings
    logging.info("Generating embeddings...")
    out_dir = os.path.join("output", iso_code, "embeddings")
    name = f"{iso_code}_{shapename}"
    embeddings = embed_utils.get_image_embeddings(
        config, data, model, in_dir=sat_dir, out_dir=out_dir, name=name
    )

    # Load shallow model
    exp_dir = os.path.join(cwd, config["exp_dir"], config["project"], f"{iso_code}-{config['config_name']}")
    model_file = os.path.join(exp_dir, config["project"], f"{iso_code}-{config['config_name']}.pkl")
    model = joblib.load(model_file)
    logging.info(f"Loaded {model_file}")

    # Model prediction
    preds = model.predict(embeddings)
    data["pred"] = preds
    results = data[["UID", "geometry", "shapeName", "pred"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")

    # Save results
    out_dir = os.path.join("output", iso_code, "results")
    out_dir = data_utils._makedir(out_dir)
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")
    results.to_file(out_file, driver="GPKG")
    logging.info(f"Generated {out_file}")
    
    return results


def filter_by_buildings(iso_code, config, data, n_seconds=10):
    cwd = os.path.dirname(os.getcwd())
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    google_path = os.path.join(raster_dir, "google_buildings", f"{iso_code}_google.tif")
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])

    pixel_sums = []
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(
        range(len(data)), 
        total=len(data), 
        mininterval=n_seconds,
        bar_format=bar_format
    )
    for index in pbar:
        subdata = data.iloc[[index]]
        pixel_sum = 0
        try:
            with rio.open(ms_path) as ms_source:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(ms_source, geometry, crop=True)
                image[image == 255] = 1
                pixel_sum = np.sum(image)
        except:
            pass
        if pixel_sum == 0 and os.path.exists(google_path):
            try:
                with rio.open(google_path) as google_source:
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(google_source, geometry, crop=True)
                    image[image == 255] = 1
                    pixel_sum = np.sum(image)
            except:
                pass
        pixel_sums.append(pixel_sum) 
        
    data["sum"] = pixel_sums
    data = data[data["sum"] > 0]
    data = data.reset_index(drop=True)
    return data


def generate_pred_tiles(config, iso_code, spacing, buffer_size, adm_level="ADM2", shapename=None):
    cwd = os.path.dirname(os.getcwd())
    out_dir = data_utils._makedir(os.path.join(cwd, "output", iso_code, "tiles"))
    out_file = os.path.join(out_dir, f"{iso_code}_{shapename}.gpkg")
    
    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        return data
    
    points = data_utils._generate_samples(
        config, 
        iso_code=iso_code, 
        buffer_size=buffer_size, 
        spacing=spacing, 
        adm_level=adm_level, 
        shapename=shapename
    )
    
    logging.info(f"Shapename: {shapename}")
    points["points"] = points["geometry"]
    points["geometry"] = points.buffer(buffer_size, cap_style=3)
    points["UID"] = list(points.index)
    
    filtered = filter_by_buildings(iso_code, config, points)
    filtered = filtered[["UID", "geometry", "shapeName", "sum"]]
    filtered.to_file(out_file, driver="GPKG", index=False)
    return filtered