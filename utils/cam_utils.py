import os
import logging
import operator
import numpy as np
from tqdm import tqdm

from shapely import geometry
import geopandas as gpd
import rasterio as rio
import rasterio.plot

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torchvision
from torchcam.utils import overlay_mask
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from utils import data_utils
from utils import pred_utils
from utils import cnn_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def get_layer_cam(config):
    if config["type"] == "cnn":
        from torchcam.methods import LayerCAM
    elif config["type"] == "vit":
        from pytorch_grad_cam import LayerCAM
    return LayerCAM


def get_cam_extractor(config, model, cam_extractor):
    def reshape_transform(tensor):
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result

    if config["type"] == "cnn":
        return cam_extractor(model)

    elif config["type"] == "vit":
        if "satlas-aerial_swinb_si" in config["model"].lower():
            target_layers = [
                model.module.model.backbone.backbone.features[-1][-2].norm1
            ]
        elif "satlas-aerial_swinb_mi" in config["model"].lower():
            target_layers = [
                model.module.model.backbone.backbone.backbone.features[-1][-1].norm1
            ]
        return cam_extractor(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )


def cam_predict(iso_code, config, data, geotiff_dir, shapename, buffer_size=50):
    classes = {1: config["pos_class"], 0: config["neg_class"]}
    out_dir = data_utils.makedir(
        os.path.join(
            os.getcwd(),
            "output",
            iso_code,
            "results",
            config["project"],
            "cams",
            config["config_name"],
        )
    )
    out_file = os.path.join(
        out_dir, f"{iso_code}_{shapename}_{config['config_name']}_cam.gpkg"
    )
    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    exp_dir = os.path.join(
        os.getcwd(),
        config["exp_dir"],
        config["project"],
        f"{iso_code}_{config['config_name']}",
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = pred_utils.load_cnn(config, classes, model_file, verbose=False).eval()
    cam_extractor = get_cam_extractor(config, model, get_layer_cam(config))

    results = generate_cam_points(
        data, config, geotiff_dir, model, cam_extractor, buffer_size
    )
    results = pred_utils.filter_by_buildings(iso_code, config, results)
    if len(results) > 0:
        results = data_utils.connect_components(results, buffer_size=0)
        results = results.sort_values("prob", ascending=False).drop_duplicates(
            ["group"]
        )
        results.to_file(out_file, driver="GPKG")
    return results


def generate_cam_points(
    data, config, in_dir, model, cam_extractor, buffer_size=50, show=False
):
    results = []
    data = data.reset_index(drop=True)
    filepaths = data_utils.get_image_filepaths(config, data, in_dir, ext=".tif")
    crs = data.crs
    for index in tqdm(list(data.index), total=len(data)):
        _, point = generate_cam(
            config, filepaths[index], model, cam_extractor, show=False
        )
        with rio.open(filepaths[index]) as map_layer:
            coord = [map_layer.xy(point[1], point[0])]
            coord = geometry.Point(coord)
            crs = map_layer.crs
            results.append(coord)
            if show:
                fig, ax = plt.subplots(figsize=(6, 6))
                rasterio.plot.show(map_layer, ax=ax)
                geom = gpd.GeoDataFrame(geometry=[coord], crs=crs)
                geom.plot(facecolor="none", edgecolor="blue", ax=ax)

    results = gpd.GeoDataFrame(geometry=results, crs=crs)
    results = results.to_crs("EPSG:3857")
    results["geometry"] = results["geometry"].buffer(buffer_size, cap_style=3)
    results = results.to_crs(crs)
    results["prob"] = data.prob
    results["UID"] = data.UID

    return results


def georeference_images(data, config, in_dir, out_dir):
    filepaths = data_utils.get_image_filepaths(config, data, in_dir=in_dir)
    data = data.reset_index(drop=True)
    for index in tqdm(range(len(data)), total=len(data)):
        filename = os.path.join(out_dir, f"{data.iloc[index].UID}.tif")
        if not os.path.exists(filename):
            dataset = rio.open(filepaths[index], "r")
            bands = [1, 2, 3]
            dataset = dataset.read(bands)
            bounds = data.iloc[index].geometry.bounds
            transform = rio.transform.from_bounds(
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                config["width"],
                config["height"],
            )
            crs = {"init": "EPSG:3857"}

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


def compare_cams(iso_code, config, filepath):
    if config["type"] == "cnn":
        from torchcam.methods import LayerCAM, GradCAM, GradCAMpp, SmoothGradCAMpp

        cams = [LayerCAM, GradCAM, GradCAMpp, SmoothGradCAMpp]
    elif config["type"] == "vit":
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM

        cams = [LayerCAM, GradCAM, GradCAMPlusPlus]

    for cam in cams:
        model = pred_utils.load_cnn(iso_code, config, verbose=False).eval()
        cam_extractor = get_cam_extractor(config, model, cam)
        title = str(cam_extractor.__class__.__name__)
        generate_cam(config, filepath, model, cam_extractor, title=title)


def compare_cams_random(iso_code, data, config, filepaths, n_samples=3, verbose=False):
    if "class" in data.columns:
        data = data[(data["class"] == config["pos_class"])]

    layer_cam = get_layer_cam(config)
    model = pred_utils.load_cnn(iso_code, config, verbose=False).eval()
    cam_extractor = get_cam_extractor(config, model, layer_cam)
    for index in list(data.sample(n_samples).index):
        title = str(cam_extractor.__class__.__name__)
        generate_cam(config, filepaths[index], model, cam_extractor, title=title)


def generate_cam(
    config, filepath, model, cam_extractor, show=True, title="", figsize=(5, 5)
):
    logger = logging.getLogger()
    logger.disabled = True

    image = Image.open(filepath).convert("RGB")
    transforms = cnn_utils.get_transforms(config["img_size"])
    input = transforms["test"](image).to(device).unsqueeze(0)
    input_image = input.detach().cpu().numpy()[0].transpose((1, 2, 0))
    input_image = np.clip(
        np.array(cnn_utils.imagenet_std) * input_image
        + np.array(cnn_utils.imagenet_mean),
        0,
        1,
    )
    output = model(input)

    if config["type"] == "cnn":
        cams = cam_extractor(1, output)
        for name, cam in zip(cam_extractor.target_names, cams):
            cam_map = cam.squeeze(0)
            result = overlay_mask(image, to_pil_image(cam_map, mode="F"), alpha=0.5)

    elif config["type"] == "vit":
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(1)]
        cam_map = cam_extractor(input_tensor=input, targets=targets)[0, :]
        result = show_cam_on_image(input_image, cam_map, use_rgb=True)

    point = generate_point_from_cam(config, cam_map, image)

    if show:
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 3, figsize=figsize)
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
        ax[2].imshow(image)
        ax[2].add_patch(rect)
        ax[1].title.set_text(title)
        ax[0].xaxis.set_visible(False)
        ax[1].xaxis.set_visible(False)
        ax[2].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[2].yaxis.set_visible(False)
        plt.show()

    return cam_map, point


def generate_point_from_cam(config, cam_map, image):
    if torch.is_tensor(cam_map):
        cam_map = np.array(cam_map.cpu())

    ten_map = torch.tensor(cam_map)
    transform = transforms.Resize(size=(image.size[0], image.size[1]), antialias=None)
    ten_map = transform(ten_map.unsqueeze(0)).squeeze()

    values = []
    for i in range(0, ten_map.shape[0]):
        index, value = max(enumerate(ten_map[i]), key=operator.itemgetter(1))
        values.append(value)

    y_index, y_value = max(enumerate(values), key=operator.itemgetter(1))
    x_index, x_value = max(enumerate(ten_map[y_index]), key=operator.itemgetter(1))

    return (x_index, y_index)
