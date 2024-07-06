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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


cams = {
    "randomcam": RandomCAM,
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "gradcamelementwise": GradCAMElementWise,
}


def get_cam_extractor(config, model, cam_extractor):
    reshape_transform = None
    cam_extractor = cams[cam_extractor]
    if "vit" in config["model"]:

        class ReshapeTransform:
            def __init__(self, height, width):
                self.height = height
                self.width = width

            def reshape_transform(self, tensor):
                result = tensor[:, 1:, :].reshape(
                    tensor.size(0), self.height, self.width, tensor.size(2)
                )
                result = result.transpose(2, 3).transpose(1, 2)
                return result

        target_layers = [model.module.encoder.layers[-1].ln_1]
        if config["model"] == "vit_h_14":
            reshape_transform = ReshapeTransform(16, 16).reshape_transform
        elif config["model"] == "vit_l_16":
            reshape_transform = ReshapeTransform(14, 14).reshape_transform

    elif "swin" in config["model"].lower():

        def reshape_transform(tensor):
            result = tensor.transpose(2, 3).transpose(1, 2)
            return result

        if "satlas-aerial_swinb_si" in config["model"].lower():
            target_layers = [
                model.module.model.backbone.backbone.features[-1][-1].norm1
            ]
        elif "satlas-aerial_swinb_mi" in config["model"].lower():
            target_layers = [
                model.module.model.backbone.backbone.backbone.features[-1][-1].norm1
            ]

    return cam_extractor(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )


def cam_predict(
    iso_code,
    config,
    data,
    geotiff_dir,
    shapename,
    cam_method="gradcam",
    buffer_size=50,
    verbose=False,
):
    out_dir = data_utils.makedir(
        os.path.join(
            os.getcwd(),
            "output",
            iso_code,
            "results",
            config["project"],
            "cams",
            config["config_name"],
            cam_method,
        )
    )
    out_file = os.path.join(
        out_dir, f"{iso_code}_{shapename}_{config['config_name']}_{cam_method}.gpkg"
    )
    if os.path.exists(out_file):
        return gpd.read_file(out_file)
    model = pred_utils.load_model(iso_code, config).eval()
    cam_extractor = get_cam_extractor(config, model, cam_method)

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
        _, point, _ = generate_cam(
            config, filepaths[index], model, cam_extractor, metrics=False, show=False
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


def compare_cams(
    iso_code, config, filepaths, percentile=90, show=True, metrics=True, verbose=False
):
    cam_scores = dict()
    for cam_name, cam in cams.items():
        model = pred_utils.load_model(iso_code, config, verbose=verbose).eval()
        model = model.to(device)
        cam_scores[cam_name] = []
        with get_cam_extractor(config, model, cam_name) as cam_extractor:
            cam_extractor.batch_size = config["batch_size"]
            pbar = data_utils.create_progress_bar(filepaths) if not show else filepaths
            for filepath in pbar:
                if not show:
                    pbar.set_description(f"Processing {cam_name}")
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
                print(score)
                cam_scores[cam_name].append(score)

    cam_scores_mean = dict()
    for cam in cam_scores:
        cam_scores_mean[cam] = np.mean(cam_scores[cam])
    return cam_scores, cam_scores_mean


def compare_random(
    iso_code,
    data,
    config,
    filepaths,
    percentile=90,
    cam="layercam",
    n_samples=3,
    show=True,
    verbose=False,
):
    if "class" in data.columns:
        data = data[(data["class"] == config["pos_class"])]

    model = pred_utils.load_model(iso_code, config, verbose=verbose).eval()
    cam_extractor = get_cam_extractor(config, model, cam)
    for index in list(data.sample(n_samples).index):
        title = str(cam_extractor.__class__.__name__)
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
    config,
    filepath,
    model,
    cam_extractor,
    percentile=90,
    metrics=True,
    show=True,
    title="",
    save=None,
    figsize=(9, 9),
):
    logger = logging.getLogger()
    logger.disabled = True

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
    targets = [ClassifierOutputTarget(1)]
    cam_map = cam_extractor(input_tensor=input_tensor, targets=targets)
    result = show_cam_on_image(input_image, cam_map[0, :], use_rgb=True)
    point = generate_point_from_cam(config, cam_map[0, :], image)

    score = 0
    if metrics:
        cam_metric = ROADMostRelevantFirst(percentile=percentile)
        scores, road_visualizations = cam_metric(
            input_tensor, cam_map, targets, model, return_visualization=True
        )
        score = scores[0]

    if show:
        road_visualization = road_visualizations[0].cpu().numpy().transpose((1, 2, 0))
        road_visualization = deprocess_image(road_visualization)
        road_visualization = Image.fromarray(road_visualization)

        thresh_cam = cam_map < np.percentile(cam_map, percentile)
        thresh_cam = thresh_cam.transpose((0, 1, 2))[0, :, :]

        n_axis = 3
        fig, ax = plt.subplots(1, n_axis, figsize=figsize, dpi=300)
        ax[0].imshow(image)
        ax[1].imshow(result)
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
