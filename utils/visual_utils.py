import os
import folium
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import translators as ts

from IPython.display import display
from ipywidgets import Layout, GridspecLayout, Button, Image, Tab

import logging
from utils import data_utils
from utils import pred_utils
from utils import model_utils

logging.basicConfig(level=logging.INFO)


def get_filename(
    iso_code: str, 
    config: dict, 
    category: str = "school", 
    name: str = "clean"
):
    """
    Constructs the file path for a GeoJSON file based on the specified parameters.

    Args:
        iso_code (str): ISO code for the country, used to identify the file.
        config (dict): Configuration dictionary with directory paths and project details. Includes:
            - vectors_dir: Directory path for vector files.
            - project: Project-specific directory name.
        category (str, optional): Category folder for the file, such as "school" or "admin". Default is "school".
        name (str, optional): Base name for the file, e.g. clean. Default is "clean".

    Returns:
        str: Full file path for the specified GeoJSON file.
    """
    # Construct the full file path for the GeoJSON file based on the given parameters
    filename = os.path.join(
        os.getcwd(),
        config["vectors_dir"],
        config["project"],
        category,
        name,
        f"{iso_code}_{name}.geojson",
    )
    return filename


def map_coordinates(
    config: dict,
    index: int,
    category: str,
    iso_code: str,
    filename: str = None,
    zoom_start: int = 18,
    max_zoom: int = 20,
    name: str = "clean",
    col_uid: str = "UID",
    col_name: str = "name"
) -> None:
    """
    Display a folium map centered on a specific feature from a GeoJSON file.

    Args:
        config (dict): Configuration dictionary containing project settings.
        index (int): Index of the feature to center the map on.
        category (str): Category of the GeoJSON file within the project.
        iso_code (str): ISO code of the country or region.
        filename (str, optional): Name of the GeoJSON file (default is None).
        zoom_start (int, optional): Initial zoom level of the map (default is 18).
        max_zoom (int, optional): Maximum zoom level allowed for the map (default is 20).
        name (str, optional): Name of the GeoJSON file category (default is "clean").
        col_uid (str, optional): Column name containing unique identifiers (default is "UID").
        col_name (str, optional): Column name containing feature names (default is "name").

    Returns:
        None
    """
    # Read GeoJSON file into a GeoDataFrame
    if not filename:
        filename = get_filename(iso_code, config, category, name)
    data = gpd.read_file(filename)

    # Extract name of the feature at the specified index
    name = data[data.index == index].iloc[0][col_name]
    logging.info(data[data.index == index].iloc[0][col_uid])
    logging.info(name)

    # Translate the feature name if available
    if name:
        logging.info(ts.translate_text(name, translator="google"))

    # Extract coordinates of the feature
    coords = data.iloc[index].geometry.y, data.iloc[index].geometry.x

    # Create a folium map centered on the feature coordinates
    map = folium.Map(location=coords, zoom_start=zoom_start, max_zoom=max_zoom)
    # Add Google Satellite layer to the map
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ).add_to(map)

    # Add marker at the feature coordinates
    folium.Marker(location=coords, fill_color="#43d9de", radius=8).add_to(map)

    # Display the map
    display(map)


def generate_predictions(
    iso_code: str, 
    config: dict,
    model_iso_code: str, 
    category: str = "school"
) -> gpd.GeoDataFrame:
    """
    Generates predictions by averaging the output probabilities from multiple models for a specified category.

    Args:
        iso_code (str): ISO code for the country for which predictions are generated.
        model_iso_code (str): ISO code for the country associated with the model configurations.
        model_configs (list): List of configuration dictionaries, each containing model settings. 
            Each dictionary includes:
            - rasters_dir: Directory containing raster files.
            - maxar_dir: Directory for high-resolution imagery.
            - project: Project directory name.
        category (str, optional): Category of the predictions, such as "school" or "admin". 
            Default is "school".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with prediction probabilities averaged across models.
    """
    # Retrieve configurations for the ensemble models
    model_configs = model_utils.get_ensemble_configs(model_iso_code)

    # Determine output file path for predictions
    out_file = get_filename(
        iso_code, config, category="school", name="clean"
    )

    # If predictions already exist, load and return them
    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        if 'prob' in data.columns:
            return data

    # Define input directory and file path based on model configuration
    in_dir = os.path.join(
        os.getcwd(),
        config["rasters_dir"],
        config["maxar_dir"],
        config["project"],
        iso_code,
        category.lower(),
    )
    in_file = get_filename(iso_code, config, category="school", name="clean")

    # Read the input data
    data = gpd.read_file(in_file)

    # Initialize probability sum for ensemble prediction
    probs = 0
    # Loop through each model configuration to generate and accumulate probabilities
    for model_config in model_configs:
        model = pred_utils.load_model(model_iso_code, model_config, verbose=True)
        data = pred_utils.cnn_predict_images(
            data[data.clean == 0], model, model_config, in_dir
        )
        # Add model probabilities to cumulative sum
        probs = probs + data["prob"].to_numpy()

    # Calculate the average probability across models
    data["prob"] = probs / len(model_configs)

    # Save the prediction data to the output file
    out_file = get_filename(
        iso_code, config, category="school", name="clean"
    )
    data.to_file(out_file)
    return data


def validate_data(
    config: dict,
    iso_code: str,
    category: str,
    row_inc: int = 3,
    n_rows: int = 4,
    n_cols: int = 4,
    start_index: int = 0,
    filename: str = None,
    name: str = "clean",
    col_uid: str = "UID",
    show_validated: bool = True,
    min_prob: float = 0,
    max_prob: float = 1.0,
) -> GridspecLayout:
    """
    Display a grid of images and validation buttons for manual validation of data samples.

    Args:
        config (dict): Configuration dictionary containing project settings.
        iso_code (str): ISO code of the country or region.
        category (str): Category of the GeoJSON file within the project.
        start_index (int, optional): Starting index of data samples (default is 0).
        row_inc (int, optional): Increment value for rows in the grid layout (default is 3).
        n_rows (int, optional): Number of rows in the grid layout (default is 4).
        n_cols (int, optional): Number of columns in the grid layout (default is 4).
        filename (str, optional): Name of the GeoJSON file (default is None).
        name (str, optional): Name of the GeoJSON file category (default is "clean").
        col_uid (str, optional): Column name containing unique identifiers (default is "UID").
        show_validated (bool, optional): Whether to show validated samples (default is True).

    Returns:
        GridspecLayout: Grid layout displaying images and validation buttons.
    """
    # Read GeoJSON file into a GeoDataFrame
    if not filename:
        filename = get_filename(iso_code, config, category, name)
    data = gpd.read_file(filename)

    # Ensure the "validated" column exists and initialize to 0 if not present
    if "validated" not in data.columns:
        data["validated"] = 0

    # Filter samples based on conditions (clean and optionally show validated)
    samples = data[(data["clean"] == 0)]
    if "prob" in samples.columns:
        samples = samples[(samples["prob"] >= min_prob) & (samples["prob"] <= max_prob)]
    if not show_validated:
        samples = samples[(samples["validated"] == 0)]
    samples = samples.iloc[start_index : start_index + (n_rows * n_cols)]

    # Create a GridspecLayout for displaying images and buttons
    grid = GridspecLayout(n_rows * row_inc + n_rows, n_cols)
    button_dict = {0: ("primary", category), -1: ("warning", "unrecognized")}

    def add_image(item):
        # Function to add image to Tab in the grid
        class_dir = os.path.join(
            os.getcwd(),
            config["rasters_dir"],
            config["maxar_dir"],
            config["project"],
            iso_code,
            category.lower(),
        )
        filepath = os.path.join(class_dir, f"{item[col_uid]}.tiff")
        img = open(filepath, "rb").read()
        image = Tab(
            [
                Image(
                    value=img,
                    format="png",
                    layout=Layout(
                        justify_content="center",
                        border="solid",
                        width="auto",
                        height="auto",
                    ),
                )
            ]
        )
        name = item["name"]
        # Truncate name if too long
        if name:
            name = name[:20]
        image.set_title(0, name)

        return image

    def on_button_click(button):
        # Function to handle button click events for validation
        index = int(button.description.split(" ")[0])
        item = data.iloc[index]

        change_value = -1
        if item["validated"] == -1:
            change_value = 0
        button_style, category = button_dict[change_value]

        button.button_style = button_style
        button.description = f"{item.name} {category.upper()}"

        data.loc[index, "validated"] = change_value
        data.to_file(filename, driver="GeoJSON")

    def create_button(item):
        # Function to create validation buttons
        val = item["validated"]
        button_style, category = button_dict[val]
        description = f"{item.name} {category.upper()}"

        return Button(
            description=description,
            button_style=button_style,
            layout=Layout(
                justify_content="center", border="solid", width="auto", height="10"
            ),
        )

    # Populate the grid with images and buttons
    row_index, col_index = 0, 0
    for index, item in samples.iterrows():
        grid[row_index : row_index + row_inc, col_index] = add_image(item)
        button = create_button(item)
        button.on_click(on_button_click)
        grid[row_index + row_inc, col_index] = button

        col_index += 1
        if col_index >= n_cols:
            row_index += row_inc + 1
            col_index = 0

    return grid


def update_validation(
    config: dict, 
    iso_code: str, 
    category: str, 
    indexes: list = [], 
    name: str = "clean", 
    filename: str = None
) -> None:
    """
    Updates the validation status of specified entries in a GeoDataFrame by 
    toggling their "validated" column value. This is an alternative to inspect_images().

    Args:
        config (dict): Configuration dictionary with required keys:
            - vectors_dir: Directory where vector data is stored.
            - project: Project name, used in directory structure.
        iso_code (str): ISO code for the country.
        category (str): Category of the data (e.g. :school").
        indexes (list, optional): List of row indices to update. Default is an empty list.
        name (str, optional): Specific name of the data subset, used to build the file path. 
            Default is "clean".
        filename (str, optional): Path to the GeoDataFrame file. 
            If not provided, it will be generated from other parameters.

    Returns:
        None
    """
    validated = {0: "VALID", -1: "INVALID"}
    if not filename:
        filename = get_filename(iso_code, config, category, name="clean")
    data = gpd.read_file(filename) 

    if "validated" not in data.columns:
        data["validated"] = 0
    
    for index in indexes:
        item = data.iloc[index]
        change_value = -1
        if item["validated"] == -1:
            change_value = 0

        data.loc[index, "validated"] = change_value
        logging.info(f"Item {index} changed to {validated[data.iloc[index]['validated']]}.")

    data.to_file(filename, driver="GeoJSON")


def inspect_images(
    config: dict,
    iso_code: str,
    category: str,
    n_rows: int = 4,
    n_cols: int = 4,
    start_index: int = 0,
    filename: str = None,
    random: bool = False,
    col_uid: str = "UID",
    col_name: str = "name",
    figsize: tuple = (15, 15),
    min_prob: float = 0,
    max_prob: float = 1.0,
    show_validated=False
) -> None:
    """
    Inspect and visualize satellite images associated with geographic data.

    Args:
        config (dict): Configuration dictionary containing project settings.
        iso_code (str): ISO code of the country or region.
        category (str): Category of the GeoJSON file within the project.
        n_rows (int, optional): Number of rows in the subplot grid (default is 4).
        n_cols (int, optional): Number of columns in the subplot grid (default is 4).
        start_index (int, optional): Starting index of data samples (default is 0).
        filename (str, optional): Name of the GeoJSON file (default is None).
        random (bool, optional): Whether to shuffle the data samples (default is False).
        col_uid (str, optional): Column name containing unique identifiers (default is "UID").
        col_name (str, optional): Column name containing names or labels (default is "name").
        figsize (tuple, optional): Figure size for matplotlib subplots (default is (15, 15)).

    Returns:
        None
    """
    # Read GeoJSON file into a GeoDataFrame
    if not filename:
        filename = get_filename(iso_code, config, category, name="clean")
    data = gpd.read_file(filename)

    # Ensure the "validated" column exists and initialize to 0 if not present
    if "validated" not in data.columns:
        data["validated"] = 0

    # Optionally shuffle the data samples
    if random:
        data = data.sample(frac=1.0)

    # Filter data based on columns "clean" and "validated"
    if "clean" in data.columns:
        data = data[data["clean"] == 0]
    if not show_validated:
        data = data[data["validated"] == 0]
    if "prob" in data.columns:
        data = data[(data["prob"] >= min_prob) & (data["prob"] <= max_prob)]
    logging.info(f"Data dimensions: {data.shape}")

    # Create a grid of subplots for visualizing images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    samples = data.iloc[start_index : start_index + (n_rows * n_cols)]
    row_index, col_index = 0, 0

    # Iterate over the samples to display associated images
    for idx, item in samples.iterrows():
        class_dir = os.path.join(
            os.getcwd(),
            config["rasters_dir"],
            config["maxar_dir"],
            config["project"],
            iso_code,
            category.lower(),
        )
        filepath = os.path.join(class_dir, f"{item[col_uid]}.tiff")
        validated = {0: "VALID", -1: "INVALID"}

        # Open and display the image on the subplot
        image = rio.open(filepath)
        show(image, ax=axes[row_index, col_index])
        axes[row_index, col_index].tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        axes[row_index, col_index].set_axis_off()
        axes[row_index, col_index].set_title(
            f"Index: {idx}\nName: {item[col_name]}\nSource: {item['source']}\n{validated[item['validated']]}", 
            fontdict={"fontsize": 11}
        )

        col_index += 1
        if col_index >= n_cols:
            row_index += 1
            col_index = 0
        if row_index >= n_rows:
            break
            
    fig.tight_layout() 