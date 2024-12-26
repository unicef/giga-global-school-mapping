<div style="padding-top: 20px;"> </div>
<h1><a id="giga-ai" class="anchor" aria-hidden="true" href="#giga-ai"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
File Organization </h1> 

<details open="open">
	<summary style="padding-bottom: 10px;"><h2>Table of Contents</h2></summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
  <li><a href="#top-level-dir">Top-level Directory</a></li>
  <li><a href="#config-dir">Configurations</a></li>
  <li><a href="#data-dir">Data</a></li>
  <li><a href="#exp-dir">Experiments</a></li>
  <li><a href="#output-dir">Outputs</a></li>
</details>

<h2><a id="overview" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Overview</h2>


The <a href="https://github.com/unicef/giga-global-school-mapping/blob/master/File-Organization.md">File Organization document</a> provides a structured overview of how files and directories are arranged for the project. It details the naming conventions, folder hierarchy, and organizational principles used to manage data, scripts, and outputs effectively, ensuring streamlined collaboration and accessibility for team members. This guidance helps maintain consistency and clarity in the project workflow.

<h2><a id="top-level-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Top-level Directory</h2>

```
.
├── configs                 # Configuration files
├── cv                      # Cross-country cross-validation experiments
├── data                    # Datasets for model development 
├── exp                     # Local and regional experiments 
├── notebooks               # Jupyter notebooks for exploratory analysis
├── output                  # Datasets for model deployment
├── src                     # Python scripts for model development and deployment
└── utils                   # Python utility functions
```

<h2><a id="config-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Configurations</h2>

<b>Path:</b> `/configs`

<p>This directory contains configuration files for training data preparation, model development, and deployment.</p>

```
.
├── ...
├── configs                                   # Configuration files
│   ├── config.yaml                           # Global configuration file
│   ├── creds.share                           # GigaMaps API credentials
│   ├── cnn_configs                           # CNN configuration files
│   │   ├── convnext_small.yaml
│   │   └── ...
│   ├── vit_configs                           # ViT/Swin configuration files 
│   │   ├── vit_h_14.yaml
│   │   ├── swin_v2_s.yaml
│   │   └── ...
│   ├── data_configs                          # Data configuration files
│   │   ├── data_config_ISO_<ISO>.yaml
│   │   └── ...
│   └── sat_config                            # Satellite image download configuration files
│       ├── sat_creds.yaml                    # Maxar satellite image credentials
│       ├── sat_config.yaml                   # Maxar satellite image configuration
│       └── ...
└── ...
```
<h3>Global Configuration File</h3>

<b>Path:</b> `config.yaml`

<p>The global configuration file <code>config.yaml</code> defines key parameters such as directories, URLs, and default settings for data preparation and model training:</p>

```
project: "GIGAv1"                 # The project name
pos_class: "school"               # The positive class name
neg_class: "non_school"           # The negative class name

exp_dir: 'exp/'                   # The experiments directory, where model experiment results are saved
vectors_dir: 'data/vectors/'      # The vector directory - all vector files (.gpkg, geojson) will be saved here
rasters_dir: 'data/rasters/'      # The raster directory - all raster TIFF files will be saved here
maxar_dir: 'maxar/500x500_60cm'   # The directory for Maxar satellite images, nested under the raster directory  

# The column names for data preparation are listed as follows.
columns: ['UID', 'source', 'iso', 'country', 'region', 'subregion', 'name', 'geometry', 'school_id_giga']

# URLs for downloading data from Overture, Microsoft, Google, GHSL, and Geoboundaries
microsoft_url: "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
google_url: "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
...

# Relevant models
all_models: {}    # A dictionary where the keys are the model type (e.g. convnext, vit, swin) 
                  # and the values are the model variants (e.g. convnext_base, vit_h_14, etc.)

# School and non-school keywords
school: {}       # A dictionary of keywords for fetching school points of interest (e.g. "school", "academy")
non_school: {}   # A dictionary of keywords for fetching non-school points of interests (e.g. "hospital", "church")

# School keyword exclusion
exclude: []     # Exclude schools containing non-relevant keywords
                # e.g. business culinary, driving, beauty         
```

<h3>Data Configuration</h3>

<b>Path:</b> `data_configs/`

This directory contains YAML files used for preparing training data and conducting model experimentation.

These configuration files define the parameters and settings required for data preparation and model training workflows.

<b>Example</b>:
For Tajikistan, the configuration file is:
`data_configs/data_config_ISO_TJK.yaml`

```
iso_codes: ['TJK']              # The iso_codes to be processed.
name:                           # Indicate the name of the experiment here, if different from the ISO code 
                                # e.g. for the African regional model, the name is set to "AF"

proximity: 300                  # All points within this proximity are grouped together and marked as "duplicates"
filter_buffer_size: 150         # The radius of the buffer area to be used for filtering uninhabited locations
sample_spacing: 300             # The spacing of tiles used for sampling tiles for non-school sample augmentation
object_proximity: 300           # Filters/removes all non-school tiles that are within this proximity to school tiles

priority: ["UNICEF", "OSM", "OVERTURE"]   # Prioritization of datasets for choosing the point 
                                          # to retain among grouped/"duplicate" points
```

<h3>Maxar Satellite Image Credentials</h3>

<b>Path:</b> `sat_configs/sat_creds.yaml`

This file stores the login credentials required to access the Maxar platform at: <a href="https://evwhs.digitalglobe.com/myDigitalGlobe/login">https://evwhs.digitalglobe.com</a>.

Ensure that this file is securely managed to prevent unauthorized access to your account.
```
username:     # Maxar username
password:     # Maxar password
connect_id:   # Maxar connect_id (see profile >> Connect ID)
```

<h3>Maxar Satellite Image Configurations</h3>

<b>Path:</b> `sat_configs/sat_config.yaml`

This file specifies the parameters for downloading Maxar satellite images. The default settings are detailed in the file, enabling users to customize the image download process to suit their needs.

For additional details about the parameters and usage, refer to the Maxar documentation: <a href="https://gcs-docs.s3.amazonaws.com/EVWHS/Miscellaneous/DevGuides/WMS/WMS_Map.htm?">Maxar WMS Developer Guide</a>.

```
size: 150                           # Radius of the bounding box in meters
width: 500                          # Width of the image in pixels
height: 500                         # Height of the image in pixels
srs: 'EPSG:4326'                    # Spatial Reference System        
transparent: True                   # Image transparency
request: 'GetMap'                   # Request should always be set to GetMap
format: 'image/geotiff'             # Output image's format.
layers: ['DigitalGlobe:Imagery']    # Outputs the raster data

exceptions: 'application/vnd.ogc.se_xml'
featureprofile: 'Most_Aesthetic_Mosaic_Profile'
digitalglobe_url: "https://evwhs.digitalglobe.com/mapservice/wmsaccess?"
```

<h3>Model Configurations</h3>

<b>Path:</b> `cnn_configs/` and `vit_configs/`

These directories contain YAML files that define the hyperparameters for training CNN (Convolutional Neural Network) and ViT (Vision Transformer) models, respectively.

For more information on hyperparameter tuning in PyTorch, refer to the official tutorial:
 <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html">PyTorch Optimization Guide</a>.

<b>Example</b>:
The file `cnn_configs/convnext_base.yaml` provides a detailed example of the configuration parameters for a CNN model.

```
beta: 2                 # The beta parameter for calculating the F-beta score (default is F-2 score)
test_size: 0.1          # The size of the test and validation set. Default train/val/test split is 0.8/0.1/0.1.
batch_size: 8           # Batch size
n_workers: 4            # Number of workers
n_epochs: 30            # Maximum number of epochs
scorer: "auprc"         # The primary performance metric (default is AUPRC).

model: "convnext_base"            # The model name  
type: "cnn"                       # The type of model (either 'cnn' or 'vit')
pretrained: True                  # Indicates whether model is pretrained on the Imagenet dataset
scheduler: "ReduceLROnPlateau"    # The learning rate scheduler
optimizer: "Adam"                 # Optimizer 
label_smoothing: 0.1              # Label smoothing parameter for regularization
lr: 0.00001                       # Initial learning rate (LR)
img_size: 224                     # Image size (all images are resized and center-cropped to this size)
step_size:                        # LR scheduler step size
patience: 7                       # The number of epochs to wait before decreasing the LR
lr_min: 0.0000001                 # The minimum LR used for early stopping
normalize: "imagenet"             # Normalizes the image by the mean and stddev of this dataset
lr_finder: False                  # Indicates whether to use the LR finder for finding the initial LR
```

<h2><a id="data-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Data</h2>

<b>Path:</b> <code>/data</code> 

This directory houses vector and raster datasets used for training data preparation and model development. 

Files are downloaded and processed via specific scripts:
- `notebooks/01_data_download.ipynb` or 
- `src/data_download.py`

```
.
├── ...
├── data                                
│   ├── rasters                   # Stores raster data (images, GeoTIFFs)            
│   │   ├── ghsl                  # Stores global GHSL rasters
│   │   ├── google_buildings      # Stores rasterized Google Open Buildings data
│   │   ├── maxar                 # Stores Maxar satellite images
│   │   └── ms_buildings          # Stores rasterized Microsoft Buildings data
│   │   
│   └── vectors                   # Stores vectors (GPKG, GeoJSON)                    
│       ├── <project_name>        # Stores training data vector files for a given project name
│       ├── google_buildings      # Stores the raw and merged Google Open Buildings vectors
│       └── ms_buildings          # Stores the raw and merged Microsoft Buildings vectors
│       
└── ...
```
<h3><a id="rasters" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Rasters</h3>

<h4>Global Human Settlements Layer (GHSL)</h4>

<b>Path:</b> `data/rasters/ghsl`

This folder contains two key spatial raster datasets:

- <a href="https://human-settlement.emergency.copernicus.eu/download.php?ds=builtC">GHSL-BUILT-C</a>:
A high-resolution dataset (10 m/pixel) that delineates the boundaries of human settlements.
  - Resolution: 10 m/pixel
  - CRS: EPSG:54009

- <a href="https://human-settlement.emergency.copernicus.eu/download.php?ds=smod">GHSL-SMOD</a>:
A classification layer that segments 1 km/pixel tiles by Degree of Urbanisation. This dataset is used to categorize schools into rural or urban subregions.
  - Resolution: 1 km/pixel
  - CRS: EPSG:54009

```
├── ...
├── data                                
│   └── rasters 
│       └── ghsl                    
│           ├── GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0.tif
│           └── GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V1_0.tif               
└── ...
```

<h4>Building Footprint Rasters</h4>

<b>Path:</b> `data/rasters/google_buildings` and `data/rasters/ms_buildings`

These directories store building footprint datasets from Google and Microsoft, rasterized for spatial analysis:

-  <a href="https://sites.research.google/gr/open-buildings/">Google Open Buildings</a>: Stored in `data/rasters/google_buildings` as `ISO_google.tiff`.
- <a href="">Microsoft Buildings</a>: Stored in `data/rasters/ms_buildings` as `ISO_ms.tiff`.

<h5>Image Specifications</h5>

- Resolution: 10 m/pixel GeoTIFFs.
- Rasterization Tool: Processed using GDAL.
- CRS: EPSG:3857.
```
├── ...
├── data                                
│   └── rasters 
│       ├── google_buildings                    
│       │   ├── <ISO>_google.tiff
│       │   └── ...
│       └── ms_buildings                    
│           ├── <ISO>_ms.tiff
│           └── ...
└── ...
```
<h4>Maxar Satellite Imagery</h4>

<b>Path:</b> `data/rasters/maxar`

This directory contains Maxar satellite images organized by country, identified using their ISO codes. These images are used for model training.

<h5>Default Image Specifications:</h5>

- <b>Dimensions</b>: 300x300 meters.
- <b>Resolution</b>: 500x500 pixels, with a ground resolution of 60 cm/pixel.
- <b>Centering</b>: Each image is centered on the latitude-longitude coordinates of a school or non-school sample.
- <b>Customization</b>:
The image size and spatial extent can be modified in the configuration file:
configs/sat_configs/sat_config.yaml.

<h5>File Naming Convention</h5>
Each file name encodes metadata about the source, country, class, and a unique identifier in the following format:

`<SOURCE>-<ISO>-<CLASS>-<UNIQUE_ID>.tiff`

- `<SOURCE>`: The data source (e.g., UNICEF, OSM, OVERTURE).
- `<ISO>`: The country’s ISO code.
- `<CLASS>`: Classification (SCHOOL or NON_SCHOOL).
- `<UNIQUE_ID>`: An 8-digit integer (e.g., 00000001).

<b>Example File Name</b>:

`UNICEF-MNG-SCHOOL-00000001.tiff`

```
├── ...
├── data                                
│   └── rasters 
│       └── maxar/500x500_60cm/<project_name>/<ISO>
│              ├── school
│              │     ├── UNICEF-<ISO>-SCHOOL-00000000.tiff
│              │     └── ...
│              └── non_school
│                     ├── UNICEF-<ISO>-NON_SCHOOL-00000000.tiff
│                     └── ...                     
└── ... 
``` 

<h3><a id="vectors" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Vectors</h3>

<h4>Building Footprint Vectors</h4>

<b>Path:</b> `data/vectors/google_buildings` and `data/rasters/ms_buildings` 

These directories contain the raw and merged datasets from <a href="https://sites.research.google/gr/open-buildings/">Google Open Buildings</a> and <a href="">Microsoft Buildings</a>. The vector datasets are available in both EPSG:4326 and EPSG:3857.
```
├── ...
├── data                                
│   └── vectors 
│       ├── google_buildings                        # Stores the raw and merged Google Open Buildings datasets                 
│       │   ├── <ISO>
│       │   │     ├── <raw_building_files>.csv.gz
│       │   │     └── ...
│       │   ├── <ISO>_google_EPSG4326.geojson
│       │   └── <ISO>_google_EPSG3857.geojson
│       │   
│       └── ms_buildings                            # Stores the raw and merged Microsoft Buildings datasets                     
│           ├── <ISO>
│           │     ├── <raw_building_files>.csv.gz
│           │     └── ...
│           ├── <ISO>_ms_EPSG4326.geojson
│           └── <ISO>_ms_EPSG3857.geojson
└── ...
```

<h4>Project Files</h4>

<b>Path:</b> `data/vectors/<project_name>`

This directory contains the vector files used for model development and evaluation.

<h5>Geoboundaries</h5>

<b>Path:</b> `data/vectors/<project_name>/geoboundaries/`

This directory contains geoboundary datasets for individual countries, organized by administrative levels (e.g., ADM0, ADM1, ADM2, etc.). These datasets are sourced from the <a href="https://www.geoboundaries.org/">GeoBoundaries project</a>.

<b>Example</b>

`data/vectors/<project_name>/geoboundaries/SEN_ADM3_geoboundary.geojson` 

- Stores the geoboundaries for Senegal at administrative level 3.


<h5>School Data</h5>

<b>Path:</b> `data/vectors/<project_name>/school/`

This directory stores school datasets for individual countries, organized by data source.

<b>Subdirectories</b>:
- `unicef/`
  -  Stores school data sourced from <a href="https://maps.giga.global/">GigaMaps</a>. 
  - e.g. `SEN_unicef.geojson` (Senegal school data sourced from UNICEF).
- `osm/`
  - Stores school data sourced from <a href="https://www.openstreetmap.org/">OpenStreetMap (OSM)</a>. 
  - e.g. `SEN_osm.geojson` (Senegal school data sourced from OSM).
- `overture/`
  - Stores school data sourced from <a href="https://overturemaps.org/">Overture Maps</a>. 
  - e.g. `SEN_overture.geojson` (Senegal school data sourced from Overture Maps).
- `clean/`
  - Stores the combined and cleaned data from GigaMaps, OSM, and Overture Maps. 
  - See <a href="#cleaned-data">Clean Data</a> for more information.

<h5>Non-school Data</h5>

<b>Path:</b> `data/vectors/<project_name>/non_school/` 

This directory stores non-school datasets for individual countries, organized by data source.

<b>Subdirectories</b>:
- `osm/`
  - Stores non-school data sourced from <a href="https://www.openstreetmap.org/">OpenStreetMap (OSM)</a>. 
  - e.g. `SEN_osm.geojson` (Senegal non-school data sourced from OSM).
- `overture/`
  - Stores non-school data sourced from <a href="https://overturemaps.org/">Overture Maps</a>. 
  - e.g. `SEN_overture.geojson` (Senegal non-school data sourced from Overture Maps).
- `clean/`
  - Stores the combined and cleaned data from OSM, and Overture Maps. 
  - See <a href="#cleaned-data">Clean Data</a> for more information.

<h5><a id="cleaned-data" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Clean Data</h5>

<b>Path:</b> `data/vectors/<project_name>/clean/<ISO_clean>.geojson`

This file integrates data from GigaMaps (for schools), OpenStreetMap (OSM), and Overture Maps. The dataset is automatically generated when `src/data_preprocess.py` is executed.

<b>Additional Columns Added During Automated Data Cleaning</b>:
- `clean`: Classifies each data point based on its validity:
  - `0`: Valid point.
  - `1`: Contains a keyword in the keyword exclusion list (see `configs/config.yaml`)
  - `2`: Duplicate of another school location (i.e., within the vicinity of an existing school point).
  - `3`: Located in an unpopulated/uninhabite/invalid area.

<h5>Manual Cleaning</h5>

Manual validation is performed using the notebook, `notebooks/03_sat_cleaning.ipynb`. This process updates the file `clean/<ISO_clean>.geojson` by adding a new column, `validated`, and typically involves removing points where schools are not visible in satellite imagery or are indistinguishable from surrounding structures.

<b>Additional Columns Added During Manual Data Cleaning</b>:
- `validated`: Labels each point based on its inclusion in the training dataset:
  - `0`: Included in the training dataset.
  - `-1`: Excluded from the training dataset.

<h5>Training Data</h5> 

<b>Path:</b> `data/vectors/<project_name>/train/<ISO>_train.geojson`

This file combines the cleaned school and non-school datasets and is automatically generated when `src/train_model.py` is executed.

<b>Additional Columns in Training Data</b>

- `dataset`: Indicates the subset of the data (`train`, `val`, or `test`) to which a sample belongs.
- `rurban`: Specifies whether the sample is in a `rural` or `urban` area, determined using the GHSL-SMOD classification.

```
├── ...
├── data                                
│   └── vectors 
│       ├── <project_name>                    
│       │   ├── geoboundaries                           # Stores the geoboundaries for a country
│       │   │     │                                     # given its ISO code, from https://www.geoboundaries.org/
│       │   │     ├── <ISO>_ADM0_geoboundary.geojson
│       │   │     ├── <ISO>_ADM1_geoboundary.geojson
│       │   │     └── ...
│       │   ├── school                                  # Stores the raw and processed files used to generate
│       │   │      │                                    # the training data for the positive class
│       │   │      ├── unicef                           # Stores school data downloaded from GigaMaps
│       │   │      │   ├──<ISO>_unicef.geojson
│       │   │      │   └── ...
│       │   │      ├── osm                              # Stores school data downloaded from OpenStreetMap (OSM)
│       │   │      │   ├──<ISO>_osm.geojson
│       │   │      │   └── ...
│       │   │      ├── overture                         # Stores school data downloaded from Overture Maps
│       │   │      │   ├──<ISO>_overture.geojson
│       │   │      │   └── ...
│       │   │      └── clean                            # Combines school data from GigaMaps, OSM, and Overture Maps
│       │   │          ├──<ISO>_clean.geojson
│       │   │          └── ...
│       │   ├── non_school                              # Stores the raw and processed files used to generate 
│       │   │      │                                    # the training data for the negative class
│       │   │      ├── osm                              # Stores non-school data downloaded from OpenStreetMap
│       │   │      │   ├──<ISO>_osm.geojson
│       │   │      │   └── ...
│       │   │      ├── overture                         # Stores non-school data downloaded from Overture Maps
│       │   │      │   ├──<ISO>_overture.geojson
│       │   │      │   └── ...
│       │   │      └── clean                            # Combines non-school data from GigaMaps, OSM, and Overture Maps
│       │   │          ├──<ISO>_clean.geojson
│       │   │          └── ...
│       │   └── train
│       │         ├── <ISO>_train.geojson
│       │         └── ...
└── ...
```

<h2><a id="exp-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Experiments</h2>

<b>Path:</b> `/exp`

This directory stores the experiment outputs for each country and model combination within a specific project.

<h5>File Naming Convention</h5>
Each experiment subdirectory encodes the project, country, and model name in the following format:

`<project_name>/<ISO>_<model_name>`

- `<project_name>`: The name of the project (e.g. GIGAv1)
- `<ISO>`: The country's ISO code
- `<model_name>`: The name of the model (e.g. `convnext_small`, `vit_h_14`)

<b>Example</b>

`exp/GIGAv1/SEN_convnext_small/` 

- Stores the outputs for the `convnext_small` model trained on Senegal data in the GIGAv1 project.

Each experiment folder will contain the following files and subdirectories:

- `<ISO>_<model_name>.log`
  - Logs the command-line output generated during the experiment run.
-  `<ISO>_<model_name>.pth`
    - Stores the trained model.
- `<ISO>_<model_name>_test.csv`
    - Stores the test set results
- `<ISO>_<model_name>_val.csv`
    - Stores the validation set results

```
├── ...
├── exp                                
│   └── <project_name> 
│       ├── <ISO>_<model_name>                   
│       │   ├── <ISO>_<model_name>.log
│       │   ├── <ISO>_<model_name>.pth
│       │   ├── <ISO>_<model_name>_test.csv
│       │   └── <ISO>_<model_name>_val.csv
└── ...
```

<b>Note:</b> We recommend running `notebooks/05_model_evaluation.ipynb` to generate the final model performance results.

<h2><a id="output-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Outputs</h2>

<b>Path:</b> `/output` 

This directory stores all the files generated for nationwide model deployment.

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── geotiff         # Stores georeferenced satellite images              
│       ├── images          # Stores the raw satellite images  
│       ├── results         # Stores the tile and CAM results 
│       └── tiles           # Stores the tiles used for downloading satellite images 
└── ...
```

<h3> Nationwide Tiles </h3>

<b>Path:</b> `/outputs/<ISO>/tiles`

This directory contains vector tiles filtered to inhabited areas using settlement datasets such as GHSL, Microsoft, and Google. These tiles are used as inputs for downloading satellite images.

<h5>File Naming Convention</h5>
Each vector file encodes the name of the country and adminstrative level 2 (ADM2) name in the following format:

`<ISO>_<ADM2>.geojson`

- `<ISO>`: The country's ISO code
- `<ADM2>`: The administrative level 2 name

<b>Example</b>

`TJK_Asht District.geojson`

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── tiles
│       │     ├── <ISO>_<ADM2>.geojson
│       │     └── ...
└── ...
```

<h3>Satellite Images</h3>

<b>Path:</b> `outputs/<ISO>/images`

This directory stores satellite images corresponding to the tiles in /tiles, downloaded from Maxar. The satellite images in this directory may or may not be georeferenced (EPSG:4326).

<b>Organization:</b>
- Images are grouped by administrative level 2 names.
- Each image is named using its unique identifier (UID).

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── images
│       │   └── <ADM2>
│       │         ├── <UID>.tiff
│       │         └── ...
└── ...
```

<h3>Georeferenced Satellite Images for CAM Generation</h3>

<b>Path:</b> `outputs/<ISO>/geotiff`

This directory contains georeferenced satellite images in GeoTIFF format with a CRS of EPSG:3857.

- Includes images where the VSC-ensemble model confidently predicts the presence of a school. 
- Confidence is determined by a predicted probability exceeding the threshold that maximizes the F2 score on the validation set.

<b>Organization:</b>
- Images are grouped by administrative level 2 names.
- Each image is named using its unique identifier (UID).


```
├── ...
├── output                             
│   └── <ISO> 
│       ├── geotiff                  
│       │   └── <ADM2>
│       │         ├── <UID>.tiff
│       │         └── ...
└── ...
```
<h3>Model Deployment Results</h3>

<b>Path:</b> `/output/<ISO>/results/<project_name>`

This directory stores the nationwide model deployment results for each country. This directory contains two subdirectories:

- `tiles`
  - Stores the model outputs at the image or tile-level
- `cams`
  - Stores the class activation map (CAM) outputs

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── results
│       │   └── <project_name> 
│       │         ├── cams
│       │         └── tiles
└── ...
```

<h4>Image or Tile-level Outputs</h4>

<b>Path:</b> `/output/<ISO>/results/<project_name>/tiles`

This directory stores the model outputs for each of the best-performing model variants for ConvNext, ViT, and Swin. This directory also stores the combined outputs from these architectures, i.e. ensemble model.

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── results
│       │   └── <project_name> 
│       │         └── tiles
│       │               ├── ensemble
│       │               │     ├── <ISO>_<ADM2>.geojson
│       │               │     └── ...
│       │               ├── <best_convnext_model>
│       │               │     ├── <ISO>_<ADM2>.geojson
│       │               │     └── ...
│       │               ├── <best_vit_model>
│       │               │     ├── <ISO>_<ADM2>.geojson
│       │               │     └── ...
│       │               └── <best_swin_model>
│       │                     ├── <ISO>_<ADM2>.geojson
│       │                     └── ...
└── ...
```

<h4>CAM-level Outputs</h4>

<b>Path:</b> `/output/<ISO>/results/<project_name>/cams`

This directory contains Class Activation Map (CAM) outputs, which highlight regions of satellite images most relevant to the model's predictions.

Files include GeoJSON outputs for administrative level 2 (ADM2) regions.

```
├── ...
├── output                             
│   └── <ISO> 
│       ├── results
│       │   └── <project_name> 
│       │         └── cams
│       │              ├── ensemble/<best_model>/<best_cam_method>
│       │              │     ├── <ISO>_<ADM2>.geojson
│       │              │     └── ...
│       │              └── <ISO_<best_model>_<best_cam_method>.geojson
└── ...
```

<h4>Final output</h4>

The final results includes the merged admin2 level outputs and is stored in:

`/output/<ISO>/results/<project_name>/cams/<ISO_<best_model>_<best_cam_method>.geojson`