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
    <ul>
    <li><a href="#rasters">Rasters</a></li>
    <li><a href="#vectors">Vectors</a></li>
  </ul>
  <li><a href="#exp-dir">Experiments</a></li>
  <li><a href="#notebook-dir">Notebooks</a></li>
  <li><a href="#output-dir">Outputs</a></li>
</details>

<h2><a id="overview" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Overview</h2>

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

The `./configs` directory contains the various configuration files for training data preparation model development, and model deployment. 

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
#### Configuration Files
`config.yaml` is a YAML file that contains the global configuration parameters, descibed as follows. For brevity, some values may be omitted or truncated. 

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

`data_configs/` contains the YAML files for training data preparation and model experimentation. See below for an example of the data config file for Tajikistan, `data_configs/data_config_ISO_TJK.yaml`.

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

`sat_configs/sat_creds.yaml` contains the credentials used for logging into <a href="https://evwhs.digitalglobe.com/myDigitalGlobe/login">https://evwhs.digitalglobe.com</a>.
```
username:     # Maxar username
password:     # Maxar password
connect_id:   # Maxar connect_id (see profile >> Connect ID)
```

`sat_configs/sat_config.yaml` contains the parameters for Maxar satellite image download. The default parameters are listed below. For more information, see <a href="https://gcs-docs.s3.amazonaws.com/EVWHS/Miscellaneous/DevGuides/WMS/WMS_Map.htm?">https://gcs-docs.s3.amazonaws.com/EVWHS/Miscellaneous/DevGuides/WMS/WMS_Map.htm?</a>.

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

The YAML files in `cnn_configs/` and `vit_configs/` contain the hyperparameters for CNN and ViT models, correspondingly. The following example lists the configuration paramters in `cnn_configs/convnext_base.yaml`. To learn more about model hyperparameter tuning in Pytorch, see: <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html">https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html</a>.

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

The `./data` directory contains all the vector and raster datasets used for training data preparation and model development.  

```
.
├── ...
├── data                                
│   ├── rasters                   # Contains raster data (images, GeoTIFFs)            
│   │   ├── ghsl                  # Contains global GHSL rasters
│   │   ├── google_buildings      # Contains rasterized Google Open Buildings data
│   │   ├── maxar                 # Contains Maxar satellite images
│   │   └── ms_buildings          # Contains rasterized Microsoft Buildings data
│   │   
│   └── vectors                   # Contains vectors (GPKG, GeoJSON)                    
│       ├── <project_name>        # Contains training data vector files for a given project name
│       ├── google_buildings      # Contains the raw and merged Google Open Buildings vectors
│       └── ms_buildings          # Contains the raw and merged Microsoft Buildings vectors
│       
└── ...
```
<h3><a id="rasters" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Rasters</h3>

All files in thid directory are automatically downloaded and processed when you run either `notebooks/01_data_download.ipynb` or `src/data_download.py`. 

#### Global Human Settlements Layer (GHSL) 
The `./data/rasters/ghsl` folder contains primarily two files: 
- <a href="https://human-settlement.emergency.copernicus.eu/download.php?ds=builtC">GHSL-Built-C </a>: The spatial raster dataset delineates the boundaries of the human settlements at 10m resolution.
  - CRS: World Mollweide (EPSG:54009) 
  - Resolution: 10m
- <a href="https://human-settlement.emergency.copernicus.eu/download.php?ds=smod">GHSL-SMOD</a>: The layer classifies 1km-resolution tiles by Degree of Urbanisation and is used to classify schools as belonging to a rural or urban subregion.
  - CRS: World Mollweide (EPSG: 54009) 
  - Resolution: 1km

```
├── ...
├── data                                
│   └── rasters 
│       └── ghsl                    
│           ├── GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0.tif
│           └── GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V1_0.tif               
└── ...
```

#### Building Footprint Rasters | Microsoft and Google
- `./data/rasters/google_buildings` contains the <a href="https://sites.research.google/gr/open-buildings/">Google Open Buildings</a> dataset rasterized to a 10m resolution GeoTIFF using GDAL. 
  - CRS: EPSG:3857
  - Resolution: 10m

- `./data/rasters/ms_buildings` contains the <a href="">Microsoft Buildings</a> dataset rasterized to a 10m resolution GeoTIFF using GDAL. 
  - CRS: EPSG:3857
  - Resolution: 10m
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
##### Maxar Satellite Images
`./data/rasters/maxar` contains the Maxar satellite images by country (i.e. ISO code) used for model training. By default, all images are 300x300 m, 500x500 px satellite images with a resolution of 60 cm/px. The image size and extent can be adjusted in `configs/sat_configs/sat_config.yaml`. 

The datasets are arranged by school and non-school subsets as follows:
```
├── ...
├── data                                
│   └── rasters 
│       └── maxar                    
│           └── 500x500_60cm
│                 └── <project_name>
│                         └── <ISO>
│                               ├── school
│                               │     ├── UNICEF-<ISO>-SCHOOL-00000000.tiff
│                               │     └── ...
│                               └── non_school
│                                     ├── UNICEF-<ISO>-NON_SCHOOL-00000000.tiff
│                                     └── ...
│                         
└── ... 
``` 

<h3><a id="vectors" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Vectors</h3>

##### Building Footprint Vectors | Microsoft and Google
`./data/vectors/google_buildings` and `./data/rasters/ms_buildings` contain the raw and merged datasets from <a href="https://sites.research.google/gr/open-buildings/">Google Open Buildings</a> and <a href="">Microsoft Buildings</a>, both in EPSG:4326 and EPSG:3857.
```
├── ...
├── data                                
│   └── vectors 
│       ├── google_buildings                        # Contains the raw and merged Google Open Buildings datasets                 
│       │   ├── <ISO>
│       │   │     ├── <raw_building_files>.csv.gz
│       │   │     └── ...
│       │   ├── <ISO>_google_EPSG4326.geojson
│       │   └── <ISO>_google_EPSG3857.geojson
│       │   
│       └── ms_buildings                            # Contains the raw and merged Microsoft Buildings datasets                     
│           ├── <ISO>
│           │     ├── <raw_building_files>.csv.gz
│           │     └── ...
│           ├── <ISO>_ms_EPSG4326.geojson
│           └── <ISO>_ms_EPSG3857.geojson
└── ...
```

#### Project Files
Under `./data/vectors/<project_name>`, we generate the following directories:

```
├── ...
├── data                                
│   └── vectors 
│       ├── <project_name>                    
│       │   ├── geoboundaries                           # Contains the geoboundaries for a country
│       │   │     │                                     # given its ISO code, from https://www.geoboundaries.org/
│       │   │     ├── <ISO>_ADM0_geoboundary.geojson
│       │   │     ├── <ISO>_ADM1_geoboundary.geojson
│       │   │     └── ...
│       │   ├── school                                  # Contains the raw and processed files used to generate
│       │   │      │                                    # the training data for the positive class
│       │   │      ├── unicef                           # Contains school data downloaded from GigaMaps
│       │   │      │   ├──<ISO>_unicef.geojson
│       │   │      │   └── ...
│       │   │      ├── osm                              # Contains school data downloaded from OpenStreetMap (OSM)
│       │   │      │   ├──<ISO>_osm.geojson
│       │   │      │   └── ...
│       │   │      ├── overture                         # Contains school data downloaded from Overture Maps
│       │   │      │   ├──<ISO>_overture.geojson
│       │   │      │   └── ...
│       │   │      ├── clean                            # Combines school data from GigaMaps, OSM, and Overture Maps
│       │   │      │   ├──<ISO>_clean.geojson
│       │   │      │   └── ...
│       │   ├── non_school                              # Contains the raw and processed files used to generate 
│       │   │      │                                    # the training data for the negative class
│       │   │      ├── osm                              # Contains non-school data downloaded from OpenStreetMap
│       │   │      │   ├──<ISO>_osm.geojson
│       │   │      │   └── ...
│       │   │      └── overture                         # Contains non-school data downloaded from Overture Maps
│       │   │          ├──<ISO>_overture.geojson
│       │   │          └── ...
│       │   └── train
│       │         ├── <ISO>_train.geojson
│       │         └── ...
└── ...
```
The file `<ISO_clean>.geojson` under `clean/` combines the data from GigaMaps, OSM, and Overture Maps. The data cleaning process will add the  following additional columns as follows:
- the `clean` column with the following values per sample:
  - 0: valid point
  - 1: point is in a unpopulated/uninhabited area
  - 2: point is within vicinity of another school location (i.e. duplicate)


The file `<ISO>_train.geojson` under `train/` combines the clean school and non-school datasets with additional columns as follows:
  - the `datasets` column indicates whether the sample belongs to the train, val, or test set
  - the `rurban` column indicates whether the sample belongs to a rural or urban area, based on GHSL-SMOD classification.

<h2><a id="exp-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Experiments</h2>

All experiments are saved to the `./exp` directory.

```
├── ...
├── exp                                
│   └── <project_name> 
│       ├── <ISO>_<model_name>                   
│       │   ├── <ISO>_<model_name>.log
│       │   ├── <ISO>_<model_name>.pth
│       │   ├── <ISO>_<model_name>_test.csv
│       │   ├── <ISO>_<model_name>_val.csv
│       │   ├── cm_metrics.csv
│       │   ├── cm_report.log
│       │   ├── confusion_matrix.csv
│       │   └── results.json
└── ...
```

<h2><a id="output-dir" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Outputs</h2>