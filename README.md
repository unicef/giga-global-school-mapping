<div align="center">

# UNICEF-Giga: Global School Mapping 

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
|
<b><a href="#-usage">Usage</a></b>
|
<b><a href="#-file-organization">File Organization</a></b>
|
<b><a href="#acknowledgement">Acknowledgment</a></b>
|
<b><a href="#citation">Citation</a></b>
</p>

</div>

## 📜 Description
This work leverages deep learning and high-resolution satellite images for automated school mapping. This work is developed under Giga, a global initiative by UNICEF-ITU to connect every school to the internet by 2030.

Obtaining complete and accurate information on schools locations is a critical first step to accelerating digital connectivity and driving progress towards SDG4: Quality Education. However, precise GPS coordinate of schools are often inaccurate, incomplete, or even completely non-existent in many developing countries.  In support of the Giga initiative, we leverage machine learning and remote sensing data to accelerate school mapping. This work aims to support government agencies and connectivity providers in improving school location data to better estimate the costs of digitally connecting schools and plan the strategic allocation of their financial resources.

<p>
<img src="./assets/workflow.png" width="80%" height="80%" />

This code accompanies the following paper(s):
- Doerksen, K., Tingzon, I., and Kim, D. (2024). AI-powered school mapping and connectivity status prediction using Earth observation. ICLR 2024 Machine Learning for Remote Sensing (ML4RS) Workshop.

## 📂 Dataset
For each school and non-school location in our dataset, we downloaded 300 x 300 m, 500 x 500 px high-resolution satellite images from Maxar with a spatial resolution of 60 cm/px. 

## 💻 Code Organization 
This repository is divided into the following files and folders:
- **notebooks/**: contains all Jupyter notebooks for exploratory data analysis and model prediction.
- **utils/**: contains utility methods for loading datasets, building model, and performing training routines.
- **src/**: contains scripts runnable scripts for automated data cleaning and model training/evaluation.

## 💻 Usage

### Setup
```sh
conda create -n envname
conda activate envname
pip install -r requirements.txt
```

### Data Download 
To download the relevant datasets, run:
```s
python data_download.py \
--config="configs/<DATA_CONFIG_FILE_NAME>.yaml"
```

### Data Preparation
To run the data cleaning pipeline:
```s
python data_preparation.py \
--config="configs/<DATA_CONFIG_FILE_NAME>.yaml"
```

### Model Training
To train the CNN model, run:
```s
python train_cnn.py \
--cnn_config="configs/cnn_configs/<CNN_CONFIG_FILE_NAME>.yaml" \
--iso="<ISO_CODE>"
```

To run the ViT-based models:
```s
python train_model.py \
--cnn_config="configs/model_configs/<MODEL_CONFIG_FILE_NAME>.yaml" \
--iso="<ISO_CODE>"
```

### Model Prediction
For model prediction, run:
```s
python sat_predict.py \
--data_config="configs/<DATA_CONFIG_FILE_NAME>.yaml" \
--model_config="configs/cnn_configs/<CNN_CONFIG_FILE_NAME>.yaml" \
--sat_config="configs/sat_configs/<SAT_CONFIG_FILE_NAME>.yaml" \
--sat_creds="configs/sat_configs/<SAT_CREDENTIALS_FILE_NAME>.yaml" \
--iso="<ISO_CODE>"
```

## 📂 File Organization 
The datasets are organized as follows:
```
data
├── rasters
│   ├── maxar
│   │   ├── ISO
│   │   │   ├── school
│   │   │   │    ├── UNICEF-ISO-SCHOOL-00000001.tiff
│   │   │   │    └── ...
│   │   │   ├── non_school
│   │   │   │    ├── UNICEF-ISO-NON_SCHOOL-00000001.tiff
│   │   │   │    └── ...
│   │   │   └── ...
│   │   └── ...
└── vectors
    ├── school
    │   ├── unicef
    │   │   ├──ISO_school_geolocation_coverage_master.csv
    │   │   └── ...
    │   ├── osm
    │   │   ├──ISO_osm.geojson
    │   │   └── ...
    │   ├── overture
    │   │   ├──ISO_overture.geojson
    │   │   └── ...
    └── non_school
        ├── osm
        │   ├──ISO_osm.geojson
        │   └── ...
        └── overture
            ├──ISO_overture.geojson
            └── ...
    
```

## Acknowledgment
Global high-resolution satellite images (60 cm/px) from Maxar made available with the generous support of the US State Department. We are also grateful to Dell for providing us with access to High Performance Computing (HPC) clusters with NVIDIA GPU support. 

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{doerksen2024aipowered,
  title={AI-powered school mapping and connectivity status prediction using Earth Observation},
  author={Doerksen, Kelsey and Tingzon, Isabelle and Kim, Ho-Hyung},
  year={2024}
}
```
