<div style="padding-left: 20px; padding-right: 10px;">
<a href="https://giga.global/">
    <img src="https://s41713.pcdn.co/wp-content/uploads/2018/11/2020.05_GIGA-visual-identity-guidelines_v1-25.png" alt="Giga logo" title="Giga" align="right" height="60" style="padding-top: 10px;"/>
</a>

<div style="padding-top: 20px;"> </div>
<h1><a id="giga-ai" class="anchor" aria-hidden="true" href="#giga-ai"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
UNICEF Giga: AI-enabled School Mapping  </h1> 

<div align="center" >

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![GitHub repo size](https://img.shields.io/github/repo-size/unicef/giga-global-school-mapping)
![GitHub stars](https://img.shields.io/github/stars/unicef/giga-global-school-mapping)
![Twitter Follow](https://img.shields.io/twitter/follow/gigaglobal)
<!---![License](https://img.shields.io/github/license/unicef/giga-giga-global-school-mapping)--->


</div>

<details open="open">
	<summary style="padding-bottom: 10px;"><h2>Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-giga">About Giga</a></li>
    <li><a href="#about">About</a>
    <ul>
        <li><a href="#project-objective">Project Objective</a></li>
        <li><a href="#flow-diagram">System Flow Diagram</a></li>
        <li><a href="#giga-repos">Github Repositories</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
      </li>
	<li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#contribution-guidelines">Contribution Guidelines</a></li>
    <li><a href="#code-design">Code Design</a></li>
    <li><a href="#code-of-conduct">Code of Conduct</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<h2><a id="about-giga" class="anchor" aria-hidden="true" href="#about-giga"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>About Giga</h2>

Giga is a UNICEF-ITU global initiative to connect every school to the Internet and every young person to information, opportunity, and choice. By connecting all schools to the Internet, we ensure that every child has a fair shot at success in an increasingly digital world.

<h2><a id="about" class="anchor" aria-hidden="true" href="#about"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
About</h2>

This work leverages deep learning and high-resolution satellite images for automated school mapping and is developed under Giga, a global initiative by UNICEF-ITU to connect every school to the internet by 2030.

Obtaining complete and accurate information on schools locations is a critical first step to accelerating digital connectivity and driving progress towards SDG4: Quality Education. However, precise GPS coordinate of schools are often inaccurate, incomplete, or even completely non-existent in many developing countries.  In support of the Giga initiative, we leverage computer and remote sensing data to accelerate school mapping. This work aims to support government agencies and connectivity providers in improving school location data to better estimate the costs of digitally connecting schools and plan the strategic allocation of their financial resources.

<h3><a id="project-objective" class="anchor" aria-hidden="true" href="#project-objective"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Project Objective</h3>

- Present a publicly available, end-to-end pipeline for automated school location detection from high-resolution satellite images.
- Help governments improve the quality of school location information in their national register. 
- Identify new, previously unmapped schools in way that is quick, efficient, and scalable.


<h3><a id="flow-diagram" class="anchor" aria-hidden="true" href="#flow-diagram"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
System Flow Diagram </h3>
For each school and non-school location in our dataset, we downloaded 300 x 300 m, 500 x 500 px high-resolution satellite images from Maxar with a spatial resolution of 60 cm/px. 

<h3><a id="giga-repos" class="anchor" aria-hidden="true" href="#giga-repos"><svg class="octicon octicon-link" align="center" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Github Repositories </h3>

- [Giga AI-enabled School Mapping](https://github.com/unicef/giga-global-school-mapping/)
- [Giga AI School Validation](https://github.com/unicef/giga-school-mapping-validator)


<h3><a id="built-with" class="anchor" aria-hidden="true" href="#built-with"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16"gig aria-hidden="true"></path></svg></a>
Built With </h3>

- <b>ML/DL Frameworks:</b>  Scikit-learn, Pytorch
- <b>Programming Language:</b>  Python
- <b>Geospatial Libraries:</b>  GeoPandas, Rasterio, Fiona, GDAL


<h2><a id="getting-started" class="anchor" aria-hidden="true" href="#getting-started"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Getting Started</h2>


### Setup
```sh
conda create -n envname
conda activate envname
pip install -r requirements.txt
```

### Data Download 
To download the relevant datasets, run `python src/data_download.py`:
```s
usage: data_download.py [-h] [--config CONFIG] [--profile PROFILE]

Data Download
options:
  -h, --help         show this help message and exit
  --config CONFIG    Path to the configuration file
  --profile PROFILE  Path to the profile file
```

### Data Preparation
To run the data cleaning pipeline, run `python src/data_preparation.py`:
```s
usage: data_preparation.py [-h] [--config CONFIG] [--name NAME] 
[--sources SOURCES [SOURCES ...]] [--clean_pos CLEAN_POS] [--clean_neg CLEAN_NEG]

Data Cleaning Pipeline
options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration file
  --name NAME           Folder name
  --sources SOURCES [SOURCES ...] Sources (e.g. unicef, osm, overture)
  --clean_pos CLEAN_POS Clean positive samples (Boolean indicator)
  --clean_neg CLEAN_NEG Clean negative samples (Boolean indicator)
```

### Satellite Image Download
To download Maxar satellite images, run `python src/sat_download.py`:
```s
usage: sat_download.py [-h] [--config CONFIG] [--creds CREDS] 
[--category CATEGORY] [--iso_code ISO_CODE] [--filename FILENAME]

Satellite Image Download
options:
  -h, --help           show this help message and exit
  --config CONFIG      Path to the configuration file
  --creds CREDS        Path to the credentials file
  --category CATEGORY  Category (e.g. school or non_school)
  --iso_code ISO_CODE  ISO 3166-1 alpha-3 code
  --filename FILENAME  Filename of data (optional)
```

### Model Training
To train the computer vision models, run `python src/train_cnn.py`:
```s
usage: train_cnn.py [-h] [--cnn_config CNN_CONFIG] [--lr_finder LR_FINDER] [--iso ISO [ISO ...]]

Model Training
options:
  -h, --help              show this help message and exit
  --cnn_config CNN_CONFIG Path to the configuration file
  --lr_finder LR_FINDER   Learning rate finder (Boolean indicator)
  --iso ISO [ISO ...]     ISO 3166-1 alpha-3 codes
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

<h2><a id="contribution-guidelines" class="anchor" aria-hidden="true" href="#contribution-guidelines"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Contribution Guidelines</h2>

Thank you for considering contributing to Giga! We value your input and aim to make the contribution process as accessible and transparent as possible. Whether you're interested in reporting bugs, discussing code, submitting fixes, proposing features, becoming a maintainer, or engaging with the Giga community, we welcome your involvement. 

[Click here for detailed Contribution Guidelines](https://github.com/unicef/https://github.com/unicef/giga-global-school-mapping/blob/master/Contribution-Guidelines.md) 

<h2><a id="code-design" class="anchor" aria-hidden="true" href="#code-design"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Code Design</h2>

This repository is divided into the following files and folders:
- **notebooks/**: contains all Jupyter notebooks for exploratory data analysis and model prediction.
- **utils/**: contains utility methods for loading datasets, building model, and performing training routines.
- **src/**: contains scripts runnable scripts for automated data cleaning and model training/evaluation.

### File Organization 
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
    │   │   ├──ISO_unicef.geojson
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

<h2><a id="code-of-conduct" class="anchor" aria-hidden="true" href="#code-of-conduct"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Code of Conduct</h2>

At Giga, we're committed to maintaining an environment that's respectful, inclusive, and harassment-free for everyone involved in our project and community. We welcome contributors and participants from diverse backgrounds and pledge to uphold the standards. 

[Click here for detailed Code of Conduct](https://github.com/unicef/https://github.com/unicef/giga-global-school-mapping/blob/master/CODE_OF_CONDUCT.md) 

<h2><a id="contact" class="anchor" aria-hidden="true" href="#contact"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Contact</h2>

Applied Science AI-enabled School Mapping Team:
- Isabelle Tingzon: itingzon@unicef.com 
- Ivan Dotu Rodriguez: jdoturodriguez@unicef.com 

Giga Website: https://giga.global/contact-us/ 


<h2><a id="acknowledgements" class="anchor" aria-hidden="true" href="#acknowledgements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
Acknowledgments💜</h2> 
Global high-resolution satellite images (60 cm/px) from Maxar made available with the generous support of the US State Department. We are also grateful to Dell for providing us with access to High Performance Computing (HPC) clusters with NVIDIA GPU support. 