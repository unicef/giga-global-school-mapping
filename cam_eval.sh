#!/bin/bash

python src/cam_evaluate.py --iso_code="SEN" --model_config="configs/vit_configs/vit_h_14.yaml" --percentile=90
python src/cam_evaluate.py --iso_code="BWA" --model_config="configs/vit_configs/vit_l_16.yaml" --percentile=90
python src/cam_evaluate.py --iso_code="SSD" --model_config="configs/vit_configs/vit_h_14.yaml" --percentile=90