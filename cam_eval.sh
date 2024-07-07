#!/bin/bash

#python src/cam_evaluate.py --iso_code="SSD" --model_config="configs/vit_configs/vit_h_14.yaml" --percentile=90
#python src/cam_evaluate.py --iso_code="BWA" --model_config="configs/vit_configs/vit_l_16.yaml" --percentile=90
#python src/cam_evaluate.py --iso_code="SEN" --model_config="configs/vit_configs/vit_h_14.yaml" --percentile=90
#python src/cam_evaluate.py --iso_code="RWA" --model_config="configs/vit_configs/vit_h_14.yaml" --percentile=90
python src/cam_evaluate.py --iso_code="NAM" --model_config="configs/vit_configs/swin_v2_s.yaml" --percentile=90
python src/cam_evaluate.py --iso_code="BEN" --model_config="configs/vit_configs/swin_v2_s.yaml" --percentile=90
python src/cam_evaluate.py --iso_code="GHA" --model_config="configs/vit_configs/swin_v2_s.yaml" --percentile=90
