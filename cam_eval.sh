#!/bin/bash

echo -n "Input ISO: "
read iso

python src/cam_evaluate.py --iso_code=$iso --model_config="configs/vit_configs/vit_h_14.yaml"