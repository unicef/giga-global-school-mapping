#!/bin/bash

echo -n "Input ISO: "
read iso

python src/train_model.py --config="configs/vit_configs/satlas-aerial_swinb_mi.yaml" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/satlas-aerial_swinb_si.yaml" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/swin_v2_b.yaml" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/vit_b_16.yaml" --iso=$iso; 