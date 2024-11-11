#!/bin/bash

echo -n "Input ISO: "
read iso

python src/train_model.py --config="configs/cnn_configs/convnext_small.yaml" --iso=$iso --lr_finder=False; 
python src/train_model.py --config="configs/cnn_configs/convnext_base.yaml" --iso=$iso --lr_finder=False;
python src/train_model.py --config="configs/cnn_configs/convnext_large.yaml" --iso=$iso --lr_finder=False;
python src/train_model.py --config="configs/vit_configs/vit_b_16.yaml" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/vit_l_16.yaml" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/vit_h_14.yaml" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/swin_v2_t.yaml" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/swin_v2_s.yaml" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/swin_v2_b.yaml" --iso=$iso;