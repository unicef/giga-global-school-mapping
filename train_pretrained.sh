#!/bin/bash

echo -n "Input ISO: "
read iso

python src/train_model.py --config="configs/cnn_configs/convnext_small.yaml" --pretrained="AF" --iso=$iso; 
python src/train_model.py --config="configs/cnn_configs/convnext_base.yaml" --pretrained="AF" --iso=$iso;
python src/train_model.py --config="configs/cnn_configs/convnext_large.yaml" --pretrained="AF" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/vit_b_16.yaml" --pretrained="AF" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/vit_l_16.yaml" --pretrained="AF" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/vit_h_14.yaml" --pretrained="AF" --iso=$iso; 
python src/train_model.py --config="configs/vit_configs/swin_v2_t.yaml" --pretrained="AF" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/swin_v2_s.yaml" --pretrained="AF" --iso=$iso;
python src/train_model.py --config="configs/vit_configs/swin_v2_b.yaml" --pretrained="AF" --iso=$iso;