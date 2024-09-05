#!/bin/bash

python src/loco_cv.py --config="configs/cnn_configs/convnext_small.yaml"; 
python src/loco_cv.py --config="configs/cnn_configs/convnext_base.yaml";
python src/loco_cv.py --config="configs/cnn_configs/convnext_large.yaml";
python src/loco_cv.py --config="configs/vit_configs/vit_b_16.yaml"; 
python src/loco_cv.py --config="configs/vit_configs/vit_l_16.yaml"; 
python src/loco_cv.py --config="configs/vit_configs/vit_h_14.yaml"; 
python src/loco_cv.py --config="configs/vit_configs/swin_v2_t.yaml";
python src/loco_cv.py --config="configs/vit_configs/swin_v2_s.yaml";
python src/loco_cv.py --config="configs/vit_configs/swin_v2_b.yaml";