#!/bin/bash

echo -n "Input ISO: "
read iso

python train_model.py --config="configs/cnn_configs/convnext_small.yaml" --iso=$iso; 
python train_model.py --config="configs/cnn_configs/convnext_base.yaml" --iso=$iso;
python train_model.py --config="configs/cnn_configs/convnext_large.yaml" --iso=$iso;
python train_model.py --config="configs/cnn_configs/vgg16.yaml" --iso=$iso; 
python train_model.py --config="configs/cnn_configs/resnet50.yaml" --iso=$iso; 
python train_model.py --config="configs/cnn_configs/resnet50_fmow_rgb_gassl.yaml" --iso=$iso; 
python train_model.py --config="configs/cnn_configs/xception.yaml" --iso=$iso; 
python train_model.py --config="configs/cnn_configs/inceptionv3.yaml" --iso=$iso;