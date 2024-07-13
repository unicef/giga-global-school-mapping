#!/bin/bash

echo -n "Input ISO: "
read iso

if [ "$iso" == "SEN" ]
then
python src/sat_predict.py --data_config="configs/data_configs/data_config_ISO_AF.yaml" --model_config="configs/best_models.yaml" --sat_config="configs/sat_configs/sat_config_500x500_60cm.yaml" --sat_creds="configs/sat_configs/issa_sat_creds.yaml" --cam_method="gradcam" --threshold=0.355 --iso_code=$iso;
fi