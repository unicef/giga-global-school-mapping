#!/bin/bash

echo -n "Input ISO: "
read iso

python src/sat_batch_download.py --data_config="configs/data_configs/data_config_ISO_AS.yaml" --sat_config="configs/sat_configs/sat_config_500x500_60cm.yaml" --sat_creds="configs/sat_configs/issa_sat_creds.yaml" --iso_code=$iso --adm_level="ADM2" --mode="sat";
