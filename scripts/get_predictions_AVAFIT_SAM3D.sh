#!/bin/bash
# This script is used to get the predictions of the 4D-humans method on the AVAFIT dataset.
# Get predictions AVAFIT
cd ../

#CPU_CORES "20-39,60-79" - 172.26.1.30
#CPU_CORES "0-11,24-35" - 172.26.1.32

python predictions_AVAFIT.py --dataset_path "/opt2/data/jzafra/datasets/avafit/" \
        --output_path "/opt4/data/jzafra/predictions/avafit_Base_SAM3DBODY/" \
        --method "SAM3D_BODY" --device 1 --cpu_cores "20-39,60-79"
