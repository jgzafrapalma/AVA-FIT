#!/bin/bash
# This script is used to get the predictions of the 4D-humans method on the AVAFIT dataset.
# Get predictions AVAFIT
cd ../

#CPU_CORES "20-39,60-79" - 172.26.1.30
#CPU_CORES "0-11,24-35" - 172.26.1.32
#"50591643" "60457274" "65906101" "58860488"

python predictions_AVAFIT.py --dataset_path "/opt2/data/jzafra/datasets/avafit/" \
        --output_path "/opt4/data/jzafra/predictions/avafit_Base_multihmr/" \
        --method "multihmr" --device 0 --cpu_cores "0-11,24-35" \
        --model_name "multiHMR_896_L" --save_mesh --participants "001_herong" --exercises "A012" #--render #--extra_views
