#!/bin/bash
# This script is used to get the predictions of the PromptHMR method on the Fit3D dataset.
# Get predictions Fit3D
cd ../

#CPU_CORES "20-39,60-79" - 172.26.1.30
#CPU_CORES "0-11,24-35" - 172.26.1.32

python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
        --output_path "/opt2/data/jzafra/predictions/fit3D_Base_PromptHMR_time/" \
        --method "PromptHMR" --participants "s03" "s04" "s05" "s07" "s08" "s09" "s10" "s11" --device 1 --cpu_cores "0-11,24-35" \
        --camera_ids "50591643" "60457274" "65906101" "58860488"

