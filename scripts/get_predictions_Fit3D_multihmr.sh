#!/bin/bash
# This script is used to get the predictions of the multihmr method on the Fit3D dataset.

# Get predictions Fit3D
cd ../

#CPU_CORES "20-39,60-79" - 172.26.1.30
#CPU_CORES "0-11,24-35" - 172.26.1.32

python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
        --output_path "/opt2/data/jzafra/predictions/fit3D_Base_multihmr/" \
        --method "multihmr" --participants "s05" --device 0 --cpu_cores "20-39,60-79" \
        --camera_ids "65906101" --model_name "multiHMR_896_L" \
        --save_mesh --exercises "warmup_14" #--render --extra_views


# python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
#         --output_path "/opt2/data/jzafra/predictions/fit3D_Base_multihmr_time/" \
#         --method "multihmr" --participants "s11" --device 0 --cpu_cores "20-39,60-79" \
#         --camera_ids "50591643" --model_name "multiHMR_896_L" \
#         --save_mesh --exercises "warmup_6" #--render --extra_views