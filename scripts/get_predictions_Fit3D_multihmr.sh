#!/bin/bash
# This script is used to get the predictions of the multihmr method on the Fit3D dataset.

# Get predictions Fit3D
cd ../
python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
        --output_path "/opt2/data/jzafra/predictions/fit3D_Base_multihmr/" \
        --method "multihmr" --participants "s11" --device 1 --cpu_cores "20-39,60-79" \
        --exercises "diamond_pushup" "man_maker" "warmup_16" "warmup_1" "pushup" --camera_ids "50591643" "60457274" "65906101" "58860488" --model_name "multiHMR_896_L" \
        --save_mesh #--render --extra_views