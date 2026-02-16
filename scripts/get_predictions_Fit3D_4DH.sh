#!/bin/bash
# This script is used to get the predictions of the 4D-humans method on the Fit3D dataset.
# Get predictions Fit3D
cd ../

#CPU_CORES "20-39,60-79" - 172.26.1.30
#CPU_CORES "0-11,24-35" - 172.26.1.32
#"50591643" "60457274" "65906101" "58860488"

python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
        --output_path "/opt2/data/jzafra/predictions/fit3D_Base_4DH/" \
        --method "4D-humans" --participants "s08" --device 0 --cpu_cores "20-39,60-79" \
        --camera_ids "58860488" \
        --save_mesh --batch_size 8 --exercises "warmup_1" #--render --extra_views 
