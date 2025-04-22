#!/bin/bash
# This script is used to get the predictions of the multihmr method on the Fit3D dataset.

# Get predictions Fit3D
python predictions_Fit3D.py --dataset_path "/opt2/data/jzafra/datasets/fit3d/" \
        --output_path "/opt2/data/jzafra/predictions/fit3D_Base_multihmr/" \
        --method "multihmr" --participants "s11" --device 0 --cpu_cores "20-39,60-79"