#!/bin/bash
# This script is used to get the ground truth data for the Fit3D dataset.
cd ../


python get_GT_Fit3D.py --smplx_path "/opt2/data/jzafra/data/smplx" \
        --gt_path "/opt2/data/jzafra/datasets/fit3d" \
        --save_path "/opt2/data/jzafra/gt/fit3d" \
        --participants "s03" "s04" "s05" "s07" "s08" "s09" "s10"