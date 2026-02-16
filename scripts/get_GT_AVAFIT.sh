#!/bin/bash
# This script is used to get the ground truth data for the Fit3D dataset.
cd ../


python get_GT_AVAFIT.py --gt_path "/opt2/data/jzafra/datasets/avafit/gt" \
        --save_path "/opt4/data/jzafra/gt/avafit"