#!/bin/bash
# This script is used to get the ground truth data for the Fit3D dataset.
cd ../

HOME="/opt2/data/jzafra"

CUDA_VISIBLE_DEVICES=0 python get_GT_Fit3D.py --gt_path "${HOME}/datasets/fit3d" \
        --save_path "${HOME}/gt/fit3d"