#!/bin/bash
# This script is used to get the ground truth data for the AVAFIT dataset.
cd ../

HOME="/opt2/data/jzafra"

python get_GT_AVAFIT.py --gt_path "${HOME}/datasets/avafit/gt" \
        --save_path "${HOME}/gt/avafit"