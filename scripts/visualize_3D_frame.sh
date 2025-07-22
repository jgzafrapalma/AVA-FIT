#!/bin/bash
cd ../

python visualize_3D_frame.py --pred_file '/opt2/data/jzafra/predictions/fit3D_Base_multihmr/s11/diamond_pushup/view_back_left/preds.npz' \
                             --gt_file '/opt2/data/jzafra/tmp/GT_FIT3D_Camera/s11/diamond_pushup/view_back_left/gt.npz' \
                             --video_file '/opt2/data/jzafra/datasets/fit3d/train/s11/videos/50591643/diamond_pushup.mp4' \
                             --cam_params_file '/opt2/data/jzafra/datasets/fit3d/train/s11/camera_parameters/50591643/diamond_pushup.json' \
                             --out_folder '/opt2/data/jzafra/tmp/visualize_3D_Fit3D' \
                             --frame_id 1