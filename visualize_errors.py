import os
import cv2
import sys
import torch
import joblib
import pathlib
import trimesh
import argparse
import numpy as np

camera_id_to_viewpoint = {
    '60457274': 'view_front_left',
    '65906101': 'view_front_right',
    '50591643': 'view_back_left',
    '58860488': 'view_back_right',
}

viewpoint_to_camera_id = {
    'view_front_left': '60457274',
    'view_front_right': '65906101',
    'view_back_left': '50591643',
    'view_back_right': '58860488',
}


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 3D meshes from top errors frames")
    parser.add_argument("--errors_path", type=str, required=True, help="Path to the errors file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the predictions file")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--out_folder", type=str, required=False, help="Output folder to save the visualization")
    
    return parser.parse_args()


def visualize_errors(errors_path: str, pred_path: str, gt_path: str, out_folder: str, dataset_path:str) -> None:

    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    errors = joblib.load(errors_path)

    for participant in errors.keys():
        for exercise in errors[participant].keys():
            for viewpoint in errors[participant][exercise].keys():
                for metric in errors[participant][exercise][viewpoint].keys():
                    print(f"Processing {participant} - {exercise} - {viewpoint} - {metric}")
                    for frame_path, error in errors[participant][exercise][viewpoint][metric]:
                        print(f"Processing {frame_path} - {error}")

                        video_file = os.path.join(dataset_path, 'train', participant, 'videos', viewpoint_to_camera_id[viewpoint], f"{exercise}.mp4")
                        cam_params_file = os.path.join(dataset_path, 'train', participant, 'camera_parameters', viewpoint_to_camera_id[viewpoint], f"{exercise}.json")
                        cmd = f"python visualize_3D_frame.py --pred_file {os.path.join(pred_path, participant, exercise, viewpoint, 'preds.npz')} --gt_file {os.path.join(gt_path, participant, exercise, viewpoint, 'gt.npz')} --frame_id {int(frame_path)} --out_folder {os.path.join(out_folder, participant, exercise, viewpoint, metric)} --error {error} --video_file {video_file} --cam_params_file {cam_params_file}"
                        os.system(cmd)

def main(args):

    visualize_errors(args.errors_path, args.pred_path, args.gt_path, args.out_folder, args.dataset_path)

if __name__ == "__main__":
    args = get_args()
    main(args)

