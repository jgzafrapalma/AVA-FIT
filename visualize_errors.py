import os
import cv2
import sys
import json
import torch
import pathlib
import trimesh
import argparse
import numpy as np
import glob

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


def visualize_errors(errors_path: str, pred_path: str, gt_path: str, out_folder: str, dataset_path: str) -> None:
    dataset_name = pathlib.Path(dataset_path).name
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    with open(errors_path, 'r') as f:
        errors = json.load(f)

    for participant, exercises in errors.items():
        for exercise, viewpoints in exercises.items():
            for viewpoint, data in viewpoints.items():
                _process_viewpoint(
                    data, dataset_name, dataset_path,
                    participant, exercise, viewpoint,
                    pred_path, gt_path, out_folder
                )


def _get_video_file(dataset_name: str, dataset_path: str, participant: str, exercise: str, viewpoint: str, repetition: str = None) -> str:
    if dataset_name == 'avafit':
        video_dir = os.path.join(dataset_path, 'videos')
        matches = glob.glob(os.path.join(video_dir, f"seq_{participant}_{exercise}_{repetition}_{viewpoint}_*.mp4"))
        
        if not matches:
            raise FileNotFoundError(f"No video found for pattern 'seq_{participant}_{exercise}_{repetition}_{viewpoint}_*.mp4' in {video_dir}")
        if len(matches) > 1:
            raise ValueError(f"Multiple videos found for pattern 'seq_{participant}_{exercise}_{repetition}_{viewpoint}_*.mp4' in {video_dir}: {matches}")
        
        return matches[0]
    
    return os.path.join(dataset_path, 'train', participant, 'videos', viewpoint_to_camera_id[viewpoint], f"{exercise}.mp4")


def _run_visualization(pred_path: str, gt_path: str, out_folder: str, participant: str, exercise: str, viewpoint: str, metric: str, frame_path: str, error: float, video_file: str, dataset_name: str, repetition: str = None) -> None:

    if dataset_name == 'avafit' and repetition is not None:
        pred_file = os.path.join(pred_path, participant, exercise, viewpoint, repetition, 'preds.npz')
        gt_file   = os.path.join(gt_path,   participant, exercise, viewpoint, repetition, 'gt.npz')
        out_dir   = os.path.join(out_folder, participant, exercise, viewpoint, repetition, metric)
    else:
        pred_file = os.path.join(pred_path, participant, exercise, viewpoint, 'preds.npz')
        gt_file   = os.path.join(gt_path,   participant, exercise, viewpoint, 'gt.npz')
        out_dir   = os.path.join(out_folder, participant, exercise, viewpoint, metric)

    cmd = (
        f"python visualize_3D_frame.py"
        f" --pred_file {pred_file}"
        f" --gt_file {gt_file}"
        f" --frame_id {int(frame_path)}"
        f" --out_folder {out_dir}"
        f" --error {error}"
        f" --video_file {video_file}"
        f" --dataset {dataset_name}"
    )
    os.system(cmd)


def _process_viewpoint(data: dict, dataset_name: str, dataset_path: str, participant: str, exercise: str, viewpoint: str, pred_path: str, gt_path: str, out_folder: str) -> None:
    is_avafit = dataset_name == 'avafit'
    items = (
        ((rep, metric, frame, error) for rep, metrics in data.items() for metric, frames in metrics.items() for frame, error in frames)
        if is_avafit else
        ((None, metric, frame, error) for metric, frames in data.items() for frame, error in frames)
    )

    for repetition, metric, frame_path, error in items:
        print(f"Processing {participant} - {exercise} - {viewpoint}" + (f" - {repetition}" if repetition else "") + f" - {metric} - {frame_path} - {error}")
        video_file = _get_video_file(dataset_name, dataset_path, participant, exercise, viewpoint, repetition)
        _run_visualization(
            pred_path, gt_path, out_folder,
            participant, exercise, viewpoint,
            metric, frame_path, error,
            video_file, dataset_name,
            repetition=repetition
        )


def main(args):
    visualize_errors(args.errors_path, args.pred_path, args.gt_path, args.out_folder, args.dataset_path)


if __name__ == "__main__":
    args = get_args()
    main(args)