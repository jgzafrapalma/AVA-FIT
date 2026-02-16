import os
import cv2
import sys
import torch
import joblib
import pathlib
import trimesh
import argparse
import numpy as np

from smplx import SMPL, SMPLX
from utils.constants import SMPL_PATH, SMPLX_PATH
#from utils.smplx_utils import render
from utils.fit3d_utils import read_cam_params
import matplotlib.pyplot as plt


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 3D meshes from SMPL/SMPLX predictions")
    parser.add_argument("--pred_file", type=str, required=True, help="Predictions file")
    parser.add_argument("--gt_file", type=str, required=False, help="File with the SMPLX/SMPL GT vertices")
    parser.add_argument("--video_file", type=str, required=True, help="Video file to visualize")
    parser.add_argument("--frame_id", type=int, required=True, help="Frame id to visualize")
    parser.add_argument("--error", type=float, required=False, default=0.0, help="Error to visualize")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the visualization")
    
    return parser.parse_args()


def create_trimesh(vertices, faces, color):
    if len(color) == 3:
        color = color + [255]  # Añadir canal alfa
    vertex_colors = np.tile(np.array(color, dtype=np.uint8), (vertices.shape[0], 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    return mesh

def visualize_meshes(pred_file: str, frame_id: int, out_folder: str, gt_file: str, error: float, frame) -> None:

    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    gt = np.load(gt_file)

    v3d = gt['v3d']
    pelvis = gt['transl_pelvis']

    preds = np.load(pred_file)

    v3d_hat = preds['v3d']
    pelvis_hat = preds['transl_pelvis']

    img_ids = np.array([int(p) for p in preds['img_path']]) - 1
    pred_position = np.where(img_ids == frame_id)[0][0]

    v3d = v3d[frame_id]
    pelvis = pelvis[frame_id]

    v3d_hat = v3d_hat[pred_position]
    pelvis_hat = pelvis_hat[pred_position]

    # Center with respect to the pelvis
    v3d_ctx = v3d - pelvis
    v3d_hat_ctx = v3d_hat - pelvis_hat

    smplx_model = SMPLX(SMPLX_PATH, create_transl=True, use_pca=False, num_betas=10, ext='pkl', gender='male').to(device)
    
    if v3d_hat_ctx.shape[0] == 6890:
        pred_neutral_model = SMPL(SMPL_PATH).to(device)

    elif v3d_hat_ctx.shape[0] == 10475:
        pred_neutral_model = SMPLX(SMPLX_PATH).to(device)

    mesh_gt = create_trimesh(v3d_ctx, smplx_model.faces, color=[0, 255, 0]) # green
    mesh_pred = create_trimesh(v3d_hat_ctx, pred_neutral_model.faces, color=[255, 0, 0]) # red

    # Faltaría renderizar los meshes y superponerlos en el frame

    cv2.imwrite(os.path.join(out_folder, f"{error:.2f}_{frame_id:06d}.png"), frame)

    # Save meshes to .obj files
    scene = trimesh.Scene()
    scene.add_geometry(mesh_gt, node_name = "ground_truth")
    scene.add_geometry(mesh_pred, node_name = "prediction")

    scene.export(os.path.join(out_folder, f"{error:.2f}_{frame_id:06d}.glb"), file_type=".glb")


def main(args):

    # Load the video
    video = cv2.VideoCapture(args.video_file)
    if not video.isOpened():
        print(f"Error opening video file: {args.video_file}")
        sys.exit(1)
    # Get a specific frame
    video.set(cv2.CAP_PROP_POS_FRAMES, args.frame_id)
    ret, frame = video.read()
    if not ret:
        print(f"Error reading frame {args.frame_id} from video file: {args.video_file}")
        sys.exit(1)
    

    visualize_meshes(args.pred_file, args.frame_id, args.out_folder, args.gt_file, args.error, frame)

if __name__ == "__main__":
    args = get_args()
    main(args)

