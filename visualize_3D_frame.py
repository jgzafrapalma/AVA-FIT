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

def plot_over_image(frame, points_2d=np.array([]), with_ids=True, with_limbs=True, path_to_write=None, fontsize=20):
    num_points = points_2d.shape[0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    if points_2d.shape[0]:
        ax.plot(points_2d[:, 0], points_2d[:, 1], 'x', markeredgewidth=10, color='white')
        if with_ids:
            for i in range(num_points):
                ax.text(points_2d[i, 0], points_2d[i, 1], str(i), color='red', fontsize=fontsize)
        if with_limbs:
            limbs = [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                    [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]]
            for limb in limbs:
                if limb[0] < num_points and limb[1] < num_points:
                    ax.plot([points_2d[limb[0], 0], points_2d[limb[1], 0]], 
                            [points_2d[limb[0], 1], points_2d[limb[1], 1]],
                            linewidth=12.0)
            
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches = 0, bbox_inches='tight')


def project_3d_to_2d(points3d, intrinsics, intrinsics_type):
    if intrinsics_type == 'w_distortion':
        p = intrinsics['p'][:, [1, 0]]
        x = points3d[:, :2] / points3d[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
        tan = np.matmul(x, np.transpose(p))
        xx = x*(tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics['f'] * xx + intrinsics['c']
    elif intrinsics_type == 'wo_distortion':
        xx = points3d[:, :2] / points3d[:, 2:3]
        proj = intrinsics['f'] * xx + intrinsics['c']
    return proj


def get_args():
    parser = argparse.ArgumentParser(description="Visualize 3D meshes from SMPL/SMPLX predictions")
    parser.add_argument("--pred_file", type=str, required=True, help="Predictions file")
    parser.add_argument("--gt_file", type=str, required=False, help="File with the SMPLX/SMPL GT vertices")
    parser.add_argument("--video_file", type=str, required=True, help="Video file to visualize")
    parser.add_argument("--cam_params_file", type=str, required=True, help="Camera parameters file")
    parser.add_argument("--frame_id", type=int, required=True, help="Frame id to visualize")
    parser.add_argument("--error", type=float, required=False, help="Error to visualize")
    parser.add_argument("--out_folder", type=str, required=False, help="Output folder to save the visualization")
    
    return parser.parse_args()


def create_trimesh(vertices, faces, color):
    if len(color) == 3:
        color = color + [255]  # Añadir canal alfa
    vertex_colors = np.tile(np.array(color, dtype=np.uint8), (vertices.shape[0], 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    return mesh

def visualize_meshes(pred_file: str, frame_id: int, out_folder: str, gt_file: str, error: float, frame, cam_params) -> None:

    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    gt = np.load(gt_file)

    v3d = gt['v3d']
    pelvis = gt['transl_pelvis']

    preds = np.load(pred_file)

    v3d_hat = preds['v3d']
    pelvis_hat = preds['transl_pelvis']

    frame_str = f"{frame_id:06d}"
    pred_position = np.where(preds['img_path'] == frame_str)[0]

    v3d = v3d[frame_id]
    pelvis = pelvis[frame_id]

    v3d_hat = v3d_hat[pred_position]
    pelvis_hat = pelvis_hat[pred_position]

    # Center with respect to the pelvis
    v3d_ctx = v3d - pelvis
    v3d_hat_ctx = v3d_hat - pelvis_hat


    if v3d_ctx.shape[0] == 6890:
        neutral_model = SMPL(SMPL_PATH).to(device)

    elif v3d_ctx.shape[0] == 10475:
        neutral_model = SMPLX(SMPLX_PATH).to(device)
    
    mesh_gt = create_trimesh(v3d_ctx, neutral_model.faces, color=[0, 255, 0])
    mesh_pred = create_trimesh(v3d_hat_ctx[0], neutral_model.faces, color=[255, 0, 0])

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

    # Load camera parameters
    cam_params = read_cam_params(args.cam_params_file)
    if cam_params is None:
        print(f"Error reading camera parameters from file: {args.cam_params_file}")
        sys.exit(1)
    

    visualize_meshes(args.pred_file, args.frame_id, args.out_folder, args.gt_file, args.error, frame, cam_params)

if __name__ == "__main__":
    args = get_args()
    main(args)

