"""
    This script precalculates the ground truth for the Fit3D dataset.
    It loads the SMPLX parameters and gets the 3D vertices and pelvis position
    for each frame.
"""

import os
import copy
import json
import torch
import pathlib
import argparse
import numpy as np
from smplx import SMPLX
from utils.fit3d_utils import read_cam_params
from pytorch3d.transforms import matrix_to_axis_angle

from smplx import build_layer

smplx_cfg = {'ext': 'npz',
             'extra_joint_path': '',
             'folder': 'transfer_data/body_models',
             'gender': 'neutral',
             'joint_regressor_path': '',
             'model_type': 'smplx',
             'num_expression_coeffs': 10,
             'smplx': {'betas': {'create': True, 'num': 10, 'requires_grad': True},
                       'body_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'expression': {'create': True, 'num': 10, 'requires_grad': True},
                       'global_rot': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'jaw_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'left_hand_pose': {'create': True,
                                          'pca': {'flat_hand_mean': False, 'num_comps': 12},
                                          'requires_grad': True,
                                          'type': 'aa'},
                       'leye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'reye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'right_hand_pose': {'create': True,
                                           'pca': {'flat_hand_mean': False,
                                                   'num_comps': 12},
                                           'requires_grad': True,
                                           'type': 'aa'},
                       'translation': {'create': True, 'requires_grad': True}},
             'use_compressed': False,
             'use_face_contour': True}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

camera_id_to_viewpoint = {
    '60457274': 'view_front_left',
    '65906101': 'view_front_right',
    '50591643': 'view_back_left',
    '58860488': 'view_back_right',
}


smplx_model = build_layer('/opt2/data/jzafra/data', **smplx_cfg)

def get_args():
    parser = argparse.ArgumentParser(description="Get Ground Truth data for evaluation on Fit3D dataset")
    parser.add_argument("--smplx_path", type=str, required=True, help="Path to the SMPLX model")
    parser.add_argument("--gt_path", type=str, required=False, help="Path to the keypoints ground truth (only for Fit3D)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    
    return parser.parse_args()


def get_camera_smplx_params(smplx_params, cam_params):
    pelvis = smplx_model(betas=torch.from_numpy(np.array(smplx_params['betas']).astype(np.float32))).joints[:, 0, :].numpy()
    camera_smplx_params = copy.deepcopy(smplx_params)
    camera_smplx_params['global_orient'] = np.matmul(np.array(smplx_params['global_orient']).transpose(0, 1, 3, 2), np.transpose(cam_params['extrinsics']['R'])).transpose(0, 1, 3, 2)
    camera_smplx_params['transl'] = np.matmul(smplx_params['transl'] + pelvis - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R'])) - pelvis
    camera_smplx_params = {key: torch.from_numpy(np.array(camera_smplx_params[key]).astype(np.float32)).to(device) for key in camera_smplx_params}

    return camera_smplx_params

def rotation_camera_view(global_orient, cam_params):
    """
    Rotates the global orientation of the SMPLX model to the camera view.
    Args:
        global_orient (torch.Tensor): Global orientation of the SMPLX model.
        cam_params (dict): Camera parameters including rotation matrix.
    Returns:
        np.ndarray: Rotated global orientation.
    """

    return np.matmul(np.array(global_orient).transpose(0, 1, 3, 2), np.transpose(cam_params['extrinsics']['R'])).transpose(0, 1, 3, 2) 

def rotation_matrix_to_axis_angle(tensor_4d: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of rotation matrices [B, J, 3, 3] to axis-angle [B, J, 3].

    Args:
        tensor_4d (torch.Tensor): Input tensor of shape [B, J, 3, 3].

    Returns:
        torch.Tensor: Output tensor of shape [B, J, 3] (axis-angle).
    """
    B, J = tensor_4d.shape[:2]
    tensor_flat = tensor_4d.view(-1, 3, 3)         # [B*J, 3, 3]
    axis_angle_flat = matrix_to_axis_angle(tensor_flat)  # [B*J, 3]
    axis_angle = axis_angle_flat.view(B, J, 3)     # [B, J, 3]

    return axis_angle

def prepate_gt(gt_path, smplx_neutral, save_path, participants=None):

    if participants is None:
        participants = os.listdir(os.path.join(gt_path, 'train'))

    for participant in participants:

        for exercise in os.listdir(os.path.join(gt_path, 'train', participant, 'smplx')):

            with open(os.path.join(gt_path, 'train', participant, 'smplx', exercise)) as f:
                smplx_params = json.load(f)

            for camera_id in os.listdir(os.path.join(gt_path, 'train', participant, 'camera_parameters')):

                viewpoint = camera_id_to_viewpoint[camera_id]

                print(f'Processing {participant} - {pathlib.Path(exercise).stem} - {camera_id} - {viewpoint}')

                cam_params = read_cam_params(os.path.join(gt_path, 'train', participant, 'camera_parameters', camera_id, exercise))

                camera_smplx_params = get_camera_smplx_params(smplx_params, cam_params)

                # betas = torch.tensor(smplx_params['betas'], dtype=torch.float32).to(device) # [B, 10]
                # global_orient = torch.tensor(rotation_camera_view(smplx_params['global_orient'], cam_params), dtype=torch.float32).to(device) # [B, 1, 3, 3]
                # body_pose = torch.tensor(smplx_params['body_pose'], dtype=torch.float32).to(device) # [B, 21, 3, 3]
                # left_hand_pose = torch.tensor(smplx_params['left_hand_pose'], dtype=torch.float32).to(device) # [B, 15, 3, 3]
                # right_hand_pose = torch.tensor(smplx_params['right_hand_pose'], dtype=torch.float32).to(device) # [B, 15, 3, 3]
                # jaw_pose = torch.tensor(smplx_params['jaw_pose'], dtype=torch.float32).to(device) # [B, 1, 3, 3]
                # leye_pose = torch.tensor(smplx_params['leye_pose'], dtype=torch.float32).to(device) # [B, 1, 3, 3]
                # reye_pose = torch.tensor(smplx_params['reye_pose'], dtype=torch.float32).to(device) # [B, 1, 3, 3]
                # expression = torch.tensor(smplx_params['expression'], dtype=torch.float32).to(device) # [B, 10]
                # #transl = torch.tensor(smplx_params['transl'], dtype=torch.float32).to(device) # [B, 3]


                # global_orient = rotation_matrix_to_axis_angle(global_orient) # [B, 1, 3]
                # body_pose = rotation_matrix_to_axis_angle(body_pose) # [B, 21, 3]
                # jaw_pose = rotation_matrix_to_axis_angle(jaw_pose) # [B, 1, 3]
                # left_hand_pose = rotation_matrix_to_axis_angle(left_hand_pose) # [B, 15, 3]
                # right_hand_pose = rotation_matrix_to_axis_angle(right_hand_pose) # [B, 15, 3]
                # leye_pose = rotation_matrix_to_axis_angle(leye_pose) # [B, 1, 3]
                # reye_pose = rotation_matrix_to_axis_angle(reye_pose) # [B, 1, 3]

                # smplx_output = smplx_neutral(betas=betas,
                #                             body_pose=body_pose,
                #                             global_orient=global_orient,
                #                             jaw_pose=jaw_pose,
                #                             leye_pose=leye_pose,
                #                             reye_pose=reye_pose,
                #                             left_hand_pose=left_hand_pose,
                #                             right_hand_pose=right_hand_pose,
                #                             expression=expression,
                #                             pose2rot=True)


                global_orient = rotation_matrix_to_axis_angle(camera_smplx_params['global_orient']) # [B, 1, 3]
                body_pose = rotation_matrix_to_axis_angle(camera_smplx_params['body_pose']) # [B, 21, 3]
                jaw_pose = rotation_matrix_to_axis_angle(camera_smplx_params['jaw_pose']) # [B, 1, 3]
                left_hand_pose = rotation_matrix_to_axis_angle(camera_smplx_params['left_hand_pose']) # [B, 15, 3]
                right_hand_pose = rotation_matrix_to_axis_angle(camera_smplx_params['right_hand_pose']) # [B, 15, 3]
                leye_pose = rotation_matrix_to_axis_angle(camera_smplx_params['leye_pose']) # [B, 1, 3]
                reye_pose = rotation_matrix_to_axis_angle(camera_smplx_params['reye_pose']) # [B, 1, 3]

                smplx_output = smplx_neutral(betas=camera_smplx_params['betas'],
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            jaw_pose=jaw_pose,
                                            leye_pose=leye_pose,
                                            reye_pose=reye_pose,
                                            left_hand_pose=left_hand_pose,
                                            right_hand_pose=right_hand_pose,
                                            expression=camera_smplx_params['expression'],
                                            transl=camera_smplx_params['transl'],
                                            pose2rot=True)


                vertices = smplx_output.vertices.detach().cpu().numpy() # [B, 10475, 3]
                transl_pelvis = smplx_output.joints[:, 0, :].detach().cpu().numpy().reshape(-1, 1, 3) # [B, 1, 3]
                
                #Save the results
                save_dir = os.path.join(save_path, participant, pathlib.Path(exercise).stem, viewpoint)

                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

                np.savez(os.path.join(save_dir, 'gt.npz'), v3d = vertices, transl_pelvis = transl_pelvis)

def main(args):

    SMPLX_PATH = args.smplx_path
    SAVE_PATH = args.save_path

    smplx_neutral = SMPLX(SMPLX_PATH, create_transl=True, use_pca=False).to(device)

    prepate_gt(args.gt_path, smplx_neutral, SAVE_PATH, args.participants)


if __name__ == "__main__":
    args = get_args()
    main(args)