"""
    This script precalculates the ground truth for the AVAFIT dataset.
    It loads the SMPLX parameters and gets the 3D vertices and pelvis position
    for each frame.
"""
import os
import copy
import torch
import pathlib
import argparse
import numpy as np
import smplx

from utils import MODEL_FOLDER

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Get Ground Truth data for evaluation on AVAFIT dataset")
    parser.add_argument("--gt_path", type=str, required=False, help="Path to the AVAFIT ground truth files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    return parser.parse_args()

smplx_model_male = smplx.create(MODEL_FOLDER, model_type='smplx',
                                gender='male',
                                ext='npz',
                                num_betas=10,
                                flat_hand_mean=True,
                                use_pca=False)

smplx_model_female = smplx.create(MODEL_FOLDER, model_type='smplx',
                                  gender='female',
                                  ext='npz',
                                  num_betas=10,
                                  flat_hand_mean=True,
                                  use_pca=False)  

smplx_model_neutral = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='neutral',
                                   ext='npz',
                                   flat_hand_mean=True,
                                   num_betas=10,
                                   use_pca=False) 

def get_smplx_vertices(poses, betas, trans, gender):
    """
    Get SMPLX vertices for a single frame
    Args:
        poses: (165,) array for single frame
        betas: (10,) array for single frame
        trans: (3,) array for single frame
        gender: string
    """
    if gender == 'male':
        model_out = smplx_model_male(betas=torch.tensor(betas).unsqueeze(0).float(),
                                     global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                                     body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                                     left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                                     right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                                     jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                                     leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                                     reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                                     transl=torch.tensor(trans).unsqueeze(0))
    elif gender == 'female':
        model_out = smplx_model_female(betas=torch.tensor(betas).unsqueeze(0).float(),
                                       global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                                       body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                                       left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                                       right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                                       jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                                       leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                                       reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                                       transl=torch.tensor(trans).unsqueeze(0))
    elif gender == 'neutral':
        model_out = smplx_model_neutral(betas=torch.tensor(betas).unsqueeze(0).float(),
                                        global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                                        body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                                        left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                                        right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                                        jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                                        leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                                        reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                                        transl=torch.tensor(trans).unsqueeze(0))
    else:
        print('Please provide gender as male, female, or neutral')
        return None, None
    
    return model_out.vertices, model_out.joints

def prepate_gt(gt_path, save_path):
    for gt_file in sorted(os.listdir(gt_path)):
        gt = np.load(os.path.join(gt_path, gt_file))
        
        gt_name_split = gt_file.split('_')
        participant = gt_name_split[1] + '_' + gt_name_split[2]
        exercise = gt_name_split[3]
        repetition = gt_name_split[4]
        viewpoint = gt_name_split[5]
        
        pose_cam = gt['pose_cam']  # [B, 165]
        betas = gt['shape']  # [B, 10]
        trans_cam = gt['trans_cam']  # [B, 3]
        cam_ext = gt['cam_ext']  # [B, 4, 4]
        
        print(f'Processing {participant} - {exercise} - {viewpoint} - {repetition}')
        print(f'pose_cam shape: {pose_cam.shape}')
        print(f'betas shape: {betas.shape}')
        print(f'trans_cam shape: {trans_cam.shape}')
        
        # Process each frame individually
        num_frames = pose_cam.shape[0]
        vertices_list = []
        transl_pelvis_list = []
        
        for frame_idx in range(num_frames):
            # Get data for this specific frame
            pose_frame = pose_cam[frame_idx]  # (165,)
            beta_frame = betas[frame_idx]  # (10,)
            trans_frame = trans_cam[frame_idx]  # (3,)
            
            # Get vertices and joints for this frame
            vertices, joints = get_smplx_vertices(pose_frame, beta_frame, trans_frame, gender=gt['gender'][0])
            cam_trans_frame = cam_ext[frame_idx, :3, 3]
            
            if vertices is None or joints is None:
                print(f'Error processing frame {frame_idx}')
                continue

            vertices_np = vertices.detach().cpu().numpy()  # (1, 10475, 3)
            vertices_cam_absolute = vertices_np + cam_trans_frame
            
            vertices_list.append(vertices_cam_absolute)  # (1, 10475, 3)
            transl_pelvis_list.append(joints[:, 0, :].detach().cpu().numpy().reshape(-1, 1, 3) + cam_trans_frame)  # (1, 1, 3)
        
        # Stack all frames
        vertices = np.concatenate(vertices_list, axis=0)  # (B, 10475, 3)
        transl_pelvis = np.concatenate(transl_pelvis_list, axis=0)  # (B, 1, 3)

        vertices = vertices.astype(np.float32)
        transl_pelvis = transl_pelvis.astype(np.float32)
        
        print(f'Final vertices shape: {vertices.shape}')
        print(f'Final transl_pelvis shape: {transl_pelvis.shape}')
        
        # Save the results
        save_dir = os.path.join(save_path, participant, exercise, viewpoint, repetition)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(save_dir, 'gt.npz'), v3d=vertices, transl_pelvis=transl_pelvis, gender=gt['gender'][0])
        
        print(f'Saved results to {save_dir}')

def main(args):
    SAVE_PATH = args.save_path
    prepate_gt(args.gt_path, SAVE_PATH)

if __name__ == "__main__":
    args = get_args()
    main(args)