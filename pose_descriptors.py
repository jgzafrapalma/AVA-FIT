# A partir del ground truth, se obtendrán cada x fotogramas los descriptores de pose.
# Estos descriptores se almacenarán para posteriormente realizar la visualización utilizando
# técnicas de reducción de dimensionalidad como PCA o t-SNE.

import os
import sys
import argparse
import numpy as np

from scipy.spatial.distance import pdist

from utils import MODEL_FOLDER, SMPLX_PATH

def get_args():
    parser = argparse.ArgumentParser(description="Get pose descriptors from 3D human mesh estimation models")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--interval", type=int, required=False, default=10, help="Interval between frames")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    return parser.parse_args()


def get_descriptor(gt_path, dataset_path, output_path, interval, verbose):
    
    # Load SMPLX file
    smplx_model = os.path.join(SMPLX_PATH, 'SMPLX_NEUTRAL.npz')
    smplx_regressor_neutral = np.load(smplx_model)['J_regressor']

    for dataset in sorted(os.listdir(gt_path)):

        for participant in sorted(os.listdir(os.path.join(gt_path, dataset))):

            for exercise in sorted(os.listdir(os.path.join(gt_path, dataset, participant))):

                # Use only one viewpoint the pose is the same in all viewpoints
                viewpoints = os.listdir(os.path.join(gt_path, dataset, participant, exercise))
                viewpoints = [viewpoints[0]]

                for viewpoint in viewpoints:

                    if dataset == 'avafit':
                        repetitions = os.listdir(os.path.join(gt_path, dataset, participant, exercise, viewpoint))
                    else:
                        repetitions = [None]

                    for repetition in repetitions:
                        if repetition is not None:
                            gt_file = os.path.join(gt_path, dataset, participant, exercise, viewpoint, repetition, 'gt.npz')
                            video_file = os.path.join(dataset_path, dataset, 'videos', f"seq_{participant}_{exercise}_{repetition}_{viewpoint}.mp4")
                        else:
                            gt_file = os.path.join(gt_path, dataset, participant, exercise, viewpoint, 'gt.npz')
                            video_file = f"seq_{participant}_{exercise}_{viewpoint}.mp4"

                        gt_data = np.load(gt_file)
                        
                        if 'gender' in gt_data:
                            smplx_file = os.path.join(SMPLX_PATH, f'SMPLX_{str(gt_data["gender"]).upper()}.npz')
                            smplx_regressor = np.load(smplx_file)['J_regressor']
                        else:
                            smplx_regressor = smplx_regressor_neutral

                        # Get Joints from gt_data
                        v3d = gt_data['v3d']
                        transl_pelvis = gt_data['transl_pelvis']

                        # Center the mesh
                        v3d = v3d - transl_pelvis

                        # Get joints
                        J = smplx_regressor @ v3d

                        # Get the interval frames joints
                        J_interval = J[::interval]

                        steps = np.arange(len(J))
                        steps_interval = steps[::interval]

                        # Calculate descriptors
                        # Seleccionamos los primeros 22 joints que corresponden al cuerpo principal en SMPL-X
                        body_joints_seq = J_interval[:, :22, :]

                        descriptors = []
                        for frame_joints in body_joints_seq:
                            # Distancias entre todos los pares de los 22 joints (231 características por fotograma)
                            dist_vec = pdist(frame_joints, metric='euclidean')
                            descriptors.append(dist_vec)

                        descriptors = np.array(descriptors)

                        if verbose:
                            print(f'Participant: {participant}, Exercise: {exercise}, Viewpoint: {viewpoint}, Repetition: {repetition}, Descriptors: {descriptors.shape}')

                        # Save like npz
                        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)
                        if repetition is not None:
                            np.savez(os.path.join(output_path, dataset, f'{participant}_{exercise}_{viewpoint}_{repetition}_descriptors.npz'), descriptors=descriptors, indexes=steps_interval)
                        else:
                            np.savez(os.path.join(output_path, dataset, f'{participant}_{exercise}_{viewpoint}_descriptors.npz'), descriptors=descriptors, indexes=steps_interval)
                            
def main(args):

    GT_PATH = args.gt_path
    DATASET_PATH = args.dataset_path
    OUTPUT_PATH = args.output_path
    INTERVAL = args.interval
    VERBOSE = args.verbose

    get_descriptor(GT_PATH, DATASET_PATH, OUTPUT_PATH, INTERVAL, VERBOSE)



if __name__ == "__main__":
    args = get_args()
    main(args)
    
