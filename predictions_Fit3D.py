import os
import sys
import pathlib
import argparse
import subprocess
from utils import HOME, CONDA_INIT, METHODS_DIR

new_home = os.environ.copy()

new_home["HOME"] = HOME

def get_args():
    parser = argparse.ArgumentParser(description="Get predictions from 3D reconstruction models for the fit3D dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--method", type=str, required=True, help="Available models: [4D-Humans]")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    parser.add_argument("--camera_ids", nargs='+', required=False, help="Viewpoints to process")
    parser.add_argument("--exercises", nargs='+', required=False, help="Exercises to process")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to checkpoint")
    parser.add_argument("--device", type=int, required=False, default=0, help='device to use')
    parser.add_argument("--render", dest='render', action='store_true', default=False, help='If set, save rendered results')
    parser.add_argument("--cpu_cores", type=str, default="20-39,60-79", help="CPU cores to use with taskset")
    parser.add_argument("--batch_size", type=int, required=False, help="path to output directory")
    parser.add_argument("--save_mesh", dest='save_mesh', action='store_true', default=False, help='save .obj file with the meshes')

    return parser.parse_args()


def run_method_video(method, video, target_dir, device, render, cpu_cores, save_mesh, batch_size = None, checkpoint = None) -> None:

    print(f"Method {method}")

    method_dir = pathlib.Path(os.path.join(METHODS_DIR, method))

    #> /dev/null 2>&1

    base_command = (
        f"source {CONDA_INIT}; "
        f"cd {method_dir}; "
        f"conda activate {method}; "
        f"python pose_estimation.py --video_path {video} --output_dir {target_dir} --device {device} --cpu_cores {cpu_cores}"
    )

    if batch_size:
        base_command += f" --batch_size {batch_size}"
    if checkpoint:
        base_command += f" --checkpoint {checkpoint}"
    if render:
        base_command += f" --render"
    if save_mesh:
        base_command += f" --save_mesh"

    command = f'/bin/bash -i -c "{base_command}"'

    subprocess.run(command, env=new_home, shell=True)

def get_fit3d_predictions(dataset_path: str, output: str, participants: list = None, camera_ids: list = None, exercises: list = None) -> None:

    if not participants:
        participants = sorted(os.listdir(os.path.join(dataset_path, 'train')))
    else:
        participants = sorted(participants)

    for participant in participants:

        if not camera_ids:
            camera_ids = sorted(os.listdir(os.path.join(dataset_path, 'train', participant, 'videos')))
        else:
            camera_ids = sorted(camera_ids)

        for camera_id in camera_ids:
            
            if not exercises:
                exercises = sorted(os.listdir(os.path.join(dataset_path, 'train', participant, 'videos', camera_id)))
                exercises = [pathlib.Path(exercise).stem for exercise in exercises]
            else:
                exercises = sorted(exercises)

            for exercise in exercises:

                if camera_id == '60457274':
                    viewpoint = 'view_front_left'
                elif camera_id == '65906101':
                    viewpoint = 'view_front_right'
                elif camera_id == '50591643':
                    viewpoint = 'view_back_left'
                elif camera_id == '58860488':
                    viewpoint = 'view_back_right'
                else:
                    print(f"Unknown camera id {camera_id}")
                    sys.exit(1)

                print(f"Processing {participant}/{viewpoint}/{exercise}")

                method_output = pathlib.Path(os.path.join(output, participant, exercise, viewpoint))

                method_output.mkdir(parents=True, exist_ok=True)

                run_method_video(args.method, os.path.join(dataset_path, 'train', participant, 'videos', camera_id, exercise + '.mp4'), method_output, args.device, args.render, args.cpu_cores, args.save_mesh, args.batch_size, args.checkpoint)
        
def main(args):

    get_fit3d_predictions(args.dataset_path, args.output_path, args.participants, args.camera_ids, args.exercises)

if __name__ == "__main__":
    args = get_args()
    main(args)

