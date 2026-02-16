import os
import sys
import pathlib
import argparse
import subprocess
from utils import HOME, CONDA_INIT, METHODS_DIR

new_home = os.environ.copy()

new_home["HOME"] = HOME

def get_args():
    parser = argparse.ArgumentParser(description="Get predictions from 3D reconstruction models for the AVA-FIT dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--method", type=str, required=True, help="Available models: [4D-Humans]")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    parser.add_argument("--camera_ids", nargs='+', required=False, help="Viewpoints to process")
    parser.add_argument("--exercises", nargs='+', required=False, help="Exercises to process")
    parser.add_argument("--repetitions", nargs='+', required=False, help="Repetitions to process")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to checkpoint")
    parser.add_argument("--model_name", type=str, required=False, help="model name")
    parser.add_argument("--device", type=int, required=False, default=0, help='device to use')
    parser.add_argument("--render", dest='render', action='store_true', default=False, help='If set, save rendered results')
    parser.add_argument("--cpu_cores", type=str, default="20-39,60-79", help="CPU cores to use with taskset")
    parser.add_argument("--batch_size", type=int, required=False, help="path to output directory")
    parser.add_argument("--save_mesh", dest='save_mesh', action='store_true', default=False, help='save .obj file with the meshes')
    parser.add_argument("--extra_views", dest='extra_views', action='store_true', default=False, help='render more views of the mesh rendered')

    return parser.parse_args()


def run_method_video(method, video, target_dir, device, render, cpu_cores, save_mesh, batch_size = None, checkpoint = None, model_name = None, extra_views = None) -> None:

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
    if model_name:
        base_command += f" --model_name {model_name}"
    if extra_views:
        base_command += f" --extra_views"

    #command = f'/bin/bash -c "{base_command}"'

    subprocess.run(base_command, env=new_home, shell=True, executable='/bin/bash')

def get_avafit_predictions(args):

    for video in sorted(os.listdir(os.path.join(args.dataset_path, 'videos'))):
        video_name_split = video.split('_')
        participant = video_name_split[1] + '_' + video_name_split[2]
        exercise = video_name_split[3]
        repetition = video_name_split[4]
        viewpoint = video_name_split[5]
    
        if args.participants and participant not in args.participants:
            continue
        if args.camera_ids and viewpoint not in args.camera_ids:
            continue
        if args.exercises and exercise not in args.exercises:
            continue
        if args.repetitions and repetition not in args.repetitions:
            continue

        print(f"Processing {participant}/{viewpoint}/{exercise}/{repetition}")

        method_output = pathlib.Path(os.path.join(args.output_path, participant, exercise, viewpoint, repetition))

        if not args.exercises and method_output.exists() and method_output.is_dir():
            # Check if exits de preds.npz file
            preds_file = method_output / "preds.npz"
            if preds_file.exists():
                print(f"[SKIP] Ya existen predicciones para {participant}/{viewpoint}/{exercise}/{repetition}")
                continue

        method_output.mkdir(parents=True, exist_ok=True)

        run_method_video(args.method, os.path.join(args.dataset_path, 'videos', video), method_output, args.device, args.render, args.cpu_cores, args.save_mesh, args.batch_size, args.checkpoint, args.model_name, args.extra_views)
        
def main(args):

    get_avafit_predictions(args)

if __name__ == "__main__":
    args = get_args()
    main(args)

