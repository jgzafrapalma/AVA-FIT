import os
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Get FPS metrics for Fit3D dataset predictions")
    parser.add_argument("--preds_path", type=str, required=True, help="Path to the predictions files")
    parser.add_argument("--methods", nargs='+', required=False, help="Methods to evaluate")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the FPS results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to evaluate")
    
    return parser.parse_args()


def get_fps(preds_file, methods, output_path, dataset):
    
    fps_dict = {}

    for method in os.listdir(preds_file):
        name_method = method.split('_')[2]
        dataset_name = method.split('_')[0]
        if dataset_name != dataset:
            continue
        if methods and name_method not in methods:
            continue
        fps_dict[name_method] = []
        for participant in os.listdir(os.path.join(preds_file, method)):
            #print(f"  Participant: {participant}")
            for exercise in os.listdir(os.path.join(preds_file, method, participant)):
                #print(f"    Exercise: {exercise}")
                for viewpoint in os.listdir(os.path.join(preds_file, method, participant, exercise)):
                    #print(f"      Viewpoint: {viewpoint}")
                    preds_path = os.path.join(preds_file, method, participant, exercise, viewpoint, "preds.npz")
                    if not os.path.isfile(preds_path):
                        #print(f"Missing predictions file: {preds_path}")
                        continue
                    data = np.load(preds_path)
                    num_frames = data['v3d'].shape[0]
                    #print(f"        Number of frames: {num_frames}")

                    # Load time file
                    time_file_path = os.path.join(preds_file, method, participant, exercise, viewpoint, "execution_time.txt")
                    if not os.path.isfile(time_file_path):
                        #print(f"Missing time file: {time_file_path}")
                        continue
                    with open(time_file_path, "r") as f:
                        time_line = f.readline()
                        time_str = time_line.split(": ")[1].strip()
                        h, m, s = map(float, time_str.split(":"))
                        total_seconds = h * 3600 + m * 60 + s
                        fps = num_frames / total_seconds if total_seconds > 0 else 0
                        #print(f"        Time elapsed (s): {total_seconds:.2f}")
                        #print(f"        FPS: {fps:.2f}")
                    fps_dict[name_method].append(fps)

    # Save FPS means to a files
    output_file = os.path.join(output_path, f"fps_means_{dataset}.txt")
    with open(output_file, "w") as f:
        for method, fps_list in fps_dict.items():
            mean_fps = np.mean(fps_list)
            f.write(f"{method}: {mean_fps:.2f}\n")

def main(args):

    PREDICTIONS_PATH = args.preds_path
    OUTPUT_PATH = args.output_path  
    DATASET = args.dataset

    get_fps(PREDICTIONS_PATH, args.methods, OUTPUT_PATH, DATASET)

if __name__ == "__main__":
    args = get_args()
    main(args)
