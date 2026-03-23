# Script para obtener las métricas a partir de las evaluaciones, tambien obtener las gráficas y los fps.

import joblib
import os
import pathlib
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Summary results of the errors")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results files")
    parser.add_argument("--preds_path", type=str, required=True, help="Path to the predictions files")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the summary")
    parser.add_argument("--fps", action="store_true", default=False, help="Calculate FPS")
    
    return parser.parse_args()

def get_fps(preds_path: str, out_folder: str) -> None:
    """Calculate FPS for each method and dataset from the predictions folder."""
    fps_dict = defaultdict(lambda: defaultdict(list))  # fps_dict[dataset][method] = [fps_values]

    for method_folder in sorted(os.listdir(preds_path)):
        method_path = os.path.join(preds_path, method_folder)
        if not os.path.isdir(method_path):
            continue
        dataset = method_folder.split('_')[0]
        method = method_folder.split('_')[2]
        is_avafit = dataset == "avafit"

        for participant in sorted(os.listdir(method_path)):
            participant_path = os.path.join(method_path, participant)
            if not os.path.isdir(participant_path):
                continue
            for exercise in sorted(os.listdir(participant_path)):
                exercise_path = os.path.join(participant_path, exercise)
                if not os.path.isdir(exercise_path):
                    continue
                for viewpoint in sorted(os.listdir(exercise_path)):
                    viewpoint_path = os.path.join(exercise_path, viewpoint)
                    if not os.path.isdir(viewpoint_path):
                        continue

                    if is_avafit:
                        # AVA-FIT: one more level for repetitions
                        for repetition in sorted(os.listdir(viewpoint_path)):
                            repetition_path = os.path.join(viewpoint_path, repetition)
                            if not os.path.isdir(repetition_path):
                                continue
                            fps = _compute_fps(repetition_path)
                            if fps is not None:
                                fps_dict[dataset][method].append(fps)
                    else:
                        fps = _compute_fps(viewpoint_path)
                        if fps is not None:
                            fps_dict[dataset][method].append(fps)

    for dataset in fps_dict:
        output_file = os.path.join(out_folder, f"fps_means_{dataset}.txt")
        with open(output_file, "w") as f:
            for method, fps_list in fps_dict[dataset].items():
                mean_fps = np.mean(fps_list)
                f.write(f"{method}: {mean_fps:.2f}\n")
        print(f"FPS guardados en: {output_file}")


def _compute_fps(sequence_path: str) -> float:
    """Compute FPS for a single sequence given its directory path."""
    preds_file = os.path.join(sequence_path, "preds.npz")
    time_file = os.path.join(sequence_path, "execution_time.txt")

    if not os.path.isfile(preds_file) or not os.path.isfile(time_file):
        return None

    data = np.load(preds_file)
    num_frames = data['v3d'].shape[0]

    with open(time_file, "r") as f:
        time_line = f.readline()
        time_str = time_line.split(": ")[1].strip()
        h, m, s = map(float, time_str.split(":"))
        total_seconds = h * 3600 + m * 60 + s

    return num_frames / total_seconds


def summary_errors(evaluation_path: str, preds_path: str, out_folder: str, fps: bool) -> None:
    # Esto permite acceder a overall_metrics[dataset][method][metric] sin crear la estructura manualmente
    overall_metrics_only_body = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
    overall_metrics_whole_body = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_metrics_hands = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_metrics_feet = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_metrics_region = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    overall_metrics_exercise = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    TARGET_METRICS_ONLY_BODY = ["PVE", "PA-PVE", "MPJPE", "PA-MPJPE"]
    TARGET_METRICS_WHOLE_BODY = ["PVE_SMPLX", "PA-PVE_SMPLX"]
    TARGET_METRICS_HANDS = ["PVE_HANDS", "PA-PVE_HANDS"]
    TARGET_METRICS_FEET = ["PVE_FEETS", "PA-PVE_FEETS"]

    def accumulate_metrics(data, dataset, method, exercise):
        """Accumulate metrics from a single sequence (viewpoint for fit3d, repetition for avafit)."""
        for metric in data:
            if metric in TARGET_METRICS_ONLY_BODY:
                overall_metrics_only_body[dataset][method][metric].extend(data[metric])
            elif metric in TARGET_METRICS_WHOLE_BODY:
                overall_metrics_whole_body[dataset][method][metric].extend(data[metric])
                overall_metrics_exercise[dataset][method][exercise][metric].extend(data[metric])
                if 'region_errors' in data:
                    for region in data['region_errors']:
                        overall_metrics_region[dataset][method][region][metric].extend(data['region_errors'][region][metric])
            elif metric in TARGET_METRICS_HANDS:
                overall_metrics_hands[dataset][method][metric].extend(data[metric])
            elif metric in TARGET_METRICS_FEET:
                overall_metrics_feet[dataset][method][metric].extend(data[metric])


    for evaluation_folder in sorted(os.listdir(evaluation_path)):
        dataset = evaluation_folder.split("_")[0]
        method = evaluation_folder.split("_")[2]
        results = joblib.load(os.path.join(evaluation_path, evaluation_folder, "results.pkl"))

        is_avafit = dataset == "avafit"

        for participant in results:
            for exercise in results[participant]:
                for viewpoint in results[participant][exercise]:
                    if is_avafit:
                        # AVA-FIT: participant > exercise > viewpoint > repetition > metric
                        for repetition in results[participant][exercise][viewpoint]:
                            accumulate_metrics(results[participant][exercise][viewpoint][repetition], dataset, method, exercise)
                    else:
                        # Fit3D: participant > exercise > viewpoint > metric
                        accumulate_metrics(results[participant][exercise][viewpoint], dataset, method, exercise)


    # Get mean values
    for dataset in overall_metrics_only_body:
        for method in overall_metrics_only_body[dataset]:
            for metric in overall_metrics_only_body[dataset][method]:
                overall_metrics_only_body[dataset][method][metric] = np.mean(overall_metrics_only_body[dataset][method][metric])
    
    for dataset in overall_metrics_hands:
        for method in overall_metrics_hands[dataset]:
            for metric in overall_metrics_hands[dataset][method]:
                overall_metrics_hands[dataset][method][metric] = np.mean(overall_metrics_hands[dataset][method][metric])

    for dataset in overall_metrics_feet:
        for method in overall_metrics_feet[dataset]:
            for metric in overall_metrics_feet[dataset][method]:
                overall_metrics_feet[dataset][method][metric] = np.mean(overall_metrics_feet[dataset][method][metric])

    for dataset in overall_metrics_whole_body:
        for method in overall_metrics_whole_body[dataset]:
            for metric in overall_metrics_whole_body[dataset][method]:
                overall_metrics_whole_body[dataset][method][metric] = np.mean(overall_metrics_whole_body[dataset][method][metric])
    
    for dataset in overall_metrics_region:
        for method in overall_metrics_region[dataset]:
            for region in overall_metrics_region[dataset][method]:
                for metric in overall_metrics_region[dataset][method][region]:
                        overall_metrics_region[dataset][method][region][metric] = np.mean(overall_metrics_region[dataset][method][region][metric])

    for dataset in overall_metrics_exercise:
        for method in overall_metrics_exercise[dataset]:
            for exercise in overall_metrics_exercise[dataset][method]:
                for metric in overall_metrics_exercise[dataset][method][exercise]:
                    overall_metrics_exercise[dataset][method][exercise][metric] = np.mean(overall_metrics_exercise[dataset][method][exercise][metric])

    os.makedirs(out_folder, exist_ok=True)

    for dataset in overall_metrics_only_body:
        df = pd.DataFrame.from_dict(overall_metrics_only_body[dataset], orient='index')
                
        csv_path = os.path.join(out_folder, f"{dataset}_metrics_only-body.csv")
        df.to_csv(csv_path, index=True, float_format='%.1f')
        
        print(f"Resultados guardados en: {csv_path}")

    for dataset in overall_metrics_whole_body:
        df = pd.DataFrame.from_dict(overall_metrics_whole_body[dataset], orient='index')
                
        csv_path = os.path.join(out_folder, f"{dataset}_metrics_whole-body.csv")
        df.to_csv(csv_path, index=True, float_format='%.1f')
        
        print(f"Resultados guardados en: {csv_path}")

    
    for dataset in overall_metrics_hands:
        df = pd.DataFrame.from_dict(overall_metrics_hands[dataset], orient='index')
                
        csv_path = os.path.join(out_folder, f"{dataset}_metrics_hands.csv")
        df.to_csv(csv_path, index=True, float_format='%.1f')
        
        print(f"Resultados guardados en: {csv_path}")

    for dataset in overall_metrics_feet:
        df = pd.DataFrame.from_dict(overall_metrics_feet[dataset], orient='index')
                
        csv_path = os.path.join(out_folder, f"{dataset}_metrics_feet.csv")
        df.to_csv(csv_path, index=True, float_format='%.1f')
        
        print(f"Resultados guardados en: {csv_path}")


    for dataset in overall_metrics_region:
        for method in overall_metrics_region[dataset]:
            
            df = pd.DataFrame.from_dict(overall_metrics_region[dataset][method], orient='index')               
            csv_path = os.path.join(out_folder, f"{dataset}_{method}_metrics_region.csv")
            df.to_csv(csv_path, index=True, float_format='%.1f')
            
            print(f"Resultados guardados en: {csv_path}")
    
    for dataset in overall_metrics_exercise:
        for method in overall_metrics_exercise[dataset]:
            
            df = pd.DataFrame.from_dict(overall_metrics_exercise[dataset][method], orient='index')
            
            csv_path = os.path.join(out_folder, f"{dataset}_{method}_metrics_exercise.csv")
            df.to_csv(csv_path, index=True, float_format='%.1f')
            
            print(f"Resultados guardados en: {csv_path}")
            
    if fps:
        # Calculate FPS
        print("Calculando FPS...")
        get_fps(preds_path, out_folder)
        print("FPS calculados correctamente")

def main(args):
    
    summary_errors(args.results_path, args.preds_path, args.out_folder, args.fps)

if __name__ == "__main__":
    args = get_args()
    main(args)