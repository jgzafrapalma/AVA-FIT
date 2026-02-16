import joblib
import os
import pathlib
import argparse
import pandas as pd
import numpy as np  # Necesario si los datos vienen en arrays de numpy
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Summary results of the errors")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results files")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the summary")
    
    return parser.parse_args()

def summary_errors(evaluation_path: str, out_folder: str) -> None:
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

    for evaluation_folder in sorted(os.listdir(evaluation_path)):
        dataset = evaluation_folder.split("_")[0]
        method = evaluation_folder.split("_")[2]
        results = joblib.load(os.path.join(evaluation_path, evaluation_folder, "results.pkl"))

        for participant in results:
            for exercise in results[participant]:
                for viewpoint in results[participant][exercise]:
                    for metric in results[participant][exercise][viewpoint]:
                        if metric in TARGET_METRICS_ONLY_BODY:
                            overall_metrics_only_body[dataset][method][metric].extend(results[participant][exercise][viewpoint][metric])
                        elif metric in TARGET_METRICS_WHOLE_BODY:
                            overall_metrics_whole_body[dataset][method][metric].extend(results[participant][exercise][viewpoint][metric]) 
                            overall_metrics_exercise[dataset][method][exercise][metric].extend(results[participant][exercise][viewpoint][metric])
                            for region in results[participant][exercise][viewpoint]['region_errors']:
                                overall_metrics_region[dataset][method][region][metric].extend(results[participant][exercise][viewpoint]['region_errors'][region][metric])
                        elif metric in TARGET_METRICS_HANDS:
                            overall_metrics_hands[dataset][method][metric].extend(results[participant][exercise][viewpoint][metric])
                        elif metric in TARGET_METRICS_FEET:
                            overall_metrics_feet[dataset][method][metric].extend(results[participant][exercise][viewpoint][metric])
                        

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
        df_whole_body = pd.DataFrame.from_dict(overall_metrics_whole_body[dataset], orient='index')
                
        csv_path = os.path.join(out_folder, f"{dataset}_metrics_only-body.csv")
        df.to_csv(csv_path, index=True, float_format='%.1f')

        csv_path_whole_body = os.path.join(out_folder, f"{dataset}_metrics_whole-body.csv")
        df_whole_body.to_csv(csv_path_whole_body, index=True, float_format='%.1f')
        
        print(f"Resultados guardados en: {csv_path}")
        print(f"Resultados guardados en: {csv_path_whole_body}")

    
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

def main(args):
    
    summary_errors(args.results_path, args.out_folder)

if __name__ == "__main__":
    args = get_args()
    main(args)