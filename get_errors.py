import os
import json
import joblib
import argparse
import pathlib
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Get top errors for participant, viewpoint and exercise")
    parser.add_argument("--evaluation_path", type=str, required=True, help="path to the evaluation directory")
    parser.add_argument("--metric", type=str, required=True, help="Metrics to process")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--n", type=int, default=10, help="Number of top errors to get")
    return parser.parse_args()


def _compute_errors(entry, metric, paths, X):
    values = entry[metric]
    arr = np.array(values)

    # Top X fotogramas con mayor error
    top_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:X]
    top = {metric: [[paths[i], float(values[i])] for i in top_indices]}

    # Fotograma más cercano a la media
    mean_val = float(np.mean(arr))
    mean_idx = int(np.argmin(np.abs(arr - mean_val)))
    mean = {metric: [[paths[mean_idx], float(values[mean_idx])]]}

    # Fotograma más cercano a la mediana
    median_val = float(np.median(arr))
    median_idx = int(np.argmin(np.abs(arr - median_val)))
    median = {metric: [[paths[median_idx], float(values[median_idx])]]}

    return top, mean, median


def get_top_mean_median_errors(results, metric, X, dataset_name):
    top_errors_per_case, mean_errors, median_errors = {}, {}, {}
    for participant, exercises in results.items():
        top_errors_per_case[participant] = {}
        mean_errors[participant] = {}
        median_errors[participant] = {}
        for exercise, viewpoints in exercises.items():
            top_errors_per_case[participant][exercise] = {}
            mean_errors[participant][exercise] = {}
            median_errors[participant][exercise] = {}
            for viewpoint, vp_data in viewpoints.items():
                if dataset_name == 'avafit':
                    top_errors_per_case[participant][exercise][viewpoint] = {}
                    mean_errors[participant][exercise][viewpoint] = {}
                    median_errors[participant][exercise][viewpoint] = {}
                    for repetition, rep_entry in vp_data.items():
                        t, m, med = _compute_errors(rep_entry, metric, rep_entry['paths'], X)
                        top_errors_per_case[participant][exercise][viewpoint][repetition] = t
                        mean_errors[participant][exercise][viewpoint][repetition] = m
                        median_errors[participant][exercise][viewpoint][repetition] = med
                else:
                    t, m, med = _compute_errors(vp_data, metric, vp_data['paths'], X)
                    top_errors_per_case[participant][exercise][viewpoint] = t
                    mean_errors[participant][exercise][viewpoint] = m
                    median_errors[participant][exercise][viewpoint] = med
    return top_errors_per_case, mean_errors, median_errors


def main(args):
    for evaluation_folder in os.listdir(args.evaluation_path):
        dataset_name = evaluation_folder.split('_')[0]
        results = joblib.load(os.path.join(args.evaluation_path, evaluation_folder, 'results.pkl'))
        top_errors_per_case, mean_errors, median_errors = get_top_mean_median_errors(
            results, args.metric, args.n, dataset_name
        )
        output_path = os.path.join(args.output_path, f"{evaluation_folder}_{args.metric}")
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'top_error.json'), 'w') as f:
            json.dump(top_errors_per_case, f, indent=4, cls=NumpyEncoder)
        with open(os.path.join(output_path, 'mean_error.json'), 'w') as f:
            json.dump(mean_errors, f, indent=4, cls=NumpyEncoder)
        with open(os.path.join(output_path, 'median_error.json'), 'w') as f:
            json.dump(median_errors, f, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    args = parse_args()
    main(args)