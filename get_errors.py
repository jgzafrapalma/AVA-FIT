import os
import pickle
import joblib
import argparse

import numpy as np
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(description="Get top errors for participant, viewpoint and exercise")

    parser.add_argument("--evaluation_file", type=str, required=True, help="path to the evaluation file")
    parser.add_argument("--exercises", nargs='+', required=False, help="Exercises to process")
    parser.add_argument("--viewpoints", nargs='+', required=False, help="Viewpoints to process")
    parser.add_argument("--participants", nargs='+', required=False, help="Participants to process")
    parser.add_argument("--metrics", nargs='+', required=False, help="Metrics to process")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")

    return parser.parse_args()


def get_top_min_errors(results, participants, viewpoints, exercises, metrics, X=10):
    # Crear dataframes para almacenar los valores medios por punto de vista
    average_metrics_df = pd.DataFrame(columns=metrics, index=['view_back_left', 'view_back_right', 'view_front_left', 'view_front_right'])

    # Diccionarios para almacenar los X fotogramas con mayor y menor error
    top_errors_per_case = {}
    mean_errors = {}
    median_errors = {}

    # Inicializar listas para calcular promedios
    metrics_totals = {viewpoint: {metric: [] for metric in metrics} for viewpoint in average_metrics_df.index}

    # Recorrer los datos para calcular los valores medios y obtener los fotogramas con mayor y menor error
    for participant in participants:
        top_errors_per_case[participant] = {}
        mean_errors[participant] = {}
        median_errors[participant] = {}

        for exercise in exercises:
            top_errors_per_case[participant][exercise] = {}
            mean_errors[participant][exercise] = {}
            median_errors[participant][exercise] = {}

            for viewpoint in viewpoints:
                top_errors_per_case[participant][exercise][viewpoint] = {}
                mean_errors[participant][exercise][viewpoint] = {}
                median_errors[participant][exercise][viewpoint] = {}

                paths = results[participant][exercise][viewpoint]['paths']

                for metric in metrics:
                    values = results[participant][exercise][viewpoint][metric]
                    metrics_totals[viewpoint][metric].extend(values)

                    # Top X errores máximos
                    top_errors_per_case[participant][exercise][viewpoint][metric] = [
                        (paths[i], values[i]) for i, _ in sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:X]
                    ]

                    # Mean and median value
                    mean_val = float(np.mean(values))
                    median_val = float(np.median(values))

                    # Find closest value to mean
                    mean_idx = int(np.argmin(np.abs(np.array(values) - mean_val)))
                    mean_errors[participant][exercise][viewpoint][metric] = [(
                        paths[mean_idx], values[mean_idx]
                    )]

                    # Find closest value to median
                    median_idx = int(np.argmin(np.abs(np.array(values) - median_val)))
                    median_errors[participant][exercise][viewpoint][metric] = [(
                        paths[median_idx], values[median_idx]
                    )]

    return top_errors_per_case, mean_errors, median_errors




def main(args):

    PATH_HUMMAN_METRICS = args.evaluation_file
    results = joblib.load(PATH_HUMMAN_METRICS)

    if args.viewpoints: 
        viewpoints = args.viewpoints
    else:
        viewpoints = ['view_back_left', 'view_back_right', 'view_front_left', 'view_front_right']
    
    if args.participants:
        participants = args.participants
    else:
        participants = ['s11']

    if args.exercises:
        exercises = args.exercises
    else:
        exercises = sorted(results[list(results.keys())[0]])

    top_errors_per_case, mean_errors, median_errors = get_top_min_errors(results, participants, viewpoints, exercises, args.metrics)

    with open(os.path.join(args.output_path, 'top_error.pkl'), 'wb') as f:

        pickle.dump(top_errors_per_case, f)

    with open(os.path.join(args.output_path, 'mean_error.pkl'), 'wb') as f:

        pickle.dump(mean_errors, f)

    with open(os.path.join(args.output_path, 'median_error.pkl'), 'wb') as f:

        pickle.dump(median_errors, f)

if __name__ == '__main__':

    args = parse_args()

    main(args)
