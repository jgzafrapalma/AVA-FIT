
import joblib
import pathlib
import argparse
import pandas as pd
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="Summary results of the errors")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results file")
    parser.add_argument("--out_folder", type=str, required=False, help="Output folder to save the summary")
    
    return parser.parse_args()


def summary_errors(errors_path: str, out_folder: str) -> None:
    # Crear la carpeta de salida si no existe
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    # Cargar los errores desde el archivo
    results = joblib.load(errors_path)

    # Diccionarios para almacenar datos
    overall_data = defaultdict(lambda: defaultdict(list))  # Promedio general por ejercicio y métrica
    viewpoint_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # Por punto de vista

    for participant in results:
        for exercise in results[participant]:
            for viewpoint in results[participant][exercise]:
                for metric in results[participant][exercise][viewpoint]:
                    if metric in ["MPVPE", "PA-MPVPE", "MPJPE", "PA-MPJPE"]:
                        for error in results[participant][exercise][viewpoint][metric]:
                            # General: ignorando punto de vista
                            overall_data[exercise][metric].append(error)
                            # Por punto de vista
                            viewpoint_data[viewpoint][exercise][metric].append(error)

    # Guardar resumen general
    overall_summary = {
        exercise: {metric: sum(values) / len(values) if values else None
                   for metric, values in metrics.items()}
        for exercise, metrics in overall_data.items()
    }

    df_overall = pd.DataFrame.from_dict(overall_summary, orient='index').round(2)
    df_overall.to_csv(pathlib.Path(out_folder) / "summary_errors_all_viewpoints.csv")
    print(f"Resumen general guardado en: {out_folder}/summary_errors_all_viewpoints.csv")

    # Guardar un CSV por cada punto de vista
    for viewpoint, exercises in viewpoint_data.items():
        summary = {
            exercise: {metric: sum(values) / len(values) if values else None
                       for metric, values in metrics.items()}
            for exercise, metrics in exercises.items()
        }
        df = pd.DataFrame.from_dict(summary, orient='index').round(2)
        csv_name = f"summary_errors_{viewpoint}.csv"
        df.to_csv(pathlib.Path(out_folder) / csv_name)
        print(f"Resumen para punto de vista '{viewpoint}' guardado en: {out_folder}/{csv_name}")



def main(args):

    summary_errors(args.results_path, args.out_folder)

if __name__ == "__main__":
    args = get_args()
    main(args)

