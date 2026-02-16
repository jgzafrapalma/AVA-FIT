import joblib
import pathlib
import argparse
import pandas as pd
import numpy as np  # Necesario si los datos vienen en arrays de numpy
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Summary results of the errors")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results file")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the summary")
    
    return parser.parse_args()

def save_region_csv(data_dict, output_path):
    """
    Función auxiliar para convertir el diccionario de regiones en CSV.
    Estructura esperada de data_dict: 
    { exercise: { region: { metric: [list_of_errors] } } }
    """
    rows = []
    
    # 1. Recorremos los ejercicios (Filas)
    for exercise, regions in data_dict.items():
        row_data = {'Exercise': exercise}
        
        # 2. Recorremos las regiones
        for region, metrics_dict in regions.items():
            # 3. Recorremos las métricas dentro de cada región
            for metric, values in metrics_dict.items():
                if values:
                    # Calculamos la media
                    mean_val = np.mean(values)
                    # Creamos el nombre de la columna: Ej. "rightHand_MPVPE"
                    col_name = f"{region}_{metric}"
                    row_data[col_name] = mean_val
        
        rows.append(row_data)

    if not rows:
        return

    # Crear DataFrame
    df = pd.DataFrame(rows)
    df.set_index('Exercise', inplace=True)
    
    # Redondear y ordenar columnas alfabéticamente para facilitar la lectura
    df = df.round(2)
    df = df.sort_index(axis=1)
    
    df.to_csv(output_path)


def summary_errors(errors_path: str, out_folder: str) -> None:
    # Cargar los errores desde el archivo
    results = joblib.load(errors_path)
    
    base_out_path = pathlib.Path(out_folder)

    print(f"Procesando {len(results)} participantes...")

    # Métricas que nos interesan (para filtrar si hay muchas)
    TARGET_METRICS = ["MPVPE", "PA-MPVPE", "MPJPE", "PA-MPJPE"]

    # Iterar sobre cada participante (NIVEL SUPERIOR)
    for participant, participant_data in results.items():
        
        participant_out_folder = base_out_path / participant
        participant_out_folder.mkdir(parents=True, exist_ok=True)

        # --- ESTRUCTURAS DE DATOS ---
        # 1. Para errores globales (Main body metrics)
        overall_data = defaultdict(lambda: defaultdict(list)) 
        viewpoint_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 

        # 2. Para errores por REGIONES
        # Estructura: [Ejercicio][Region][Metrica] -> lista de errores
        overall_region_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # Estructura: [Vista][Ejercicio][Region][Metrica] -> lista de errores
        viewpoint_region_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        # --- RECOPILACIÓN DE DATOS ---
        for exercise in participant_data:
            for viewpoint in participant_data[exercise]:
                
                # A) Procesar Métricas Generales (Globales)
                for metric in participant_data[exercise][viewpoint]:
                    if metric in TARGET_METRICS:
                        errors_list = participant_data[exercise][viewpoint][metric]
                        # Acumuladores generales
                        overall_data[exercise][metric].extend(errors_list)
                        viewpoint_data[viewpoint][exercise][metric].extend(errors_list)

                # B) Procesar Métricas por Región (NUEVO)
                if 'region_errors' in participant_data[exercise][viewpoint]:
                    regions_dict = participant_data[exercise][viewpoint]['region_errors']
                    
                    for region_name, region_metrics in regions_dict.items():
                        for metric, values in region_metrics.items():
                            if metric in TARGET_METRICS:
                                # Nota: 'values' puede ser numpy array, extend funciona igual
                                # Acumular para el resumen general (todas las vistas juntas)
                                overall_region_data[exercise][region_name][metric].extend(values)
                                
                                # Acumular separado por punto de vista
                                viewpoint_region_data[viewpoint][exercise][region_name][metric].extend(values)

        # --- GENERACIÓN DE CSVs ---

        # 1. CSVs ESTÁNDAR (Lo que ya tenías)
        # ------------------------------------
        if overall_data:
            overall_summary = {
                exercise: {metric: np.mean(values) if values else None
                           for metric, values in metrics.items()}
                for exercise, metrics in overall_data.items()
            }
            df_overall = pd.DataFrame.from_dict(overall_summary, orient='index').round(2)
            df_overall = df_overall.sort_index().sort_index(axis=1)
            df_overall.to_csv(participant_out_folder / "summary_errors_all_viewpoints.csv")

        for viewpoint, exercises in viewpoint_data.items():
            summary = {
                exercise: {metric: np.mean(values) if values else None
                           for metric, values in metrics.items()}
                for exercise, metrics in exercises.items()
            }
            df = pd.DataFrame.from_dict(summary, orient='index').round(2)
            df = df.sort_index().sort_index(axis=1)
            df.to_csv(participant_out_folder / f"summary_errors_{viewpoint}.csv")

        # 2. CSVs DE REGIONES (NUEVO)
        # ------------------------------------
        
        # A) Resumen de regiones promediando todas las vistas
        if overall_region_data:
            out_file = participant_out_folder / "regions_errors_all_viewpoints.csv"
            save_region_csv(overall_region_data, out_file)

        # B) Resumen de regiones por cada vista
        for viewpoint, exercises_data in viewpoint_region_data.items():
            out_file = participant_out_folder / f"regions_errors_{viewpoint}.csv"
            save_region_csv(exercises_data, out_file)
        
        print(f"✓ Datos generados para participante: {participant}")

def main(args):
    summary_errors(args.results_path, args.out_folder)

if __name__ == "__main__":
    args = get_args()
    main(args)