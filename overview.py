import os
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


regions = [
  "head", "neck", "spine", "spine1", "spine2", "hips",
  "leftShoulder", "rightShoulder",
  "leftArm", "leftForeArm", "leftHand", "leftHandIndex1",
  "rightArm", "rightForeArm", "rightHand", "rightHandIndex1",
  "leftUpLeg", "leftLeg", "leftFoot", "leftToeBase",
  "rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"
]

def get_args():
    parser = argparse.ArgumentParser(
        description="Get bar plots comparing different methods on Fit3D dataset."
    )
    parser.add_argument(
        "--evaluations_path",
        type=str,
        required=True,
        help="Path to the evaluations files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the plots",
    )
    return parser.parse_args()


def plot_grouped_metrics(results, save_path: pathlib.Path):
    # Añadimos 3DBodySam
    method_order = ["4DH", "multihmr", "PromptHMR", "SAM3DBODY"]
    method_label = {
        "4DH": "HMR2.0",
        "multihmr": "MULTI-HMR",
        "PromptHMR": "PromptHMR",
        "SAM3DBODY": "SAM3DBODY",
    }
    method_color = {
        "4DH": "tab:blue",
        "multihmr": "tab:orange",
        "PromptHMR": "tab:red",
        "SAM3DBODY": "tab:green",
    }

    metrics = ["MPJPE", "PA-MPJPE", "MPVPE", "PA-MPVPE"]
    means_keys = ["MPJPE_mean", "PA-MPJPE_mean", "MPVPE_mean", "PA-MPVPE_mean"]
    stds_keys  = ["MPJPE_std",  "PA-MPJPE_std",  "MPVPE_std",  "PA-MPVPE_std"]

    # Filtra solo métodos que realmente estén en los resultados
    method_order = [m for m in method_order if m in results]
    x = np.arange(len(metrics)) * 1.5  # MODIFICADO: Aumentar separación entre grupos
    width = 0.25

    plt.figure(figsize=(12, 5))  # MODIFICADO: Aumentar el ancho de la figura
    for i, m in enumerate(method_order):
        offset = (i - (len(method_order) - 1) / 2) * width
        means = [results[m][k] for k in means_keys]
        stds  = [results[m][k] for k in stds_keys]
        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=method_label.get(m, m),
            color=method_color.get(m, None),
        )

    plt.xticks(x, metrics)
    plt.ylabel("Error (mm)")
    plt.title("Fit3D: methods comparison (mean ± std)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    out_file = save_path / "fit3d_metrics_barplot_mean_std.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved plot to: {out_file}")


def plot_mvpe_by_participant(participant_results, save_path: pathlib.Path):
    # Añadimos 3DBodySam
    method_order = ["4DH", "multihmr", "PromptHMR", "SAM3DBODY"]
    method_label = {
        "4DH": "HMR2.0",
        "multihmr": "MULTI-HMR",
        "PromptHMR": "PromptHMR",
        "SAM3DBODY": "SAM3DBODY",
    }
    method_color = {
        "4DH": "tab:blue",
        "multihmr": "tab:orange",
        "PromptHMR": "tab:red",
        "SAM3DBODY": "tab:green",
    }

    participants = sorted(participant_results.keys())
    # Solo métodos que aparezcan al menos en un participante
    method_order = [m for m in method_order if any(m in participant_results[p] for p in participants)]
    if not method_order:
        print("No hay datos para MVPE por participante.")
        return

    x = np.arange(len(participants)) * 1.3  # MODIFICADO: Aumentar separación entre grupos
    width = 0.25

    plt.figure(figsize=(16, 5))  # MODIFICADO: Aumentar el ancho de la figura
    for i, m in enumerate(method_order):
        offset = (i - (len(method_order) - 1) / 2) * width
        means, stds = [], []
        for p in participants:
            if m in participant_results[p]:
                means.append(participant_results[p][m]["MVPE_mean"])
                stds.append(participant_results[p][m]["MVPE_std"])
            else:
                means.append(np.nan)
                stds.append(0.0)

        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=method_label.get(m, m),
            color=method_color.get(m, None),
        )

    plt.xticks(x, participants, rotation=45, ha="right")
    plt.ylabel("MVPE (mm)")
    plt.title("Fit3D: MVPE by participant (mean ± std)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    out_file = save_path / "fit3d_mvpe_by_participant_mean_std.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved plot to: {out_file}")


def plot_top10_mpvpe_exercises(exercise_stats, save_path: pathlib.Path):
    # Añadimos 3DBodySam
    method_order = ["4DH", "multihmr", "PromptHMR", "SAM3DBODY"]
    method_label = {
        "4DH": "HMR2.0",
        "multihmr": "MULTI-HMR",
        "PromptHMR": "PromptHMR",
        "SAM3DBODY": "SAM3DBODY",
    }
    method_color = {
        "4DH": "tab:blue",
        "multihmr": "tab:orange",
        "PromptHMR": "tab:red",
        "SAM3DBODY": "tab:green",
    }

    # Solo métodos con estadísticas de ejercicios
    method_order = [m for m in method_order if m in exercise_stats]
    if not method_order:
        print("No hay datos para MPVPE por ejercicio.")
        return

    # Ranking global: promedio entre métodos disponibles
    all_exercises = set()
    for m in method_order:
        all_exercises |= set(exercise_stats[m].keys())

    global_score = {}
    for ex in all_exercises:
        vals = [exercise_stats[m][ex]["mean"] for m in method_order if ex in exercise_stats[m]]
        if vals:
            global_score[ex] = float(np.mean(vals))

    top10 = [ex for ex, _ in sorted(global_score.items(), key=lambda kv: kv[1], reverse=True)[:10]]

    x = np.arange(len(top10)) * 1.3  # MODIFICADO: Aumentar separación entre grupos
    width = 0.25

    plt.figure(figsize=(16, 5))  # MODIFICADO: Aumentar el ancho de la figura
    for i, m in enumerate(method_order):
        offset = (i - (len(method_order) - 1) / 2) * width
        means = [exercise_stats[m].get(ex, {"mean": np.nan})["mean"] for ex in top10]
        stds  = [exercise_stats[m].get(ex, {"std": 0.0})["std"] for ex in top10]

        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=method_label.get(m, m),
            color=method_color.get(m, None),
        )

    plt.xticks(x, top10, rotation=45, ha="right")
    plt.ylabel("MPVPE (mm)")
    plt.title("Fit3D: Top-10 exercises with highest MPVPE (mean ± std)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    out_file = save_path / "fit3d_top10_exercises_mpvpe_mean_std.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved plot to: {out_file}")


def evaluation(eval_path: pathlib.Path, save_path: pathlib.Path):
    results = {}
    participant_results = {}
    exercise_values = {}

    results_regions = {}

    for method in os.listdir(eval_path):
        method_dir = os.path.join(eval_path, method)
        if not os.path.isdir(method_dir):
            continue

        method_name = method.split("_")[2]
        results[method_name] = {}
        results_regions[method_name] = {}

        # Inicializamos listas para las métricas globales
        mean_mpjpe, mean_pa_mpjpe, mean_mpvpe, mean_pa_mpvpe = [], [], [], []
        std_mpjpe, std_pa_mpjpe, std_mpvpe, std_pa_mpvpe = [], [], [], []

        # --- NUEVO: Diccionario para acumular valores de regiones por participante ---
        region_accumulator = {} 
        # Estructura esperada: {'head': [val_p1, val_p2...], 'neck': [...], ...}

        for participant in os.listdir(method_dir):
            csv_path = os.path.join(method_dir, participant, "summary_errors_all_viewpoints.csv")
            if not os.path.exists(csv_path):
                continue

            # 1. Procesamiento del summary (MPJPE, MPVPE, etc.)
            df = pd.read_csv(csv_path, index_col=0)

            # ... (Aquí va tu código existente para participant_results y exercise_values) ...
            part_mvpe_mean = float(df["MPVPE"].mean())
            part_mvpe_std  = float(df["MPVPE"].std())
            participant_results.setdefault(participant, {})
            participant_results[participant][method_name] = {
                "MVPE_mean": part_mvpe_mean,
                "MVPE_std": part_mvpe_std,
            }
            
            for ex_name, ex_mpvpe in df["MPVPE"].items():
                exercise_values.setdefault(method_name, {})
                exercise_values[method_name].setdefault(ex_name, []).append(float(ex_mpvpe))

            mean_mpjpe.append(df["MPJPE"].mean())
            mean_pa_mpjpe.append(df["PA-MPJPE"].mean())
            mean_mpvpe.append(df["MPVPE"].mean())
            mean_pa_mpvpe.append(df["PA-MPVPE"].mean())

            std_mpjpe.append(df["MPJPE"].std())
            std_pa_mpjpe.append(df["PA-MPJPE"].std())
            std_mpvpe.append(df["MPVPE"].std())
            std_pa_mpvpe.append(df["PA-MPVPE"].std())

            # 2. Procesamiento de regiones (MODIFICADO)
            regions_csv_path = os.path.join(method_dir, participant, "regions_errors_all_viewpoints.csv")
            
            if os.path.exists(regions_csv_path):
                df_regions = pd.read_csv(regions_csv_path, index_col=0)
                
                for region in df_regions.columns:
                    if region not in region_accumulator:
                        region_accumulator[region] = []
                    
                    # Guardamos la media de ESTE participante para ESTA región
                    region_mean_participant = df_regions[region].mean()
                    region_accumulator[region].append(region_mean_participant)

        # --- FUERA DEL BUCLE DE PARTICIPANTES ---
        
        # Calcular medias globales de las métricas principales
        results[method_name]["MPJPE_mean"] = float(np.mean(mean_mpjpe)) if mean_mpjpe else np.nan
        results[method_name]["PA-MPJPE_mean"] = float(np.mean(mean_pa_mpjpe)) if mean_pa_mpjpe else np.nan
        results[method_name]["MPVPE_mean"] = float(np.mean(mean_mpvpe)) if mean_mpvpe else np.nan
        results[method_name]["PA-MPVPE_mean"] = float(np.mean(mean_pa_mpvpe)) if mean_pa_mpvpe else np.nan

        results[method_name]["MPJPE_std"] = float(np.mean(std_mpjpe)) if std_mpjpe else 0.0
        results[method_name]["PA-MPJPE_std"] = float(np.mean(std_pa_mpjpe)) if std_pa_mpjpe else 0.0
        results[method_name]["MPVPE_std"] = float(np.mean(std_mpvpe)) if std_mpvpe else 0.0
        results[method_name]["PA-MPVPE_std"] = float(np.mean(std_pa_mpvpe)) if std_pa_mpvpe else 0.0

        # --- NUEVO: Calcular la media global por región ---
        for region, values in region_accumulator.items():
            # Calculamos la media de todos los participantes para esa región
            results_regions[method_name][region] = float(np.mean(values))
        
    #Save to a json file
    regions_out_path = save_path / "regions_mean_errors.json"
    with open(regions_out_path, "w") as f:
        import json
        json.dump(results_regions, f, indent=4)
    print(f"Saved region mean errors to: {regions_out_path}")

    # Plot 1: 4 métricas
    plot_grouped_metrics(results, save_path)

    # Plot 2: MVPE por participante
    plot_mvpe_by_participant(participant_results, save_path)

    # Plot 3: Top-10 ejercicios por MPVPE
    exercise_stats = {}
    for m, ex_dict in exercise_values.items():
        exercise_stats[m] = {}
        for ex_name, vals in ex_dict.items():
            exercise_stats[m][ex_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    plot_top10_mpvpe_exercises(exercise_stats, save_path)


def main(args):
    evaluations_path = pathlib.Path(args.evaluations_path)
    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    evaluation(evaluations_path, save_path)


if __name__ == "__main__":
    args = get_args()
    main(args)