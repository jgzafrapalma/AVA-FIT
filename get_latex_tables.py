import joblib
import os
import pathlib
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# --- CONFIGURACIÓN Y MAPEOS (Extraídos de source: 1) ---

AVAFIT_MAPPING = {
    "A001": "Svend Press", "A002": "Kneeling Push-ups",
    "A003": "Standing Straight-arm Chest Press With Resistance Band",
    "A004": "Small Dumbbell Floor Flies", "A005": "Chest Fly", "A006": "Push-ups",
    "A007": "Bent-over W-shape Stretch", "A008": "Bent-over Y-shape Stretch",
    "A009": "Bent-over A-shape Stretch", "A010": "Squat With Arm Lift",
    "A011": "Breaststroke Arm Pull", "A012": "Prone Y-shape Stretch",
    "A013": "Lateral Raise Forward Circles", "A014": "Lateral Raise Backward Circles",
    "A015": "Lying Shoulder Joint Upward Round", "A016": "Lying Shoulder Joint Downward Round",
    "A017": "Bare-handed Full Lateral Raise", "A018": "Bare-handed Cuban Press",
    "A019": "Fortune Cat", "A020": "Dumbbell Curls", "A021": "Alternate Dumbbell Curls",
    "A022": "Right-side Kettlebell Bent-over Row", "A023": "Wrist Joint Warm-up",
    "A024": "Right-side Bent-over Tricep Extension With Resistance Band",
    "A025": "Bent-over Dumbbell Tricep Extension", "A026": "Nod And Raise Head",
    "A027": "Two-way Head Turn", "A028": "Shrug And Sink The Shoulders",
    "A029": "Four-way Nod Head", "A030": "Lying Alternate Upper-half Leg Raise",
    "A031": "Half Roll Back", "A032": "Kneeling Right-side Torso Twist",
    "A033": "Left-side Knee Raise And Abdominal Muscles Contract",
    "A034": "Kneeling Right Leg Backward Stretch", "A035": "Kneeling Right Arm Raise",
    "A036": "Shoulder Bridge", "A037": "Prone Press-ups", "A038": "Bent-over Torso Rotation",
    "A039": "Lying Arm Pull", "A040": "Prone Press Up With Torso Rotation",
    "A041": "Breaststroke Push-ups", "A042": "Sit-ups",
    "A043": "Side-lying Left Leg Forward Raise", "A044": "Right Leg Reverse Lunge",
    "A045": "Kneeling Left Knee Lift", "A046": "Alternate Reverse Lunge",
    "A047": "Side-lying Right Leg Backward Kick", "A048": "Left Leg Lunge With Knee Lift",
    "A049": "Sumo Squat", "A050": "Straight Leg Calf Raise", "A051": "Squat Jump",
    "A052": "Squat With Alternate Knee Lift", "A053": "Standing Alternate Butt Kick",
    "A054": "Knee Warm-up", "A055": "Butt Kicks", "A056": "Jump Left and Right",
    "A057": "Jumping Jacks", "A058": "High Knee", "A059": "Clap Jacks",
    "A060": "Run In Place With Arm Swing"
}

AVAFIT_ID_TO_CAT = {
    **{f"A00{i}": "Chest" for i in range(1, 7)},
    **{f"A0{i:02d}": "Back" for i in range(7, 13)},
    **{f"A0{i:02d}": "Shoulder" for i in range(13, 20)},
    **{f"A0{i:02d}": "Arm" for i in range(20, 26)},
    **{f"A0{i:02d}": "Neck" for i in range(26, 30)},
    **{f"A0{i:02d}": "Abdomen" for i in range(30, 36)},
    **{f"A0{i:02d}": "Waist" for i in range(36, 43)},
    **{f"A0{i:02d}": "Hip" for i in range(43, 50)},
    **{f"A0{i:02d}": "Leg" for i in range(50, 55)},
    **{f"A0{i:02d}": "Whole body" for i in range(55, 61)},
}

AVAFIT_CAT_ORDER = ["Chest", "Abdomen", "Back", "Waist", "Shoulder", "Hip", "Arm", "Leg", "Neck", "Whole body"]

METRIC_DISPLAY_NAME = {
    "MPVPE": "PVE",
    "PA-MPVPE": "PA-PVE",
    "MPJPE": "MPJPE",
    "PA-MPJPE": "PA-MPJPE"
}

# --- FUNCIONES ---

def get_args():
    parser = argparse.ArgumentParser(description="Get latex tables from the results")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results files")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the summary")
    parser.add_argument("--dataset", type=str, required=True, choices=["fit3d", "avafit"], help="Dataset name")
    return parser.parse_args()

def get_tables(evaluation_path: str, out_folder: str, dataset: str) -> None:
    if not os.path.exists(evaluation_path): return
    df = pd.read_csv(evaluation_path)
    df = df.rename(columns={df.columns[0]: "Method"})
    
    method_name_mapping = {"4DH": "HMR2.0", "multihmr": "Multi-HMR", "SAM3DBODY": "3DB-DINOv3"}
    df["Method"] = df["Method"].replace(method_name_mapping)
    method_order = ["HMR2.0", "Multi-HMR", "prompthmr", "3DB-DINOv3"]
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values('Method').reset_index(drop=True)
    
    metrics = ["MPJPE", "PA-MPJPE", "MPVPE", "PA-MPVPE"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    latex_str = ["\\begin{table}[h]", "\\centering", "\\begin{tabular}{l" + "c" * len(available_metrics) + "}",
                 f"& \\multicolumn{{{len(available_metrics)}}}{{c}}{{{dataset.upper()}}} \\\\",
                 f"\\cmidrule(lr){{2-{len(available_metrics)+1}}}"]
    
    header_row = ["Method"] + [f"{METRIC_DISPLAY_NAME.get(m, m)} $\\downarrow$" for m in available_metrics]
    latex_str.append(" & ".join(header_row) + " \\\\")
    latex_str.append("\\midrule")
    latex_str.append("\\rowcolor{gray!20}")
    latex_str.append("\\textit{Body-only} & " + " & ".join([""] * len(available_metrics)) + " \\\\")
    
    for _, row in df.iterrows():
        method_name = str(row["Method"])
        row_str = [method_name] + [f"{row[m]:.1f}" if isinstance(row[m], float) else str(row[m]) for m in available_metrics]
        latex_str.append(" & ".join(row_str) + " \\\\")
        if method_name == "HMR2.0":
            latex_str.append("\\rowcolor{gray!20}")
            latex_str.append("\\textit{Whole-body} & " + " & ".join([""] * len(available_metrics)) + " \\\\")
    
    latex_str.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, f"{dataset}_table.tex"), "w") as f:
        f.write("\n".join(latex_str))

def get_region_metrics_table(results_folder: str, out_folder: str, dataset: str) -> None:
    methods = {"Multi-HMR": f"{dataset}_multihmr_metrics_region.csv", 
               "PromptHMR": f"{dataset}_PromptHMR_metrics_region.csv", 
               "SAM3DBODY": f"{dataset}_SAM3DBODY_metrics_region.csv"}
    csv_metrics = ["MPVPE", "PA-MPVPE"]
    data = defaultdict(lambda: defaultdict(dict))
    regions = []

    for method_name, filename in methods.items():
        file_path = os.path.join(results_folder, filename)
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        reg_col = "Region" if "Region" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            region = row[reg_col]
            if region not in regions: regions.append(region)
            for m in csv_metrics:
                col = f"{m}_SMPLX" if f"{m}_SMPLX" in df.columns else m
                if col in df.columns: data[region][method_name][m] = row[col]

    if not regions: return
    
    latex_str = ["\\begin{table}[h]", "\\centering", "\\resizebox{\\textwidth}{!}{"]
    num_mth, num_mtr = len(methods), len(csv_metrics)
    latex_str.append(f"\\begin{{tabular}}{{l{'c' * (num_mth * num_mtr)}}}")
    latex_str.append("\\toprule")
    latex_str.append(" & " + " & ".join([f"\\multicolumn{{{num_mtr}}}{{c}}{{{m}}}" for m in methods.keys()]) + " \\\\")
    latex_str.append(" ".join([f"\\cmidrule(lr){{{2 + i*num_mtr}-{1 + (i+1)*num_mtr}}}" for i in range(num_mth)]))
    
    metric_headers = [f"{METRIC_DISPLAY_NAME.get(m, m)} $\\downarrow$" for _ in range(num_mth) for m in csv_metrics]
    latex_str.append("Region & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")
    
    for r in regions:
        row = [str(r)]
        for mth in methods:
            for mtr in csv_metrics:
                val = data[r][mth].get(mtr, '-')
                row.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
        latex_str.append(" & ".join(row) + " \\\\")

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, f"{dataset}_region_table.tex"), "w") as f:
        f.write("\n".join(latex_str))

def get_exercise_metrics_table(results_folder: str, out_folder: str, dataset: str) -> None:
    methods = {"Multi-HMR": f"{dataset}_multihmr_metrics_exercise.csv", 
               "PromptHMR": f"{dataset}_PromptHMR_metrics_exercise.csv", 
               "SAM3DBODY": f"{dataset}_SAM3DBODY_metrics_exercise.csv"}
    csv_metrics = ["MPVPE", "PA-MPVPE"]
    data = defaultdict(lambda: defaultdict(dict))
    
    # Inicializar con todos los ejercicios si es AVAFIT 
    exercise_ids = list(AVAFIT_ID_TO_CAT.keys()) if dataset.lower() == "avafit" else []

    for method_name, filename in methods.items():
        file_path = os.path.join(results_folder, filename)
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        ex_col = "Exercise" if "Exercise" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            ex_id = str(row[ex_col])
            if ex_id not in exercise_ids: exercise_ids.append(ex_id)
            for m in csv_metrics:
                col = f"{m}_SMPLX" if f"{m}_SMPLX" in df.columns else m
                if col in df.columns: data[ex_id][method_name][m] = row[col]

    if not exercise_ids: return

    latex_str = ["\\begin{table}[h]", "\\centering", "\\resizebox{\\textwidth}{!}{"]
    num_mth, num_mtr = len(methods), len(csv_metrics)
    latex_str.append(f"\\begin{{tabular}}{{l{'c' * (num_mth * num_mtr)}}}")
    latex_str.append("\\toprule")
    latex_str.append(" & " + " & ".join([f"\\multicolumn{{{num_mtr}}}{{c}}{{{m}}}" for m in methods.keys()]) + " \\\\")
    latex_str.append(" ".join([f"\\cmidrule(lr){{{2 + i*num_mtr}-{1 + (i+1)*num_mtr}}}" for i in range(num_mth)]))
    
    metric_headers = [f"{METRIC_DISPLAY_NAME.get(m, m)} $\\downarrow$" for _ in range(num_mth) for m in csv_metrics]
    latex_str.append("Exercise & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")

    if dataset.lower() == "avafit":
        for cat in AVAFIT_CAT_ORDER:
            cat_ids = sorted([eid for eid in exercise_ids if AVAFIT_ID_TO_CAT.get(eid) == cat])
            if not cat_ids: continue
            
            # Fila de categoría gris
            latex_str.append("\\rowcolor{gray!20}")
            latex_str.append(f"\\textit{{{cat}}} " + " & " * (num_mth * num_mtr) + " \\\\")
            
            for eid in cat_ids:
                ex_name = AVAFIT_MAPPING.get(eid, eid).replace("_", "\\_")
                row = [ex_name]
                for mth in methods:
                    for mtr in csv_metrics:
                        val = data[eid][mth].get(mtr, '-')
                        row.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
                latex_str.append(" & ".join(row) + " \\\\")
    else:
        for eid in sorted(exercise_ids):
            row = [eid.replace("_", "\\_")]
            for mth in methods:
                for mtr in csv_metrics:
                    val = data[eid][mth].get(mtr, '-')
                    row.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
            latex_str.append(" & ".join(row) + " \\\\")

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f"{dataset}_exercise_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_str))
    print(f"Tabla de ejercicios guardada en {out_file}")

def main(args):
    get_tables(os.path.join(args.results_path, f"{args.dataset}_metrics.csv"), args.out_folder, args.dataset)
    get_region_metrics_table(args.results_path, args.out_folder, args.dataset)
    get_exercise_metrics_table(args.results_path, args.out_folder, args.dataset)

if __name__ == "__main__":
    main(get_args())