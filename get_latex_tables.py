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

# --- FUNCIONES ---

def get_args():
    parser = argparse.ArgumentParser(description="Get latex tables from the results")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results files")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the summary")
    return parser.parse_args()

def get_region_metrics_tables(results_folder: str, out_folder: str, dataset: str) -> None:
    methods = {"Multi-HMR": f"{dataset}_multihmr_metrics_region.csv",
               "PromptHMR": f"{dataset}_PromptHMR_metrics_region.csv",
               "3DB": f"{dataset}_SAM3DBODY_metrics_region.csv"}
    csv_metrics = ["PVE_SMPLX", "PA-PVE_SMPLX"]
    display_metrics = ["PVE", "PA-PVE"]
    data = defaultdict(lambda: defaultdict(dict))

    regions = []
    for method_name, filename in methods.items():
        file_path = os.path.join(results_folder, filename)
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        region_col = df.columns[0]
        for _, row in df.iterrows():
            region = str(row[region_col])
            if region not in regions:
                regions.append(region)
            for csv_m, disp_m in zip(csv_metrics, display_metrics):
                if csv_m in df.columns:
                    data[region][method_name][disp_m] = row[csv_m]

    if not regions:
        return

    # Find best (min) per region per metric
    best = defaultdict(dict)
    for region in regions:
        for m in display_metrics:
            vals = []
            for method_name in methods:
                val = data[region].get(method_name, {}).get(m)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    vals.append(val)
            if vals:
                best[region][m] = min(vals)

    num_mth = len(methods)
    num_mtr = len(display_metrics)

    latex_str = ["\\begin{table}[h]", "\\centering", "\\resizebox{\\textwidth}{!}{"]
    latex_str.append(f"\\begin{{tabular}}{{l{'c' * (num_mth * num_mtr)}}}")

    # Method headers
    latex_str.append(" & " + " & ".join(
        [f"\\multicolumn{{{num_mtr}}}{{c}}{{{m}}}" for m in methods.keys()]
    ) + " \\\\")

    # Cmidrules
    latex_str.append(" ".join(
        [f"\\cmidrule(lr){{{2 + i*num_mtr}-{1 + (i+1)*num_mtr}}}" for i in range(num_mth)]
    ))

    # Metric headers
    metric_headers = [f"{m} $\\downarrow$" for _ in range(num_mth) for m in display_metrics]
    latex_str.append("Region & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")

    # Data rows
    for region in regions:
        row = [region]
        for method_name in methods:
            for m in display_metrics:
                val = data[region].get(method_name, {}).get(m)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row.append("-")
                else:
                    formatted = f"{val:.1f}"
                    best_val = best[region].get(m)
                    if best_val is not None and abs(val - best_val) < 1e-6:
                        formatted = f"\\textbf{{{formatted}}}"
                    row.append(formatted)
        latex_str.append(" & ".join(row) + " \\\\")

    # Caption and label per dataset
    if dataset.lower() == "fit3d":
        caption = "Region-based metrics comparison on FIT3D."
        label = "tab:region_metrics"
    elif dataset.lower() == "avafit":
        caption = "Region-based metrics comparison on AVAFIT dataset (only one subject)."
        label = "tab:avafit_region_metrics"
    else:
        caption = f"Region-based metrics comparison on {dataset}."
        label = f"tab:{dataset}_region_metrics"

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "}",
                      f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}",
                      "\\end{table}"])

    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f"{dataset}_region_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_str))
    print(f"Region table saved to {out_file}")


def get_only_body_table(results_path: str, out_folder: str) -> None:
    datasets = {
        "FIT3D": os.path.join(results_path, "fit3d_metrics_only-body.csv"),
        "AVAFIT (only one subject)": os.path.join(results_path, "avafit_metrics_only-body.csv"),
    }
    metrics = ["MPJPE", "PA-MPJPE", "PVE", "PA-PVE"]
    metric_display = {"MPJPE": "MPJPE", "PA-MPJPE": "PA-MPJPE", "PVE": "PVE", "PA-PVE": "PA-PVE"}
    method_name_mapping = {"4DH": "HMR2.0", "multihmr": "Multi-HMR", "PromptHMR": "PromptHMR", "SAM3DBODY": "3DB-DINOv3"}
    body_only_methods = ["HMR2.0"]
    method_order = ["HMR2.0", "Multi-HMR", "PromptHMR", "3DB-DINOv3"]

    # Read data per dataset
    data = {}  # data[dataset_name][method] = {metric: value}
    all_methods = set()
    for ds_name, filepath in datasets.items():
        data[ds_name] = {}
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        df = df.rename(columns={df.columns[0]: "Method"})
        df["Method"] = df["Method"].replace(method_name_mapping)
        for _, row in df.iterrows():
            method = str(row["Method"])
            all_methods.add(method)
            data[ds_name][method] = {}
            for m in metrics:
                if m in df.columns:
                    data[ds_name][method][m] = row[m]

    if not all_methods:
        return

    # Order methods
    ordered_methods = [m for m in method_order if m in all_methods]

    # Find best (min) values per dataset per metric (across all methods)
    best = {}
    for ds_name in datasets:
        best[ds_name] = {}
        for m in metrics:
            vals = []
            for method in ordered_methods:
                val = data[ds_name].get(method, {}).get(m)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    vals.append(val)
            if vals:
                best[ds_name][m] = min(vals)

    num_metrics = len(metrics)
    num_datasets = len(datasets)
    total_cols = num_metrics * num_datasets

    latex_str = []
    latex_str.append("\\begin{table}[h]")
    latex_str.append("\\centering")
    latex_str.append("\\footnotesize")
    latex_str.append("\\setlength{\\tabcolsep}{3pt}")
    latex_str.append(f"\\begin{{tabular}}{{l*{{{total_cols}}}{{c}}}}")

    # Dataset header
    ds_headers = []
    for ds_name in datasets:
        ds_headers.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{{ds_name}}}")
    latex_str.append("& " + " & ".join(ds_headers) + " \\\\")

    # Cmidrules
    cmidrules = []
    for i, ds_name in enumerate(datasets):
        start = 2 + i * num_metrics
        end = start + num_metrics - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    latex_str.append(" ".join(cmidrules))

    # Metric header
    metric_headers = []
    for _ in datasets:
        for m in metrics:
            metric_headers.append(f"{metric_display[m]} $\\downarrow$")
    latex_str.append("Method & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")

    # Body-only section
    latex_str.append("\\rowcolor{gray!20}")
    latex_str.append("\\textit{Body-only} \\\\")

    for method in ordered_methods:
        if method in body_only_methods:
            row_vals = [method]
            for ds_name in datasets:
                for m in metrics:
                    val = data[ds_name].get(method, {}).get(m)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row_vals.append("-")
                    else:
                        formatted = f"{val:.1f}"
                        if best[ds_name].get(m) is not None and abs(val - best[ds_name][m]) < 1e-6:
                            formatted = f"\\textbf{{{formatted}}}"
                        row_vals.append(formatted)
            latex_str.append(" & ".join(row_vals) + " \\\\")

    # Whole-body section
    latex_str.append("\\rowcolor{gray!20}")
    latex_str.append("\\textit{Whole-body} \\\\")

    for method in ordered_methods:
        if method not in body_only_methods:
            row_vals = [method]
            for ds_name in datasets:
                for m in metrics:
                    val = data[ds_name].get(method, {}).get(m)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row_vals.append("-")
                    else:
                        formatted = f"{val:.1f}"
                        if best[ds_name].get(m) is not None and abs(val - best[ds_name][m]) < 1e-6:
                            formatted = f"\\textbf{{{formatted}}}"
                        row_vals.append(formatted)
            latex_str.append(" & ".join(row_vals) + " \\\\")

    latex_str.append("\\bottomrule")
    latex_str.append("\\end{tabular}")
    latex_str.append("\\caption{\\textbf{Body-only benchmark.} Results on FIT3D and AVAFIT datasets using SMPL meshes.}")
    latex_str.append("\\label{tab:fit3d_avafit_bodyonly}")
    latex_str.append("\\end{table}")

    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "body_only_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_str))
    print(f"Body-only table saved to {out_file}")


def get_whole_body_table(results_path: str, out_folder: str) -> None:
    method_name_mapping = {"multihmr": "Multi-HMR", "PromptHMR": "PromptHMR", "SAM3DBODY": "3DB-DINOv3"}
    method_order = ["Multi-HMR", "PromptHMR", "3DB-DINOv3"]

    datasets = {
        "FIT3D": "fit3d",
        "AVAFIT (only one subject)": "avafit",
    }

    # CSV file suffixes and corresponding column names per body part
    parts = {
        "All":   ("whole-body", "PVE_SMPLX",  "PA-PVE_SMPLX"),
        "Hands": ("hands",      "PVE_HANDS",  "PA-PVE_HANDS"),
        "Feet":  ("feet",       "PVE_FEETS",  "PA-PVE_FEETS"),
    }
    metrics = ["PVE", "PA-PVE"]
    part_names = ["All", "Hands", "Feet"]

    # data[ds_label][method] = {"PVE": {"All": v, "Hands": v, "Feet": v}, "PA-PVE": {...}}
    data = {ds: {} for ds in datasets}

    for ds_label, ds_prefix in datasets.items():
        for part_name, (suffix, pve_col, pa_pve_col) in parts.items():
            filepath = os.path.join(results_path, f"{ds_prefix}_metrics_{suffix}.csv")
            if not os.path.exists(filepath):
                continue
            df = pd.read_csv(filepath)
            df = df.rename(columns={df.columns[0]: "Method"})
            df["Method"] = df["Method"].replace(method_name_mapping)
            for _, row in df.iterrows():
                method = str(row["Method"])
                if method not in data[ds_label]:
                    data[ds_label][method] = {m: {} for m in metrics}
                if pve_col in df.columns:
                    data[ds_label][method]["PVE"][part_name] = row[pve_col]
                if pa_pve_col in df.columns:
                    data[ds_label][method]["PA-PVE"][part_name] = row[pa_pve_col]

    # Find best (min) per dataset, metric, part
    best = {ds: {m: {} for m in metrics} for ds in datasets}
    for ds_label in datasets:
        for m in metrics:
            for p in part_names:
                vals = []
                for method in method_order:
                    val = data[ds_label].get(method, {}).get(m, {}).get(p)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        vals.append(val)
                if vals:
                    best[ds_label][m][p] = min(vals)

    # Build LaTeX
    # Total columns: 2 datasets × 2 metrics × 3 parts = 12
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\setlength{\\tabcolsep}{3pt}")
    latex.append("\\begin{tabular}{l*{12}{c}}")

    # Row 1: dataset headers (6 cols each)
    ds_headers = " & ".join([f"\\multicolumn{{6}}{{c}}{{{ds}}}" for ds in datasets])
    latex.append(f"& {ds_headers} \\\\")

    # Cmidrules for datasets
    latex.append("\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}")

    # Row 2: metric headers (3 cols each, 4 groups)
    metric_headers = []
    for _ in datasets:
        for m in metrics:
            metric_headers.append(f"\\multicolumn{{3}}{{c}}{{{m} $\\downarrow$}}")
    latex.append("& " + " & ".join(metric_headers) + " \\\\")

    # Cmidrules for metrics
    latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")

    # Row 3: part headers
    part_headers = " & ".join(part_names * (len(datasets) * len(metrics)))
    latex.append(f"Method & {part_headers} \\\\")
    latex.append("\\midrule")

    # Data rows
    for method in method_order:
        row_vals = [method]
        for ds_label in datasets:
            for m in metrics:
                for p in part_names:
                    val = data[ds_label].get(method, {}).get(m, {}).get(p)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row_vals.append("-")
                    else:
                        formatted = f"{val:.1f}"
                        best_val = best[ds_label][m].get(p)
                        if best_val is not None and abs(val - best_val) < 1e-6:
                            formatted = f"\\textbf{{{formatted}}}"
                        row_vals.append(formatted)
        latex.append(" & ".join(row_vals) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{\\textbf{Whole-body benchmark.} Results on FIT3D and AVAFIT datasets using SMPLX meshes.}")
    latex.append("\\label{tab:fit3d_avafit_wholebody}")
    latex.append("\\end{table}")

    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "whole_body_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex))
    print(f"Whole-body table saved to {out_file}")


def get_exercise_avafit_metrics_table(results_path: str, out_folder: str) -> None:
    methods = {"Multi-HMR": "avafit_multihmr_metrics_exercise.csv",
               "PromptHMR": "avafit_PromptHMR_metrics_exercise.csv",
               "3DB": "avafit_SAM3DBODY_metrics_exercise.csv"}
    csv_metrics = ["PVE_SMPLX", "PA-PVE_SMPLX"]
    display_metrics = ["PVE", "PA-PVE"]
    data = defaultdict(lambda: defaultdict(dict))

    # Initialise with all exercise IDs
    exercise_ids = list(AVAFIT_ID_TO_CAT.keys())

    for method_name, filename in methods.items():
        file_path = os.path.join(results_path, filename)
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        ex_col = df.columns[0]
        for _, row in df.iterrows():
            ex_id = str(row[ex_col])
            for csv_m, disp_m in zip(csv_metrics, display_metrics):
                if csv_m in df.columns:
                    data[ex_id][method_name][disp_m] = row[csv_m]

    if not exercise_ids:
        return

    # Find best (min) per exercise per metric
    best = defaultdict(dict)
    for ex_id in exercise_ids:
        for m in display_metrics:
            vals = []
            for method_name in methods:
                val = data[ex_id].get(method_name, {}).get(m)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    vals.append(val)
            if vals:
                best[ex_id][m] = min(vals)

    num_mth = len(methods)
    num_mtr = len(display_metrics)

    latex_str = ["\\begin{table}[h]", "\\centering", "\\resizebox{\\textwidth}{!}{"]
    latex_str.append(f"\\begin{{tabular}}{{l{'c' * (num_mth * num_mtr)}}}")

    # Method headers
    latex_str.append(" & " + " & ".join(
        [f"\\multicolumn{{{num_mtr}}}{{c}}{{{m}}}" for m in methods.keys()]
    ) + " \\\\")

    # Cmidrules
    latex_str.append(" ".join(
        [f"\\cmidrule(lr){{{2 + i*num_mtr}-{1 + (i+1)*num_mtr}}}" for i in range(num_mth)]
    ))

    # Metric headers
    metric_headers = [f"{m} $\\downarrow$" for _ in range(num_mth) for m in display_metrics]
    latex_str.append("Exercise & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")

    # Data rows grouped by category
    for cat in AVAFIT_CAT_ORDER:
        cat_ids = sorted([eid for eid in exercise_ids if AVAFIT_ID_TO_CAT.get(eid) == cat])
        if not cat_ids:
            continue

        # Category header row (gray)
        latex_str.append("\\rowcolor{gray!20}")
        latex_str.append(f"\\textit{{{cat}}} " + " & " * (num_mth * num_mtr) + " \\\\")

        for eid in cat_ids:
            ex_name = AVAFIT_MAPPING.get(eid, eid)
            row = [ex_name]
            for method_name in methods:
                for m in display_metrics:
                    val = data[eid].get(method_name, {}).get(m)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row.append("-")
                    else:
                        formatted = f"{val:.1f}"
                        best_val = best[eid].get(m)
                        if best_val is not None and abs(val - best_val) < 1e-6:
                            formatted = f"\\textbf{{{formatted}}}"
                        row.append(formatted)
            latex_str.append(" & ".join(row) + " \\\\")

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "}",
                      "\\caption{Exercise-based metrics comparison on AVAFIT dataset (only one subject).}",
                      "\\label{tab:exercise_metrics_avafit}",
                      "\\end{table}"])

    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "avafit_exercise_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_str))
    print(f"AVAFIT exercise table saved to {out_file}")


def get_exercise_fit3d_metrics_table(results_path: str, out_folder: str) -> None:
    methods = {"Multi-HMR": "fit3d_multihmr_metrics_exercise.csv",
               "PromptHMR": "fit3d_PromptHMR_metrics_exercise.csv",
               "3DB": "fit3d_SAM3DBODY_metrics_exercise.csv"}
    csv_metrics = ["PVE_SMPLX", "PA-PVE_SMPLX"]
    display_metrics = ["PVE", "PA-PVE"]
    data = defaultdict(lambda: defaultdict(dict))

    exercises = []
    for method_name, filename in methods.items():
        file_path = os.path.join(results_path, filename)
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        ex_col = df.columns[0]
        for _, row in df.iterrows():
            ex_id = str(row[ex_col])
            if ex_id not in exercises:
                exercises.append(ex_id)
            for csv_m, disp_m in zip(csv_metrics, display_metrics):
                if csv_m in df.columns:
                    data[ex_id][method_name][disp_m] = row[csv_m]

    if not exercises:
        return

    exercises.sort()

    # Find best (min) per exercise per metric
    best = defaultdict(dict)
    for ex_id in exercises:
        for m in display_metrics:
            vals = []
            for method_name in methods:
                val = data[ex_id].get(method_name, {}).get(m)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    vals.append(val)
            if vals:
                best[ex_id][m] = min(vals)

    num_mth = len(methods)
    num_mtr = len(display_metrics)

    latex_str = ["\\begin{table}[h]", "\\centering", "\\resizebox{\\textwidth}{!}{"]
    latex_str.append(f"\\begin{{tabular}}{{l{'c' * (num_mth * num_mtr)}}}")

    # Method headers
    latex_str.append(" & " + " & ".join(
        [f"\\multicolumn{{{num_mtr}}}{{c}}{{{m}}}" for m in methods.keys()]
    ) + " \\\\")

    # Cmidrules
    latex_str.append(" ".join(
        [f"\\cmidrule(lr){{{2 + i*num_mtr}-{1 + (i+1)*num_mtr}}}" for i in range(num_mth)]
    ))

    # Metric headers
    metric_headers = [f"{m} $\\downarrow$" for _ in range(num_mth) for m in display_metrics]
    latex_str.append("Exercise & " + " & ".join(metric_headers) + " \\\\")
    latex_str.append("\\midrule")

    # Data rows
    for ex_id in exercises:
        ex_name = ex_id.replace("_", "\\_")
        row = [ex_name]
        for method_name in methods:
            for m in display_metrics:
                val = data[ex_id].get(method_name, {}).get(m)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row.append("-")
                else:
                    formatted = f"{val:.1f}"
                    best_val = best[ex_id].get(m)
                    if best_val is not None and abs(val - best_val) < 1e-6:
                        formatted = f"\\textbf{{{formatted}}}"
                    row.append(formatted)
        latex_str.append(" & ".join(row) + " \\\\")

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "}",
                      "\\caption{Exercise-based metrics comparison on FIT3D dataset.}",
                      "\\label{tab:exercise_metrics}",
                      "\\end{table}"])

    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "fit3d_exercise_table.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_str))
    print(f"FIT3D exercise table saved to {out_file}")




def main(args):
    get_only_body_table(args.results_path, args.out_folder)
    get_whole_body_table(args.results_path, args.out_folder)
    get_region_metrics_tables(args.results_path, args.out_folder, "fit3d")
    get_region_metrics_tables(args.results_path, args.out_folder, "avafit")
    get_exercise_avafit_metrics_table(args.results_path, args.out_folder)
    get_exercise_fit3d_metrics_table(args.results_path, args.out_folder)

if __name__ == "__main__":
    main(get_args())