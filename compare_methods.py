import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(output_dir, triplets):
    os.makedirs(output_dir, exist_ok=True)

    dataframes = []
    method_names = []
    colors = []

    for i in range(0, len(triplets), 3):
        file = triplets[i]
        name = triplets[i + 1]
        color = triplets[i + 2]

        df = pd.read_csv(file, index_col=0).sort_index()
        dataframes.append(df)
        method_names.append(name)
        colors.append(color)

    # Índices comunes
    common_indices = dataframes[0].index
    for df in dataframes[1:]:
        common_indices = common_indices.intersection(df.index)
    dataframes = [df.loc[common_indices] for df in dataframes]

    # Métricas comunes
    common_metrics = dataframes[0].columns
    for df in dataframes[1:]:
        common_metrics = common_metrics.intersection(df.columns)
    dataframes = [df[common_metrics] for df in dataframes]

    # Graficar
    for metric in common_metrics:
        plt.figure(figsize=(12, 6))
        max_val = max(df[metric].max() for df in dataframes)
        y_max = (int(max_val / 20) + 1) * 20
        x = range(len(common_indices))
        width = 0.8 / len(dataframes)

        for i, df in enumerate(dataframes):
            offset = i * width
            plt.bar([xi + offset for xi in x], df[metric], width=width,
                    label=method_names[i], color=colors[i])

        plt.xticks([i + 0.4 for i in x], common_indices, rotation=45, ha='right')
        plt.xlabel('Exercise')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric}')
        plt.ylim(0, y_max)
        plt.yticks(range(0, y_max + 1, 20))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(output_path)
        plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 5 or (len(sys.argv) - 2) % 3 != 0:
        print("Uso: python compare_methods.py <output_dir> <csv1> <nombre1> <color1> <csv2> <nombre2> <color2> ...")
        sys.exit(1)

    output_dir = sys.argv[1]
    triplets = sys.argv[2:]
    main(output_dir, triplets)
