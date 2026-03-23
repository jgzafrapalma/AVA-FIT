import os
import argparse
import numpy as np
import pathlib
import pandas as pd
import umap
import base64
from tqdm import tqdm
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

def get_args():
    parser = argparse.ArgumentParser(description="Visualize pose descriptors from 3D human mesh estimation models")
    parser.add_argument("--descriptors_path",       type=str, required=True,  help="Path to the descriptors")
    parser.add_argument("--output_path",            type=str, required=True,  help="Output path")
    parser.add_argument("--frames_descriptors_path",type=str, required=True,  help="Path to the frames")
    parser.add_argument("--num_samples",            type=int, default=None,
                        help="Number of samples to visualize (stratified by label). "
                             "If not set, all samples are used.")
    return parser.parse_args()

def get_descriptor(descriptors_path, output_path, frames_descriptors_path):
    labels = []
    descriptors = []
    frames_path = []

    for dataset in os.listdir(descriptors_path):
        for file_descriptor in tqdm(os.listdir(os.path.join(descriptors_path, dataset))):
            descriptor = np.load(os.path.join(descriptors_path, dataset, file_descriptor))['descriptors']
            descriptors.append(descriptor)

            frames_dir = os.path.join(frames_descriptors_path, pathlib.Path(file_descriptor).stem)
            frames_path.extend(sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)]))

            label = "fit3d" if dataset == "fit3d" else "avafit"
            labels.extend([label] * descriptor.shape[0])

    descriptors = np.vstack(descriptors)
    labels      = np.array(labels)
    frames_path = np.array(frames_path)

    np.savez(os.path.join(output_path, 'descriptors.npz'),
             descriptors=descriptors, labels=labels, frames_path=frames_path)

def stratified_sample(descriptors, labels, frames_path, num_samples):
    """
    Devuelve un subconjunto de tamaño num_samples manteniendo
    la proporción original de cada label.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    selected = []

    for lbl, count in zip(unique_labels, counts):
        # Fracción proporcional de num_samples para este label
        n = max(1, round(num_samples * count / total))
        idx = np.where(labels == lbl)[0]
        n = min(n, len(idx))                         # no pedir más de los disponibles
        chosen = np.random.choice(idx, size=n, replace=False)
        selected.append(chosen)

    selected_idx = np.concatenate(selected)
    np.random.shuffle(selected_idx)                  # mezclar para que no queden agrupados

    return descriptors[selected_idx], labels[selected_idx], frames_path[selected_idx]

def encode_image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def plot_descriptors(output_path, num_samples=None):
    # Load descriptors
    data        = np.load(os.path.join(output_path, 'descriptors.npz'), allow_pickle=True)
    descriptors = data['descriptors']
    labels      = data['labels']
    frames_path = data['frames_path']

    # --- Subsampling ---
    if num_samples is not None:
        if num_samples >= len(labels):
            print(f"[INFO] num_samples ({num_samples}) >= total samples ({len(labels)}). Using all samples.")
        else:
            print(f"[INFO] Sampling {num_samples} / {len(labels)} samples (stratified by label)...")
            descriptors, labels, frames_path = stratified_sample(
                descriptors, labels, frames_path, num_samples
            )
            print(f"[INFO] Samples per label: "
                  f"{ {l: int((labels==l).sum()) for l in np.unique(labels)} }")

    # Normalize
    descriptors = (descriptors - descriptors.mean(axis=0)) / descriptors.std(axis=0)

    # UMAP reduction
    print("Computing UMAP...")
    embedding    = umap.UMAP(n_components=2).fit_transform(descriptors)
    embedding_df = pd.DataFrame(embedding, columns=['umap_1', 'umap_2'])
    embedding_df['label'] = labels

    # Encode frames
    print("Encoding frames to base64...")
    embedding_df['img'] = [encode_image_to_base64(p) for p in tqdm(frames_path)]

    # Bokeh plot
    datasource    = ColumnDataSource(embedding_df)
    unique_labels = sorted(embedding_df['label'].unique().tolist())
    color_mapper  = CategoricalColorMapper(
        factors=unique_labels,
        palette=Spectral10[:len(unique_labels)]
    )

    n_shown  = len(labels)
    subtitle = f"{n_shown} samples" if num_samples else f"all {n_shown} samples"
    plot_figure = figure(
        title=f'UMAP projection — pose descriptors ({subtitle})',
        width=800,
        height=800,
        tools='pan, wheel_zoom, reset, save'
    )

    plot_figure.add_tools(HoverTool(tooltips="""
        <div style="margin: 5px;">
            <img src='data:image/jpeg;base64, @img'
                 style='max-width: 120px; max-height: 120px; float: left; margin: 5px;'/>
            <div style="margin-top: 5px;">
                <span style="font-weight: bold;">@label</span>
            </div>
        </div>
    """))

    plot_figure.circle(
        x='umap_1', y='umap_2',
        source=datasource,
        color={'field': 'label', 'transform': color_mapper},
        legend_field='label',
        line_alpha=0.6, fill_alpha=0.6, size=5
    )

    plot_figure.legend.location    = 'top_left'
    plot_figure.legend.click_policy = 'hide'

    # Nombre del fichero refleja si es subconjunto o no
    suffix      = f"_n{n_shown}" if num_samples else ""
    output_html = os.path.join(output_path, f'descriptors_umap{suffix}.html')
    output_file(filename=output_html, title="UMAP — Pose Descriptors")
    save(plot_figure)
    print(f"HTML saved to: {output_html}")

def main(args):
    get_descriptor(args.descriptors_path, args.output_path, args.frames_descriptors_path)
    plot_descriptors(args.output_path, num_samples=args.num_samples)

if __name__ == "__main__":
    args = get_args()
    main(args)