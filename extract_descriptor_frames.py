import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import pathlib
import re
import sys

viewpoint_to_camera_id = {
    'view_front_left': '60457274',
    'view_front_right': '65906101',
    'view_back_left': '50591643',
    'view_back_right': '58860488',
}

def get_args():
    parser = argparse.ArgumentParser(description="Extraer fotogramas de videos usando los índices guardados en los descriptores.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directorio raíz que contiene los videos (se buscará en subdirectorios)")
    parser.add_argument("--descriptors_dir", type=str, required=True, help="Directorio que contiene los archivos de descriptores (.npz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio donde se guardarán los fotogramas extraídos")
    parser.add_argument("--resolution", type=int, default=360, help="Resolución (altura en píxeles) a la que redimensionar, p. ej. 360 o 480. Por defecto 360.")
    return parser.parse_args()


def find_matching_video(descriptor_path, dataset_path):

    descriptor_name = pathlib.Path(descriptor_path).name
    
    dataset = pathlib.Path(descriptor_path).parent.name

    video_path = None
    
    if dataset == "fit3d":
        pattern = r"^(?P<participant>s\d+)_(?P<exercise>.+)_view_(?P<view>.+)_descriptors\.npz$"
        match = re.match(pattern, descriptor_name)
        if match:
            data = match.groupdict()
            participant = data["participant"]
            exercise = data["exercise"]
            viewpoint = f"view_{data['view']}"

            video_path = os.path.join(dataset_path, "fit3d", 'train', participant, 'videos', viewpoint_to_camera_id[viewpoint], f'{exercise}.mp4')
    
    else:
        
        descriptor_name_split = descriptor_name.split('_')
        participant = descriptor_name_split[0] + '_' + descriptor_name_split[1]
        exercise = descriptor_name_split[2]
        viewpoint = descriptor_name_split[3]
        repetition = descriptor_name_split[4]
        
        
        pattern = rf"seq_{participant}_{exercise}_{repetition}_{viewpoint}_.*\.mp4$"

        for video in os.listdir(os.path.join(dataset_path, dataset, "videos")):
            match = re.match(pattern, video)
            if match:
                video_path = os.path.join(dataset_path, dataset, "videos", video)
                break
        
        if not video_path:
            print(f"[Aviso] No se encontró el video para el descriptor {descriptor_name}")
            video_path = None
            sys.exit(1)
        
    return video_path

def main():
    args = get_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: El directorio de videos no existe: {args.dataset_path}")
        return
        
    if not os.path.exists(args.descriptors_dir):
        print(f"Error: El directorio de descriptores no existe: {args.descriptors_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Obtener todas las rutas de los descriptores
    descriptor_paths = []
    for root, _, files in os.walk(args.descriptors_dir):
        for f in files:
            if f.endswith('.npz'):
                descriptor_paths.append(os.path.join(root, f))
                
    print(f"Se encontraron {len(descriptor_paths)} archivos de descriptores (.npz).")
    
    for desc_path in tqdm(descriptor_paths, desc="Procesando descriptores"):
        desc_filename = os.path.basename(desc_path)
        
        try:
            data = np.load(desc_path)
            if 'indexes' in data:
                steps_interval = data['indexes']
            else:
                print(f"\n[Advertencia] No se encontró el arreglo de índices en {desc_filename}. Claves: {data.files}")
                continue
        except Exception as e:
            print(f"\n[Error] al cargar {desc_filename}: {e}")
            continue
            
        video_path = find_matching_video(desc_path, args.dataset_path)
        
        if video_path is None:
            print(f"\n[Aviso] Video no encontrado para el descriptor {desc_filename}.")
            continue
            
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"\n[Error] No se pudo abrir el video: {video_path}")
            continue
            

        video_out_dir = os.path.join(args.output_dir, pathlib.Path(desc_filename).stem)
        os.makedirs(video_out_dir, exist_ok=True)
        
        # Iterar sobre los frames que queremos guardar (ordenados para ser eficientes)
        steps_interval = sorted(list(set(steps_interval))) # Aseguramos unicidad y orden
        
        for frame_idx in steps_interval:
            # OpenCV usa índices basados en 0, coincidente con np.arange()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                # Solo redimensionar si la resolución original es mayor
                if h != args.resolution:
                    new_h = args.resolution
                    new_w = int((new_h / h) * w)
                    frame = cv2.resize(frame, (new_w, new_h))

                frame_filename = os.path.join(video_out_dir, f"frame_{int(frame_idx):06d}.jpg")
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"\n[Aviso] No se pudo leer el fotograma {frame_idx} del video {video_filename}")
                
        cap.release()

if __name__ == "__main__":
    main()
