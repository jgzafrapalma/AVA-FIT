import os
import cv2
import sys
import torch
import pathlib
import trimesh
import argparse
import numpy as np
from smplx import SMPL, SMPLX
from utils.constants import SMPL_PATH, SMPLX_PATH

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Install with: pip install pyvista")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Visualize 3D meshes from SMPL/SMPLX predictions")
    parser.add_argument("--pred_file", type=str, required=True, help="Predictions file")
    parser.add_argument("--gt_file", type=str, required=False, help="File with the SMPLX/SMPL GT vertices")
    parser.add_argument("--video_file", type=str, required=True, help="Video file to visualize")
    parser.add_argument("--frame_id", type=int, required=True, help="Frame id to visualize")
    parser.add_argument("--error", type=float, required=False, default=0.0, help="Error to visualize")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder to save the visualization")
    parser.add_argument("--cam_file", type=str, required=False, help="Camera parameters file (optional)")
    parser.add_argument("--use_pyvista", action='store_true', help="Use PyVista for visualization")
    return parser.parse_args()

def trimesh_to_pyvista(mesh):
    """
    Convierte un mesh de trimesh a PyVista.
    """
    faces = np.concatenate([np.full((len(mesh.faces), 1), 3), mesh.faces], axis=1)
    pv_mesh = pv.PolyData(mesh.vertices, faces.ravel())
    
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        pv_mesh['colors'] = colors
    
    return pv_mesh

def create_camera_actor(scale=0.5):
    """
    Crea un actor de cámara para PyVista.
    """
    # Crear líneas para el frustum
    points = np.array([
        [0, 0, 0],  # Centro
        [-scale, -scale, scale * 2],
        [scale, -scale, scale * 2],
        [scale, scale, scale * 2],
        [-scale, scale, scale * 2]
    ])
    
    lines = np.array([
        [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4],
        [2, 1, 2], [2, 2, 3], [2, 3, 4], [2, 4, 1]
    ])
    
    camera_frustum = pv.PolyData(points, lines=lines.ravel())
    return camera_frustum

def visualize_with_pyvista(mesh_gt, mesh_pred, frame, cam_params=None, error=0.0, frame_id=0):
    """
    Visualización interactiva usando PyVista desde la vista de cámara.
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista is not available. Please install it: pip install pyvista")
        return
    
    # Crear plotter
    plotter = pv.Plotter(window_size=[1280, 720], 
                         lighting='three lights',
                         title='Interactive SMPL/SMPLX Viewer - Camera View')
    
    # Convertir mallas a PyVista
    pv_gt = trimesh_to_pyvista(mesh_gt)
    pv_pred = trimesh_to_pyvista(mesh_pred)
    
    # Añadir mallas
    plotter.add_mesh(pv_gt, color='green', opacity=0.5, 
                     label='Ground Truth', smooth_shading=True)
    plotter.add_mesh(pv_pred, color='red', opacity=0.5, 
                     label='Prediction', smooth_shading=True)
    
    # Configurar el plano de imagen (Image Plane)
    # Suponemos coordenadas OpenCV: Z positivo hacia adelante, Y abajo, X derecha.
    # Colocamos el plano a una distancia arbitraria frente a la cámara.
    # Para que coincida con la vista, necesitamos saber el FOV o calibración,
    # pero haremos una aproximación visual o usaremos una distancia fija pequeña.
    
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    
    # Distancia focal aproximada si no hay cam_params
    # Un valor típico de fhol para cámaras 1080p es ~1000-1500 px, 
    # pero aquí trabajamos en espacio métrico.
    # Si las mallas están en metros, la distancia Z suele ser 2-5 metros.
    # Pondremos el plano de imagen "detrás" de las mallas si queremos verlo de fondo,
    # o mejor, posicionamos la cámara en 0,0,0 y el plano LEJOS, 
    # pero PyVista permite poner una imagen de fondo fija (background image).
    # Sin embargo, el usuario pidió "en la posición de la cámara aparezca el fotograma".
    # Esto suena a un "Billboard" en el origen o el plano de proyección.
    
    # Opción A: Usar add_background_image (fijo en ventana)
    # Opción B: Plano 3D texturizado.
    # El usuario dijo: "en la posición de la cámara aparezca el fotograma".
    # Si navegamos, queremos ver dónde estaba la cámara original.
    
    # Vamos a crear un plano pequeño en el origen (0,0,0) que represente la cámara/frame.
    plane_height = 0.5  # tamaño arbitrario en metros
    plane_width = plane_height * aspect_ratio
    
    # El plano debe estar orientado mirando hacia +Z (hacia la escena)
    # Center en 0,0,0
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), 
                     i_size=plane_width, j_size=plane_height)
    
    # Textura
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Voltear verticalmente para ajustar a las coordenadas de textura de VTK (origen abajo-izquierda)
    frame_rgb = cv2.flip(frame_rgb, 0)
    
    texture = pv.Texture(frame_rgb)
    plotter.add_mesh(plane, texture=texture, opacity=0.9, label='Camera Frame')
    
    # Añadir ejes en el origen
    plotter.add_axes()
    plotter.show_grid()

    # Información
    text = f"Frame: {frame_id}\nError: {error:.4f}m\nGreen: GT | Red: Pred"
    plotter.add_text(text, position='upper_left', font_size=10)
    
    # Configurar la cámara de PyVista para coincidir inicialmente con la cámara virutal (0,0,0)
    # Posición (0,0,0), Focal Point (0,0,1) (hacia adelante), View Up (0,-1,0) (Y abajo en OpenCV)
    # Pero si ponemos la cámara EN el plano, no lo veremos.
    # La pondremos un poco atrás para ver el plano y las mallas.
    plotter.camera.position = (0, 0, -1)
    plotter.camera.focal_point = (0, 0, 2)
    plotter.camera.up = (0, -1, 0) # OpenCV convention Y down
    
    plotter.show()

def create_trimesh(vertices, faces, color):
    if len(color) == 3:
        color = color + [255]
    vertex_colors = np.tile(np.array(color, dtype=np.uint8), (vertices.shape[0], 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    return mesh

def visualize_meshes(pred_file: str, frame_id: int, out_folder: str, gt_file: str, 
                     error: float, frame, cam_file: str = None, use_pyvista: bool = False) -> None:
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    
    # Cargar datos GT
    gt = np.load(gt_file)
    v3d = gt['v3d']
    pelvis = gt['transl_pelvis']
    
    # Cargar predicciones
    preds = np.load(pred_file)
    v3d_hat = preds['v3d']
    pelvis_hat = preds['transl_pelvis']
    img_ids = np.array([int(p) for p in preds['img_path']]) - 1
    
    pred_position = np.where(img_ids == frame_id)[0][0]
    
    v3d = v3d[frame_id]
    pelvis = pelvis[frame_id]
    v3d_hat = v3d_hat[pred_position]
    pelvis_hat = pelvis_hat[pred_position]
    
    # Preparar mallas
    smplx_model = SMPLX(SMPLX_PATH, create_transl=True, use_pca=False, 
                        num_betas=10, ext='pkl', gender='male').to(device)
    
    if hasattr(preds, 'files') and 'betas' in preds:
        # Si hay betas en la predicción, usarlas podría ser mejor, 
        # pero mantenemos la lógica original de neutral/template si no se especifica.
        pass

    # Determinar modelo para predicción
    if v3d_hat.shape[0] == 6890:
        pred_neutral_model = SMPL(SMPL_PATH).to(device)
    elif v3d_hat.shape[0] == 10475:
        pred_neutral_model = SMPLX(SMPLX_PATH).to(device)
    
    # Lógica de coordenadas
    if use_pyvista:
        # Usar coordenadas ABSOLUTAS (Cámara)
        # No restamos pelvis
        mesh_gt = create_trimesh(v3d, smplx_model.faces, color=[0, 255, 0])
        mesh_pred = create_trimesh(v3d_hat, pred_neutral_model.faces, color=[255, 0, 0])
        
        print("\n" + "="*60)
        print("LAUNCHING PYVISTA INTERACTIVE VIEWER (CAMERA VIEW)")
        print("="*60)
        visualize_with_pyvista(mesh_gt, mesh_pred, frame, None, error, frame_id)
        
    else:
        # Comportamiento original: Centrado en pelvis para trimesh scene estática/interactiva simple
        v3d_ctx = v3d - pelvis
        v3d_hat_ctx = v3d_hat - pelvis_hat
        
        mesh_gt = create_trimesh(v3d_ctx, smplx_model.faces, color=[0, 255, 0])
        mesh_pred = create_trimesh(v3d_hat_ctx, pred_neutral_model.faces, color=[255, 0, 0])
        
        # Guardar archivos estáticos
        cv2.imwrite(os.path.join(out_folder, f"{error:.2f}_{frame_id:06d}.png"), frame)
        
        scene = trimesh.Scene()
        scene.add_geometry(mesh_gt, node_name="ground_truth")
        scene.add_geometry(mesh_pred, node_name="prediction")
        scene.export(os.path.join(out_folder, f"{error:.2f}_{frame_id:06d}.glb"), file_type="glb")

def main(args):
    # Cargar el video
    video = cv2.VideoCapture(args.video_file)
    if not video.isOpened():
        print(f"Error opening video file: {args.video_file}")
        sys.exit(1)
    
    # Obtener el frame específico
    video.set(cv2.CAP_PROP_POS_FRAMES, args.frame_id)
    ret, frame = video.read()
    if not ret:
        print(f"Error reading frame {args.frame_id} from video file: {args.video_file}")
        sys.exit(1)
    
    # Visualización
    visualize_meshes(
        args.pred_file, 
        args.frame_id, 
        args.out_folder, 
        args.gt_file, 
        args.error, 
        frame,
        args.cam_file,
        args.use_pyvista
    )
    
    video.release()

if __name__ == "__main__":
    args = get_args()
    main(args)