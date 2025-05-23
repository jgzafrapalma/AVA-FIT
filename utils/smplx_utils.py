import trimesh
import pyrender
import numpy as np

import tensorflow as tf

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

mesh_rasterizer = pyrender.OffscreenRenderer(viewport_width=900, 
                                          viewport_height=900, 
                                          point_size=1.0)

def render(vertices, frame, cam_params, faces):
    
    blending_weight=1.0
    
    vertices_to_render = vertices
    intrinsics = cam_params['intrinsics_wo_distortion']['f'].tolist() + cam_params['intrinsics_wo_distortion']['c'].tolist()
    background_image = frame 

    vertex_colors = np.ones([vertices_to_render.shape[0], 4]) * [0.3, 0.3, 0.3, 1]
    tri_mesh = trimesh.Trimesh(vertices_to_render, faces,
                                vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    rot = trimesh.transformations.euler_matrix(0, np.pi, np.pi, 'rxyz')
    camera_pose[:3, :3] = rot[:3, :3]

    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics[0],
        fy=intrinsics[1],
        cx=intrinsics[2],
        cy=intrinsics[3])

    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)

    scene.add(light, pose=camera_pose)
    color, rend_depth = mesh_rasterizer.render(scene, flags=pyrender.RenderFlags.RGBA)
    img = color.astype(np.float32) / 255.0

    blended_image = img[:, :, :3]

    if background_image is not None:
        background_image = tf.convert_to_tensor(background_image,
                                                tf.float32) / 255.

        # Rendering results needs to be rescaled to blend with background image.
        img = tf.image.resize(
            img, (background_image.shape[0], background_image.shape[1]),
            antialias=True)
        # Blend the rendering result with the background image.
        foreground = (rend_depth > 0)[:, :, None] * blending_weight
        foreground = tf.image.resize(
            foreground, (background_image.shape[0], background_image.shape[1]),
            antialias=True)
        blended_image = (foreground * img[:, :, :3]
                            + (1. - foreground) * background_image)
        
    return blended_image