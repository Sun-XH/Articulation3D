import random
from black import T
import cv2
from cv2 import transform
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os, re, glob, natsort, json

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
import imageio
import pdb
from PIL import Image

import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    BlendParams,
    SoftSilhouetteShader,
    TexturesVertex
)
from pytorch3d.io import load_obj, save_obj

def visualize_optimal_poses_humans_video(meshes, images, vis_renderer, right_renderer, frame_ids, output_path):

    # writer = imageio.get_writer(os.path.join(
    #         output_path, '{}.mp4'.format('output')), fps=10)
    frames = []
    for idx in range(images.shape[0]):
        # pdb.set_trace()
        if idx in frame_ids:
            R, T = look_at_view_transform(0.1, 0.0, 0.0,device='cuda')
            T[0,2] = 0.0  # manually set to zero
            MESH = meshes[frame_ids.index(idx)]

            projection = vis_renderer(MESH, R=R, T=T)
            # left - right viewpoint
            bbox = MESH.get_bounding_boxes()
            _at = bbox[0].mean(dim=1)
            R, T = look_at_view_transform(_at[-1], 0, 90, at=_at[None], device='cuda')
            left_proj = right_renderer(MESH, R=R, T=T)

            R, T = look_at_view_transform(_at[-1], 0, 270, at=_at[None], device='cuda')
            right_proj = right_renderer(MESH, R=R, T=T)
            
            proj_frame = projection[0,...,:3].detach().cpu().numpy()
            
            H, W, _ = images[idx].shape
            if H > W:
                diff = (H - W) // 2
                proj_frame = proj_frame[:, diff:-diff]
            else:
                diff = (W - H) // 2
                proj_frame = proj_frame[diff:-diff, :]

            left_frame = left_proj[0, ..., :3].detach().cpu().numpy()
            right_frame = right_proj[0, ..., :3].detach().cpu().numpy()

            fig = plt.figure(figsize=(10, 10))
            canvas = FigureCanvasAgg(fig)

            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(images[idx])
            # pdb.set_trace()
            ax.imshow(proj_frame, alpha=0.6)
            ax.set_title("Predicted Overlayed")
            # ax.axis('off')

            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(left_frame)
            ax.set_title("Predicted Left View Point")
            ax.axis('off')

            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(right_frame)
            ax.set_title("Predicted Right View Point")
            ax.axis('off')

            # pdb.set_trace()
            canvas.draw()
            buf = canvas.buffer_rgba()
            im = np.asarray(buf)

            # pdb.set_trace()
            image = Image.fromarray(im)
            frames.append(image)

    # pdb.set_trace()
    # if len(frames) >= 2:
    #     try:
    #         os.makedirs(output_path[:-10], exist_ok=True)
    #         os.makedirs(output_path, exist_ok=True)
    #     except:
    #         os.makedirs(output_path, exist_ok=True)
    #     output_gif_path = f'{output_path}final_result.gif'
    #     frames[0].save(output_gif_path, format = 'GIF', append_images = frames[1:], save_all = True, duration = 300, loop = 0)

    if len(frames) >= 2:
        os.makedirs(output_path, exist_ok=True)
        output_gif_path = f'{output_path}/final_result.gif'
        frames[0].save(output_gif_path, format = 'GIF', append_images = frames[1:], save_all = True, duration = 300, loop = 0)



            # plt.imsave(f'test_{idx}.png', im)



def initialize_render(device, focal_x, focal_y, img_square_size, img_small_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)
    shrink_ratio = int(img_square_size/img_small_size)
    focal_x_small = int(focal_x/shrink_ratio)
    focal_y_small = int(focal_y/shrink_ratio)
    img_small_center = int(img_small_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_x, focal_y),),
                # principal_point=((img_square_center, img_square_center),),
                # image_size = ((img_square_size, img_square_size),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                in_ndc=False,
                device=device)

    # camera_sfm_small = PerspectiveCameras(
    #             focal_length=((focal_x_small, focal_y_small),),
    #             principal_point=((img_small_center, img_small_center),),
    #             image_size = ((img_small_size, img_small_size),),
    #             in_ndc=False,
    #             device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # # Define the settings for rasterization and shading. Here we set the output image to be of size
    # # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # # the difference between naive and coarse-to-fine rasterization.
    # raster_settings = RasterizationSettings(
    #     image_size=img_small_size,
    #     blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    #     faces_per_pixel=100,
    # )

    # # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    # silhouette_renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=camera_sfm_small,
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftSilhouetteShader(blend_params=blend_params)
    # )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        # image_size=img_square_size,
        image_size=(640, 640),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return None, phong_renderer


def transform_image(im):
    height = im.shape[0]
    width = im.shape[1]
    center = np.array(im.shape[:2]) / 2
    if height > 480:
        y = int(center[0] - 480/2)
        im = im[y:y+480, :, :]
    elif height < 480:
        pad = int((480 - height)/2)
        if (480 - height) % 2 == 0:
            im = cv2.copyMakeBorder(
                im.copy(), pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            im = cv2.copyMakeBorder(
                im.copy(), pad, pad+1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if width > 640:
        x = int(center[1] - 640/2)
        im = im[:, x:x+640, :]
    elif width < 640:
        pad = int((640-width)/2)
        if (640-width) % 2 == 0:
            im = cv2.copyMakeBorder(
                im.copy(), 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            im = cv2.copyMakeBorder(
                im.copy(), 0, 0, pad, pad+1, cv2.BORDER_CONSTANT, value=0)

    return im

def get_frames(path, step, resize=False):

    frame_path = f"../dataset/{path.replace('_b', '/b', 1)}/frames/"
    # pdb.set_trace()
    fpaths = natsorted(glob.glob(os.path.join(frame_path, '*.jpg')))

    frames = []
    for fp in fpaths[::step]:
        img = imageio.imread(fp)
        h, w, _ = img.shape
        # pdb.set_trace()
        if resize:
            # frames.append(cv2.resize(img, (h // 2, w // 2)) )
            im = cv2.resize(img, (int(w*(517.97/983)),
                        int(h*(517.97/983))))
            im = transform_image(im)
            frames.append(im)
        else:
            frames.append(img)

    return np.array(frames)

def get_meshes(path, step):


    # smpl_paths = natsorted(glob.glob(os.path.join(path, 'frame_*/extrude_pred.obj')))
    # pdb.set_trace()
    # frame_ids = []
    # meshes = []
    # for smp in smpl_paths[::step]:
    #     obj_verts, obj_faces, _ = load_obj(smp, device='cuda:0')
    #     tex = torch.ones_like(obj_verts).unsqueeze(0)
    #     textures = TexturesVertex(verts_features=tex).to('cuda:0')
    #     mesh = Meshes(verts=[obj_verts],faces=[obj_faces.verts_idx],textures=textures)
    #     meshes.append(mesh)
    #     # pdb.set_trace()
    #     if smp.split('/')[4][6:10] != '0000':
    #         frame_id = int(smp.split('/')[4][6:].lstrip('0'))
    #     else:
    #         frame_id = int(0)
    #     frame_ids.append(frame_id)

    smpl_paths = natsorted(glob.glob(os.path.join(path, 'frame_*/arti_pred.obj')))
    frame_ids = []
    meshes = []
    # pdb.set_trace()
    for smp in smpl_paths[::step]:
        obj_verts, obj_faces, _ = load_obj(smp, device='cuda:0')
        # pdb.set_trace()
        # obj_verts[:, 1] = -obj_verts[:, 1]
        # obj_verts[:, 2] = -obj_verts[:, 2]
        # obj_verts[:, 2] = 1 / obj_verts[:, 2]
        # pdb.set_trace()
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        textures = TexturesVertex(verts_features=tex).to('cuda:0')
        mesh = Meshes(verts=[obj_verts],faces=[obj_faces.verts_idx],textures=textures)
        meshes.append(mesh)
        # pdb.set_trace()
        if smp.split('/')[7][6:10] != '0000':
            frame_id = int(smp.split('/')[7][6:].lstrip('0'))
        else:
            frame_id = int(0)
        frame_ids.append(frame_id)

    return meshes, frame_ids

def single_video_vis(image_path, mesh_path, step, output_path):

    images = get_frames(image_path, step, resize=True)
    # pdb.set_trace()
    N, H, W, _ = images.shape
    # pdb.set_trace()

    # focal_x = 983
    # focal_y = 983
    focal_x = 517.97
    focal_y = 517.97
    device = 'cuda:0'
    silhouette_renderer, phong_renderer = initialize_render(device, focal_x=focal_x, focal_y=focal_y, 
                                                                                img_square_size=max(H, W), img_small_size=256)
    # pdb.set_trace()
    
    meshes, frame_ids= get_meshes(mesh_path, step)


    visualize_optimal_poses_humans_video(meshes, images, phong_renderer, phong_renderer, frame_ids, output_path)

def test_sample(images, mesh_path, step, output_path):
    N, H, W, _ = images.shape
    focal_x = 517.97
    focal_y = 517.97
    device = 'cuda:0'
    # pdb.set_trace()
    silhouette_renderer, phong_renderer = initialize_render(device, focal_x=focal_x, focal_y=focal_y, 
                                                                                img_square_size=max(H, W), img_small_size=256)
    # pdb.set_trace()
    
    meshes, frame_ids= get_meshes(mesh_path, step)


    visualize_optimal_poses_humans_video(meshes, images, phong_renderer, phong_renderer, frame_ids, output_path)


def main():
    # parser = argparse.ArgumentParser(
    #     description="A script that generates results of articulation prediction."
    # )
    # parser.add_argument("--image_path", required=True, help="input frames name")
    # # parser.add_argument("--mesh_path", required=True, help = 'input meshes path')
    # # parser.add_argument("--output", required=True, help="output directory")
    # parser.add_argument("--step", default=1)
    # args = parser.parse_args()

    # videos = open(args.image_path,'r').read().splitlines()
    # step = args.step
    # # pdb.set_trace()

    # for video in tqdm(videos):
    #     # pdb.set_trace()
    #     cat = video.split('_')[0]
    #     idx = video.split('_')[1]
    #     mesh_path = f'output/output_val_extrude_0.05/{cat}/{idx}/'
    #     # output_path = f'output/output_gif_new/{cat}/{idx}/'
    #     output_path = f'output/output_gif_check/{cat}/{idx}/'
    #     single_video_vis(video, mesh_path, step, output_path)

    output_path = 'output/output_gif_sample_1'
    mesh_path = "/localhome/xsa55/Xiaohao/Articulation3D/articulation3d/output_test_modified"
    cap = cv2.VideoCapture("/localhome/xsa55/Xiaohao/Articulation3D/articulation3d/tools/demo/teaser.mp4")
    step = 1
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        h, w, n = frame.shape
        # pdb.set_trace()
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h, w, _ = frame.shape
        # im = cv2.resize(frame, (int(w*(517.97/983)),
        #                 int(h*(517.97/983))))
        # im = transform_image(im)
        frames.append(frame)
    test_sample(np.array(frames), mesh_path, step, output_path)


if __name__ == '__main__':
    main()