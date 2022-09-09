import random
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import os, re, glob, natsort, json

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def visualize_optimal_poses_humans_video(model, images, vis_renderer, right_renderer):


    for idx in range(images.shape[0]):
        R, T = look_at_view_transform(0.1, 0.0, 0.0,device='cuda')
        T[0,2] = 0.0  # manually set to zero
        MESH = model.render(idx)

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

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(images[idx])
        ax.imshow(proj_frame, alpha=0.4 )
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

        plt.show()