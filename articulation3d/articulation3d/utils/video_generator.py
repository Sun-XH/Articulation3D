import argparse
import json
import numpy as np
import os
from regex import P
import torch
import torch.nn.functional as F
from collections import defaultdict
import cv2
from tqdm import tqdm
import pickle
import imageio
import random
import math
import pdb
import glob


def save(path):
    os.chdir(path)
    v_path = f'{path}video.mp4'
    # pdb.set_trace()
    if not os.path.exists(v_path):
        os.system('ffmpeg -r 10 -i images-%04d.jpg -c:v libx264 -vf fps=10 -pix_fmt yuv420p video.mp4')


all_path = glob.glob('/localhome/xsa55/Xiaohao/Articulation3D/dataset/*/*/frames/')
for path in all_path:
    # pdb.set_trace()
    save(path)