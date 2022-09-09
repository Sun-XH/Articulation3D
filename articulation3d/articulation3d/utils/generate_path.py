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

all_path = glob.glob('../dataset/*/*')
with open('all_video.txt', 'w') as f:
    for path in all_path:
        tmp = path[11:].replace('/', '_')
        f.write(tmp)
        f.write('\n')