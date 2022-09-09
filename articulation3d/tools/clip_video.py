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
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input video series name")
    args = parser.parse_args()

    # read path
    videos = open(args.input,'r').read().splitlines()

    # split video into parts if have different moving status
    for video in videos:
        video_path = f"../dataset/{video.replace('_b', '/b', 1)}/frames/video.mp4"
        motion_gt_path = f"../dataset/{video.replace('_b', '/b', 1)}/jointstate.txt"
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        try:
            motion_gt = np.loadtxt(motion_gt_path, delimiter=",", unpack=False)
        except:
            pdb.set_trace()
        

        if motion_gt[0] == motion_gt[-1]:
            max_value = np.amax(motion_gt)
            max_ids = np.where(motion_gt == max_value)
            if len(max_ids[0]) != 1:
                mid = int(len(max_ids[0])/2)
                max_id = max_ids[0][mid]
            else:
                max_id = max_ids[0]

            frames_open = []
            frames_close = []
            for i, im in enumerate(tqdm(reader)):
                if i <= max_id:
                    frames_open.append(im)
                else:
                    frames_close.append(im)
            
            reader.close()

            output_path_open = f"../dataset/{video.replace('_b', '/b', 1)}/frames/video_open.mp4"
            output_path_close = f"../dataset/{video.replace('_b', '/b', 1)}/frames/video_close.mp4"
            
            writer_open = imageio.get_writer(output_path_open, fps=fps)
            for i, im in (enumerate(frames_open)):
                writer_open.append_data(im)
            
            writer_close = imageio.get_writer(output_path_close, fps=fps)
            for i, im in (enumerate(frames_close)):
                writer_close.append_data(im)
            

if __name__ == "__main__":
    main()
