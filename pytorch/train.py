import os
import sys
import random
import shutil
import argparse
import time
import platform

import numpy as np
import torch
import OpenImageIO as oiio

from pprint import pprint

def find_folders_with_images(path, ext_list=['.exr']):
    directories_with_imgs = set()
    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(path):
        # exclude certain folders
        if 'preview' in root:
            continue
        if 'eval' in root:
            continue
        for file in files:
            if any(file.lower().endswith(ext) for ext in ext_list):
                directories_with_imgs.add(root)
                break  # No need to check other files in the same directory
    return directories_with_imgs

class MLDataset(torch.utils.data.Dataset):
    def __init__(   
        self,
        dataset_path, 
        device = None,
        frame_size=128,
        generalize = 80,
        ):

        self.dataset_path = dataset_path
        self.generalize = generalize
        print (f'scanning for exr files in {self.dataset_path}...')
        self.folders_with_imgs = find_folders_with_images(self.dataset_path)
        print (f'found {len(self.folders_with_imgs)} clip folders.')

def main():
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--onecycle', type=int, default=10000, help='Train one cycle for N epochs (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (int) (default: 1)')
    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')

    training_dataset = MLDataset(
        os.path.join(args.dataset_path, 'train'), 
        device=device, 
        )
    

if __name__ == "__main__":
    main()