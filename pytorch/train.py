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



class MLDataset(torch.utils.data.Dataset):
    def __init__(   
        self,
        dataset_path, 
        device = None,
        frame_size=128,
        generalize = 80,
        ):

        self.dataset_path = dataset_path
        self.device = device
        self.frame_size = frame_size
        self.generalize = generalize
        print (f'scanning for exr files in {self.dataset_path}...')
        self.imgs = self.find_images(self.dataset_path)
        print (f'found {len(self.imgs)} images.')

    def find_images(self, path, ext_list=['.exr']):
        imgs = set()
        # Walk through all directories and files in the given path
        for root, dirs, files in os.walk(path):
            # exclude certain folders
            for file in files:
                if any(file.lower().endswith(ext) for ext in ext_list):
                    imgs.add(os.path.join(
                        os.path.abspath(root),
                        file
                        ))
        return list(imgs)

    def read_image_file(self, file_path):
        inp = oiio.ImageInput.open(file_path)
        if inp:
            spec = inp.spec()
            # height = spec.height
            # width = spec.width
            channels = spec.nchannels
            result = inp.read_image(0, 0, 0, channels)
            inp.close()
        return result
    
    def crop(self, img, h, w):
        try:
            np.random.seed(None)
            ih, iw, _ = img.shape
            x = np.random.randint(0, ih - h + 1)
            y = np.random.randint(0, iw - w + 1)
            img = img[x:x+h, y:y+w, :]
        except:
            print (f'Cannot crop: h: {h}, w: {w}, img shape: {img.shape}')
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.read_image_file(self.imgs[index])
        img = self.crop(img, self.frame_size, self.frame_size)

        # Horizontal flip (reverse width)
        if random.uniform(0, 1) < (self.generalize / 100):
            if random.uniform(0, 1) < 0.5:
                img = np.flip(img, axis=1)

        # Vertical flip (reverse height)
        if random.uniform(0, 1) < (self.generalize / 100):
            if random.uniform(0, 1) < 0.5:
                img = np.flip(img, axis=0)

        # Rotation
        if random.uniform(0, 1) < (self.generalize / 100):
            p = random.uniform(0, 1)
            if p < 0.25:
                img = np.rot90(img, k=-1, axes=(0, 1))
            elif p < 0.5:
                img = np.rot90(img, k=1, axes=(0, 1))
            elif p < 0.75:
                img = np.rot90(img, k=2, axes=(0, 1))

        return img

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
    
    validation_dataset = MLDataset(
        os.path.join(args.dataset_path, 'validate'),
        device=device, 
        )
    
    for img in training_dataset:
        print (f'shape: {img.shape}')

if __name__ == "__main__":
    main()