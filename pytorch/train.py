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
from model import myNet

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
        print (f'scanning for exr files in {self.dataset_path}...')
        self.imgs = self.find_images(self.dataset_path)
        print (f'found {len(self.imgs)} images.')

    def reshuffle(self):
        random.shuffle(self.imgs)

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
        return img

def write_image_file(file_path, image_data, image_spec):
    out = oiio.ImageOutput.create(file_path)
    if out:
        out.open(file_path, image_spec)
        out.write_image(image_data)
        out.close ()

def augment(img): 
    generalize = 100

    # Horizontal flip (reverse width)
    if random.uniform(0, 1) < (generalize / 100):
        if random.uniform(0, 1) < 0.5:
            img = np.flip(img, axis=1)

    # Vertical flip (reverse height)
    if random.uniform(0, 1) < (generalize / 100):
        if random.uniform(0, 1) < 0.5:
            img = np.flip(img, axis=0)

    # Rotation
    if random.uniform(0, 1) < (generalize / 100):
        p = random.uniform(0, 1)
        if p < 0.25:
            img = np.rot90(img, k=-1, axes=(0, 1))
        elif p < 0.5:
            img = np.rot90(img, k=1, axes=(0, 1))
        elif p < 0.75:
            img = np.rot90(img, k=2, axes=(0, 1))

    return np.ascontiguousarray(img)

def main():
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--epochs', type=int, default=10000, help='Train one cycle for N epochs (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (int) (default: 1)')
    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')

    net = myNet().to(device)

    training_dataset = MLDataset(
        os.path.join(args.dataset_path, 'train'),
        device=device, 
        )
        
    validation_dataset = MLDataset(
        os.path.join(args.dataset_path, 'validate'),
        device=device, 
        )

    steps_per_epoch = -(-len(training_dataset.imgs) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    preview_index = 0

    start_timestamp = time.time()
    time_stamp = time.time()

    for epoch in range (args.epochs):
        for step in range (steps_per_epoch):
            data_time = time.time() - time_stamp
            time_stamp = time.time()
            # Get data here:

            first_index = step*args.batch_size
            last_index = min(len(training_dataset)-1, (step+1)*args.batch_size)
            batch_list = [augment(training_dataset[x]) for x in range(first_index, last_index)]
            # batch_list = [training_dataset[x] for x in range(first_index, last_index)]

            if not batch_list:
                continue
            tensor_list = [torch.from_numpy(array).permute(2, 1, 0) for array in batch_list]
            

            batch_tensor = torch.stack(tensor_list)
            batch_tensor = batch_tensor.to(device=device, dtype=torch.float32)

            data_time += time.time() - time_stamp
            data_time_str = str(f'{data_time:.2f}')
            time_stamp = time.time()

            # train here

            mask = net(batch_tensor[:,:3,:,:])
            
            if platform.system() == 'Darwin':
                torch.mps.synchronize()
            else:
                torch.cuda.synchronize(device=device)

            train_time = time.time() - time_stamp
            time_stamp = time.time()
            train_time_str = str(f'{train_time:.2f}')
            # end of training

            if step % 100 == 1:
                pr_src = batch_tensor[:, :3, :, :]
                pr_mask = batch_tensor[:, -1:, :, :].repeat_interleave(3, dim=1)
                pr_pred = mask.repeat_interleave(3, dim=1)
                preview_index = preview_index + 1 if preview_index < 9 else 0
                preview_folder = os.path.join(os.path.dirname(os.path.abspath(args.dataset_path)), 'preview')
                if not os.path.isdir(preview_folder):
                    os.makedirs(preview_folder)
                n, ref_d, ref_h, ref_w = pr_src.shape

                src = pr_src[0].numpy(force=True).transpose(2, 1, 0).copy() * 255
                gt = pr_mask[0].numpy(force=True).transpose(2, 1, 0).copy() * 255
                pred = pr_pred[0].numpy(force=True).transpose(2, 1, 0).copy() * 255

                spec = oiio.ImageSpec(ref_w, ref_h, 3, 'uint8')
                write_image_file(
                    os.path.join(preview_folder, f'{preview_index:02}_src.jpg'),
                    src.astype(np.uint8),
                    spec)

                write_image_file(
                    os.path.join(preview_folder, f'{preview_index:02}_gt.jpg'),
                    gt.astype(np.uint8),
                    spec)

                write_image_file(
                    os.path.join(preview_folder, f'{preview_index:02}_pred.jpg'),
                    pred.astype(np.uint8),
                    spec)


            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            current_lr_str = ''
            loss_str = ''

            print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, [Step: {step+1} / {steps_per_epoch}], Lr: {current_lr_str}, Loss: {loss_str}', end='')         

        training_dataset.reshuffle()


if __name__ == "__main__":
    main()