import json
import os
import random
from pathlib import Path
from random import shuffle
import numpy as np
import tifffile
import cv2
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, batch_size=1, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.root = root
        
        # GT : Ground Truth
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

        # Set Data loader metadata
        set_file = Path(Path.cwd() / "settings.json")
        if not set_file.is_file():
            raise ValueError("The JSON file path provided is not a file.")
        settings = type('Settings', (object,), json.loads(set_file.read_text(encoding='utf-8')))()

        # Read metadata
        ge_metadata = np.load(settings.ge_metadata)
        self.ge_means = ge_metadata['means']
        self.ge_stds = ge_metadata['stds']
        wv2_metadata = np.load(settings.wv2_metadata)
        self.wv2_means = wv2_metadata['means']
        self.wv2_stds = wv2_metadata['stds']
        wv3_metadata = np.load(settings.wv3_metadata)
        self.wv3_means = wv3_metadata['means']
        self.wv3_stds = wv3_metadata['stds']
        wv4_metadata = np.load(settings.wv4_metadata)
        self.wv4_means = wv4_metadata['means']
        self.wv4_stds = wv4_metadata['stds']

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]

        # Read image and pred
        img, mask = self.load_data_from_file(image_path)
        
        # Data augmentation
        if self.mode=='train':
            img, mask = self.random_rotate_flip(img, mask)
        
        # Normalization
        img = self.normalize_meanstd(img, image_path)
        
        # Resize data
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # prepare arrays dtype and dimensions
        img = np.float32(img)
        mask = np.expand_dims(mask, -1).astype(np.float32)

        # prepare tensors
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)

        return img, mask

    def load_data_from_file(self, data_path):
        data = tifffile.imread(data_path)
        img = np.copy(data[...,:-1])
        mask = np.copy(data[...,-1])
        return img, mask
    
    def normalize_meanstd(self, img, path):
        if '_GE01_' in path:
            means = self.ge_means
            stds = self.ge_stds
        elif '_WV02_' in path:
            means = self.wv2_means
            stds = self.wv2_stds
        elif '_WV03_' in path:
            means = self.wv3_means
            stds = self.wv3_stds
        elif '_WV04_' in path:
            means = self.wv4_means
            stds = self.wv4_stds
        return np.true_divide(np.subtract(img, means), stds)
    
    def random_rotate_flip(self, img, pred):
        # chose a random transformation
        rot_to_do = random.sample([None, 1, 2, 3], 1)[0]
        flip_to_do = random.sample([None, 0, 1], 1)[0]
        # do transformation on image and mask
        if rot_to_do is not None:
            img = np.rot90(img, k=rot_to_do, axes=(0,1))
            pred = np.rot90(pred, k=rot_to_do, axes=(0,1))
        if flip_to_do is not None:
            img = np.flip(img, flip_to_do)
            pred = np.flip(pred, flip_to_do)
        return img, pred

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(root=image_path, image_size=image_size, batch_size=batch_size, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
