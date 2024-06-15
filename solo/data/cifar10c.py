import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets


corruptions = [
"natural",
"gaussian_noise",
"shot_noise",
"speckle_noise",
"impulse_noise",
"defocus_blur",
"gaussian_blur",
"motion_blur",
"zoom_blur",
"snow",
"fog",
"brightness",
"contrast",
"elastic_transform",
"pixelate",
"jpeg_compression",
"spatter",
"saturate",
"frost",
]

class CIFARC(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFARC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)
