import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.ndimage
from skimage import measure

class brats3dDataset(Dataset):
    """
    Needs a rootPath.
    Expects to find 'source' and 'target' folders.
    Expects to deal with preprocessed numpy files.
    """
    def __init__(self, rootPath, downscale=False, crop_bbox=True):
        self.downscale = downscale
        self.crop_bbox = crop_bbox
        
        self.source = []
        self.target = []
        
        pathSource = os.path.join(rootPath, 'source')
        imgPaths = os.listdir(pathSource)
        for path in imgPaths:
            if path.endswith('.npy'):
                self.source.append(os.path.join(pathSource, path))
        
        pathTarget = os.path.join(rootPath, 'target')
        imgPaths = os.listdir(pathTarget)
        for path in imgPaths:
            if path.endswith('.npy'):
                self.target.append(os.path.join(pathTarget, path))
        
        # Do not train on last 200 images, keep for validation
        n = 200
        self.source = self.source[:-n]
        self.target = self.target[:-n]
        
    def __len__(self):
        assert len(self.source) == len(self.target)
        return len(self.source)
    
    def __getitem__(self,idx):
        # crop to brain extent
        x = np.load(self.source[idx])
        rmin, rmax, cmin, cmax, zmin, zmax = self.get_bbox(x)
        x = x[:, rmin:rmax, cmin:cmax, zmin:zmax]
        
        # normalize
        x = self.normalize(x)
        x = torch.from_numpy(x)
        x = x.float()
        
        y = np.load(self.target[idx])
        y = y[rmin:rmax, cmin:cmax, zmin:zmax]
        y = self.preprocess_mask_labels(y)
        y = torch.from_numpy(y)
        y = y.float()
        
        return x,y
    
    
    def get_bbox(self, img):
        r = np.any(img[0], axis=(1, 2))
        c = np.any(img[0], axis=(0, 2))
        z = np.any(img[0], axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax
    
    def normalize(self, data: np.ndarray):
        return (data - data.mean()) / data.std()
    
    
    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_TC, mask_WT , mask_ET])
        return mask