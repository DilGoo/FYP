import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.transform import resize

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
        y = np.load(self.target[idx])
        if self.downscale:
            y = scipy.ndimage.zoom(y, (0.3, 0.3, 0.3), order=0)
        rmin, rmax, cmin, cmax, zmin, zmax = self.get_padded_bbox(y)
        y = y[rmin:rmax, cmin:cmax, zmin:zmax]
        y = self.pad_to_cuboid(y)
        y = resize(y, (128, 128, 128), order=0)
        y = self.preprocess_mask_labels(y)
        y = torch.from_numpy(y)
        y = y.float()
        
        x = np.load(self.source[idx])
        if self.downscale:
            x = scipy.ndimage.zoom(x, (1, 0.3, 0.3, 0.3), order=0)
        x = x[:, rmin:rmax, cmin:cmax, zmin:zmax]
        x = self.pad_x_to_cuboid(x)
        x = resize(x, (4, 128, 128, 128), order=0)
        x = self.normalize(x)
        x = torch.from_numpy(x)
        x = x.float()
        
        return x,y
    
    
    def get_padded_bbox(self, labels, padding=5):
        # crop to bbox
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bbox(labels)
        cropped = labels[xmin:xmax, ymin:ymax, zmin:zmax]

        # add padding
        xmin -= padding
        ymin -= padding
        zmin -= padding
        xmax += padding
        ymax += padding
        zmax += padding

        # make sure coords are within bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        zmin = max(0, zmin)
        xmax = min(240, xmax)
        ymax = min(240, ymax)
        zmax = min(155, zmax)

        return xmin, xmax, ymin, ymax, zmin, zmax

    
    def pad_to_cuboid(self, arr):
        xlen = arr.shape[0]
        ylen = arr.shape[1]
        zlen = arr.shape[2]
        if (xlen != ylen) or (ylen != zlen) or (zlen != xlen):
            size = max(xlen, ylen, zlen)

            a = (size - xlen) // 2
            aa = size - a - xlen

            b = (size - ylen) // 2
            bb = size - b - ylen

            c = (size - zlen) // 2
            cc = size - c - zlen

            return np.pad(arr, [(a, aa), (b, bb), (c, cc)], mode='constant')
        return arr
    
    
    def pad_x_to_cuboid(self, arr):
        xlen = arr.shape[1]
        ylen = arr.shape[2]
        zlen = arr.shape[3]
        if (xlen != ylen) or (ylen != zlen) or (zlen != xlen):
            size = max(xlen, ylen, zlen)

            a = (size - xlen) // 2
            aa = size - a - xlen

            b = (size - ylen) // 2
            bb = size - b - ylen

            c = (size - zlen) // 2
            cc = size - c - zlen

            return np.pad(arr, [(0, 0), (a, aa), (b, bb), (c, cc)], mode='constant')
        return arr
    
    def get_bbox(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

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