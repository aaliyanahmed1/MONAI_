"""Transforms for medical image processing."""

import numpy as np
import torch
import SimpleITK as sitk
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union


class Transform:
    """Base class for all transforms in Mini MONAI."""
    
    def __call__(self, data):
        """Apply the transform to `data`."""
        raise NotImplementedError


class Compose(Transform):
    """Compose several transforms together."""
    
    def __init__(self, transforms=None):
        self.transforms = transforms or []
        
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class LoadImage(Transform):
    """Load medical images from file."""
    
    def __init__(self, keys=('image',), reader=None):
        self.keys = keys
        self.reader = reader
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                if isinstance(d[key], str):
                    # Load the image using SimpleITK
                    if self.reader is None:
                        img = sitk.ReadImage(d[key])
                        d[key] = sitk.GetArrayFromImage(img)
                    else:
                        d[key] = self.reader(d[key])
        return d


class ScaleIntensity(Transform):
    """Scale the intensity of input image to the given range."""
    
    def __init__(self, keys=('image',), minv=0.0, maxv=1.0):
        self.keys = keys
        self.minv = minv
        self.maxv = maxv
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                if img.max() != img.min():
                    img = (img - img.min()) / (img.max() - img.min())
                    img = img * (self.maxv - self.minv) + self.minv
                d[key] = img
        return d


class NormalizeIntensity(Transform):
    """Normalize the intensity of input image by mean and standard deviation."""
    
    def __init__(self, keys=('image',), nonzero=False):
        self.keys = keys
        self.nonzero = nonzero
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                if self.nonzero:
                    mask = img != 0
                    if mask.any():
                        mean = img[mask].mean()
                        std = img[mask].std()
                        if std > 0:
                            img = (img - mean) / std
                else:
                    mean = img.mean()
                    std = img.std()
                    if std > 0:
                        img = (img - mean) / std
                d[key] = img
        return d


class Resize(Transform):
    """Resize the input image to given spatial size."""
    
    def __init__(self, keys=('image',), spatial_size=None, mode='bilinear'):
        self.keys = keys
        self.spatial_size = spatial_size
        self.mode = mode
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                if torch.is_tensor(img):
                    # Handle PyTorch tensors
                    import torch.nn.functional as F
                    if img.dim() == 3:  # 3D image (C, H, W)
                        img = img.unsqueeze(0)  # Add batch dimension
                        img = F.interpolate(img, size=self.spatial_size, mode=self.mode)
                        img = img.squeeze(0)  # Remove batch dimension
                    elif img.dim() == 4:  # 4D image (C, D, H, W)
                        img = img.unsqueeze(0)  # Add batch dimension
                        img = F.interpolate(img, size=self.spatial_size, mode=self.mode)
                        img = img.squeeze(0)  # Remove batch dimension
                else:
                    # Handle numpy arrays
                    from skimage.transform import resize
                    img = resize(img, self.spatial_size, order=1, preserve_range=True)
                d[key] = img
        return d


class ToTensor(Transform):
    """Convert numpy array to PyTorch tensor."""
    
    def __init__(self, keys=('image', 'label'), dtype=torch.float32):
        self.keys = keys
        self.dtype = dtype
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                if not torch.is_tensor(d[key]):
                    d[key] = torch.as_tensor(d[key], dtype=self.dtype)
        return d


class AddChannel(Transform):
    """Add a channel dimension to the input image."""
    
    def __init__(self, keys=('image', 'label')):
        self.keys = keys
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                if torch.is_tensor(img):
                    if img.ndimension() == 2 or img.ndimension() == 3:
                        # Add channel dimension (C, H, W) or (C, D, H, W)
                        d[key] = img.unsqueeze(0)
                else:  # numpy array
                    if img.ndim == 2 or img.ndim == 3:
                        # Add channel dimension (C, H, W) or (C, D, H, W)
                        d[key] = np.expand_dims(img, axis=0)
        return d


class RandFlip(Transform):
    """Randomly flip the input image along specified spatial axes."""
    
    def __init__(self, keys=('image', 'label'), spatial_axis=0, prob=0.5):
        self.keys = keys
        self.spatial_axis = spatial_axis
        self.prob = prob
        
    def __call__(self, data):
        d = dict(data)
        if np.random.random() < self.prob:
            for key in self.keys:
                if key in d:
                    img = d[key]
                    if torch.is_tensor(img):
                        # PyTorch tensor flip
                        dims = list(range(img.dim()))
                        actual_axis = dims[self.spatial_axis + 1]  # +1 because of channel dimension
                        d[key] = torch.flip(img, [actual_axis])
                    else:
                        # Numpy array flip
                        d[key] = np.flip(img, axis=self.spatial_axis + 1)  # +1 because of channel dimension
        return d


class RandRotate90(Transform):
    """Randomly rotate the input image by 90 degrees."""
    
    def __init__(self, keys=('image', 'label'), prob=0.5, max_k=3):
        self.keys = keys
        self.prob = prob
        self.max_k = max_k
        
    def __call__(self, data):
        d = dict(data)
        if np.random.random() < self.prob:
            k = np.random.randint(1, self.max_k + 1)
            for key in self.keys:
                if key in d:
                    img = d[key]
                    if torch.is_tensor(img):
                        # PyTorch tensor rotation
                        if img.dim() == 3:  # (C, H, W)
                            d[key] = torch.rot90(img, k, dims=[1, 2])
                        elif img.dim() == 4:  # (C, D, H, W)
                            d[key] = torch.rot90(img, k, dims=[2, 3])
                    else:
                        # Numpy array rotation
                        if img.ndim == 3:  # (C, H, W)
                            d[key] = np.rot90(img, k, axes=(1, 2))
                        elif img.ndim == 4:  # (C, D, H, W)
                            d[key] = np.rot90(img, k, axes=(2, 3))
        return d