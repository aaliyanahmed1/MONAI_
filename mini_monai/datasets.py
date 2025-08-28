"""Dataset classes for medical image processing."""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import SimpleITK as sitk
from typing import Dict, List, Optional, Sequence, Union


class MedicalImageDataset(Dataset):
    """Generic dataset for medical images."""
    
    def __init__(self, data, transform=None):
        """
        Args:
            data: List of dictionaries containing paths to images and labels
            transform: Transform to apply to each data item
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_i = self.data[index]
        if self.transform is not None:
            data_i = self.transform(data_i)
        return data_i


class NiftiDataset(MedicalImageDataset):
    """Dataset for loading NIfTI format medical images."""
    
    def __init__(self, image_files, label_files=None, transform=None):
        """
        Args:
            image_files: List of paths to image files
            label_files: List of paths to label files (optional)
            transform: Transform to apply to each data item
        """
        data = []
        for i, image_file in enumerate(image_files):
            item = {"image": image_file}
            if label_files is not None and i < len(label_files):
                item["label"] = label_files[i]
            data.append(item)
        super().__init__(data, transform)


def load_nifti(file_path):
    """Load a NIfTI file and return as a numpy array."""
    img = nib.load(file_path)
    return img.get_fdata()


def load_sitk_image(file_path):
    """Load a medical image using SimpleITK and return as a numpy array."""
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)


class CacheDataset(MedicalImageDataset):
    """Dataset with cache mechanism that loads data only once."""
    
    def __init__(self, data, transform=None, cache_rate=1.0, cache_num=sys.maxsize):
        """
        Args:
            data: Input data list
            transform: Transform to apply to each data item
            cache_rate: Percentage of data to be cached (default: 1.0 = 100%)
            cache_num: Maximum number of items to be cached
        """
        super().__init__(data, transform)
        self.cache_rate = cache_rate
        self.cache_num = cache_num
        self._cache = {}
        
        if self.cache_rate > 0.0:
            self._cache_data()
    
    def _cache_data(self):
        """Cache data based on cache_rate and cache_num."""
        if self.cache_rate <= 0.0:
            return
        
        cache_num = min(int(len(self.data) * self.cache_rate), self.cache_num)
        for i in range(cache_num):
            data_i = super().__getitem__(i)
            self._cache[i] = data_i
    
    def __getitem__(self, index):
        if index in self._cache:
            return self._cache[index]
        
        data_i = super().__getitem__(index)
        return data_i


def create_test_dataset(num_samples=10, image_size=(20, 20, 20), num_classes=2):
    """Create a synthetic dataset for testing."""
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Create random image
        image = np.random.rand(*image_size).astype(np.float32)
        
        # Create random segmentation mask
        label = np.zeros(image_size, dtype=np.int64)
        for c in range(1, num_classes):
            mask = np.random.rand(*image_size) > 0.8
            label[mask] = c
        
        images.append(image)
        labels.append(label)
    
    return images, labels