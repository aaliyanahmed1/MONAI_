"""Utility functions for medical image processing."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple, Union


def one_hot(labels, num_classes):
    """Convert labels to one-hot encoding."""
    if torch.is_tensor(labels):
        shape = labels.shape
        # Convert to numpy for processing
        labels_np = labels.detach().cpu().numpy()
        one_hot = np.zeros((shape[0], num_classes) + shape[1:], dtype=np.float32)
        for i in range(num_classes):
            one_hot[:, i, ...] = (labels_np == i)
        # Convert back to tensor
        return torch.from_numpy(one_hot).to(device=labels.device)
    else:
        shape = labels.shape
        one_hot = np.zeros((shape[0], num_classes) + shape[1:], dtype=np.float32)
        for i in range(num_classes):
            one_hot[:, i, ...] = (labels == i)
        return one_hot


def dice_score(y_pred, y_true, num_classes=None):
    """Calculate Dice score between prediction and ground truth."""
    if num_classes is not None:
        # Convert to one-hot encoding
        y_true = one_hot(y_true, num_classes)
        
    # Flatten the tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Calculate Dice score
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    
    if union == 0:
        return 1.0  # Both are empty, perfect match
    
    return 2.0 * intersection / union


def dice_loss(y_pred, y_true, num_classes=None):
    """Calculate Dice loss for segmentation."""
    return 1 - dice_score(y_pred, y_true, num_classes)


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return model, optimizer, epoch


def visualize_slice(image, label=None, prediction=None, slice_idx=None, axis=0):
    """Visualize a slice of 3D medical image with optional label and prediction."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    if image.ndim == 4:  # (C, D, H, W)
        image = image[0]  # Take first channel
    
    if slice_idx is None:
        # Take middle slice by default
        slice_idx = image.shape[axis] // 2
    
    # Extract the slice
    if axis == 0:
        image_slice = image[slice_idx]
    elif axis == 1:
        image_slice = image[:, slice_idx]
    else:  # axis == 2
        image_slice = image[:, :, slice_idx]
    
    fig, axes = plt.subplots(1, 1 + (label is not None) + (prediction is not None), figsize=(12, 4))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot image
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Plot label if provided
    if label is not None:
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        
        if label.ndim == 4:  # (C, D, H, W)
            label = np.argmax(label, axis=0)  # Convert one-hot to class indices
        
        if axis == 0:
            label_slice = label[slice_idx]
        elif axis == 1:
            label_slice = label[:, slice_idx]
        else:  # axis == 2
            label_slice = label[:, :, slice_idx]
        
        axes[1].imshow(label_slice, cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
    
    # Plot prediction if provided
    if prediction is not None:
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        
        if prediction.ndim == 4:  # (C, D, H, W)
            prediction = np.argmax(prediction, axis=0)  # Convert one-hot to class indices
        
        if axis == 0:
            pred_slice = prediction[slice_idx]
        elif axis == 1:
            pred_slice = prediction[:, slice_idx]
        else:  # axis == 2:
            pred_slice = prediction[:, :, slice_idx]
        
        idx = 1 + (label is not None)
        axes[idx].imshow(pred_slice, cmap='viridis')
        axes[idx].set_title('Prediction')
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def set_determinism(seed=None):
    """Set random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False