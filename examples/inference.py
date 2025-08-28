"""Inference script for segmentation models."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import mini_monai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_monai.transforms import (
    Compose, LoadImage, ScaleIntensity, NormalizeIntensity,
    Resize, ToTensor, AddChannel
)
from mini_monai.datasets import create_test_dataset
from mini_monai.networks import UNet
from mini_monai.utils import load_checkpoint, visualize_slice, set_determinism


def main():
    """Run the main inference process."""
    # Set random seed for reproducibility
    set_determinism(seed=42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a test image for demonstration
    print("Creating test image...")
    image_size = (64, 64)  # 2D images for simplicity
    num_classes = 3  # Background + 2 classes
    test_images, _ = create_test_dataset(
        num_samples=1, 
        image_size=image_size, 
        num_classes=num_classes
    )
    test_image = test_images[0]
    
    # Define transforms
    transforms = Compose([
        AddChannel(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor()
    ])
    
    # Apply transforms
    data = {"image": test_image}
    data = transforms(data)
    image = data["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Load model
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    
    # Check if model checkpoint exists
    if os.path.exists("best_model.pth"):
        print("Loading model checkpoint...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Dummy optimizer for loading checkpoint
        model, _, _ = load_checkpoint(model, optimizer, "best_model.pth")
    else:
        print("No model checkpoint found. Using untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    print("Performing inference...")
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).detach().cpu().numpy()
    
    # Visualize results
    print("Visualizing results...")
    fig = visualize_slice(image[0].detach().cpu().numpy(), prediction=pred[0])
    plt.savefig("inference_result.png")
    plt.close(fig)
    
    print(f"Inference completed! Results saved to 'inference_result.png'")
    
    # Print segmentation statistics
    print("\nSegmentation Statistics:")
    for i in range(num_classes):
        pixel_count = np.sum(pred[0] == i)
        percentage = (pixel_count / pred[0].size) * 100
        print(f"Class {i}: {pixel_count} pixels ({percentage:.2f}%)")


def load_and_segment_nifti(model_path, image_path, output_path=None):
    """Load a NIfTI image and perform segmentation.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the input NIfTI image
        output_path: Optional path to save visualization output
        
    Returns:
        Segmentation mask as numpy array
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transforms
    transforms = Compose([
        LoadImage(keys=["image"]),
        AddChannel(keys=["image"]),
        ScaleIntensity(keys=["image"], minv=0.0, maxv=1.0),
        ToTensor(keys=["image"])
    ])
    
    # Load image
    data = {"image": image_path}
    data = transforms(data)
    image = data["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Load model
    num_classes = 3  # Adjust based on your model
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Dummy optimizer for loading checkpoint
    model, _, _ = load_checkpoint(model, optimizer, model_path)
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).detach().cpu().numpy()
    
    # Save result if output path is provided
    if output_path:
        fig = visualize_slice(image[0].detach().cpu().numpy(), prediction=pred[0])
        plt.savefig(output_path)
        plt.close(fig)
    
    return pred[0]  # Return the segmentation mask


if __name__ == "__main__":
    main()