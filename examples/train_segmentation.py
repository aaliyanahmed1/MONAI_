"""Training script for segmentation models."""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import mini_monai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_monai.transforms import (
    Compose, LoadImage, ScaleIntensity, NormalizeIntensity,
    Resize, ToTensor, AddChannel, RandFlip, RandRotate90
)
from mini_monai.datasets import create_test_dataset, NiftiDataset
from mini_monai.networks import UNet
from mini_monai.utils import dice_loss, dice_score, save_checkpoint, visualize_slice, set_determinism


def main():
    """Run the main training process."""
    # Set random seed for reproducibility
    set_determinism(seed=42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic dataset for demonstration
    print("Creating synthetic dataset...")
    image_size = (64, 64)  # 2D images for simplicity
    num_classes = 3  # Background + 2 classes
    images, labels = create_test_dataset(
        num_samples=20, 
        image_size=image_size, 
        num_classes=num_classes
    )
    
    # Split into train and validation sets
    train_images, train_labels = images[:15], labels[:15]
    val_images, val_labels = images[15:], labels[15:]
    
    # Define transforms
    train_transforms = Compose([
        AddChannel(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandFlip(prob=0.5),
        RandRotate90(prob=0.5),
        ToTensor()
    ])
    
    val_transforms = Compose([
        AddChannel(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor()
    ])
    
    # Create datasets
    train_data = []
    for i in range(len(train_images)):
        train_data.append({"image": train_images[i], "label": train_labels[i]})
    
    val_data = []
    for i in range(len(val_images)):
        val_data.append({"image": val_images[i], "label": val_labels[i]})
    
    # Create data loaders
    train_ds = NiftiDataset(train_images, train_labels, transform=train_transforms)
    val_ds = NiftiDataset(val_images, val_labels, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
    
    # Create model
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_dice = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1).long())
            
            # Calculate Dice score
            probs = torch.softmax(outputs, dim=1)
            dice = 1 - dice_loss(probs, labels.squeeze(1), num_classes=num_classes)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice.item()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(1).long())
                
                probs = torch.softmax(outputs, dim=1)
                dice = 1 - dice_loss(probs, labels.squeeze(1), num_classes=num_classes)
                
                val_loss += loss.item()
                val_dice += dice.item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, "best_model.pth")
            print("Saved best model checkpoint")
        
        # Visualize results
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get a sample from validation set
                val_sample = next(iter(val_loader))
                val_image = val_sample["image"].to(device)
                val_label = val_sample["label"]
                
                # Get prediction
                val_output = model(val_image)
                val_probs = torch.softmax(val_output, dim=1)
                val_pred = torch.argmax(val_probs, dim=1).detach().cpu()
                
                # Visualize
                fig = visualize_slice(
                    val_image[0].detach().cpu(),
                    val_label[0],
                    val_pred[0],
                    axis=0
                )
                plt.savefig(f"segmentation_epoch_{epoch+1}.png")
                plt.close(fig)
    
    print("Training completed!")


if __name__ == "__main__":
    main()