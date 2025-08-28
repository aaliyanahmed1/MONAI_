"""Neural network models for medical image processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block for UNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        return x


class DownSample(nn.Module):
    """Downsampling block for UNet."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip_connection = self.conv_block(x)
        x = self.pool(skip_connection)
        return x, skip_connection


class UpSample(nn.Module):
    """Upsampling block for UNet."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.up(x)
        
        # Handle different sizes
        diff_h = skip_connection.size()[2] - x.size()[2]
        diff_w = skip_connection.size()[3] - x.size()[3]
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """2D UNet implementation for medical image segmentation."""
    
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Downsampling
        in_features = in_channels
        for feature in features:
            self.downs.append(DownSample(in_features, feature))
            in_features = feature
        
        # Bottom layer
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Upsampling
        for feature in reversed(features):
            self.ups.append(UpSample(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for upsampling
        
        # Upsampling
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i])
        
        return self.final_conv(x)


class UNet3D(nn.Module):
    """3D UNet implementation for volumetric medical image segmentation."""
    
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Downsampling
        in_features = in_channels
        for feature in features:
            self.downs.append(Conv3DBlock(in_features, feature))
            in_features = feature
            
        # Pooling layers
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottom layer
        self.bottleneck = Conv3DBlock(features[-1], features[-1] * 2)
        
        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2),
                    Conv3DBlock(feature * 2, feature)
                )
            )
        
        # Final layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for upsampling
        
        # Upsampling
        for i, up in enumerate(self.ups):
            x = up[0](x)  # ConvTranspose3d
            
            # Handle different sizes
            if x.shape[2:] != skip_connections[i].shape[2:]:
                x = F.interpolate(x, size=skip_connections[i].shape[2:], mode='trilinear', align_corners=True)
                
            x = torch.cat([skip_connections[i], x], dim=1)
            x = up[1](x)  # Conv3DBlock
        
        return self.final_conv(x)


class Conv3DBlock(nn.Module):
    """Basic 3D convolutional block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        return x