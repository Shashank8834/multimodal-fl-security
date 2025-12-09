"""
CUB-200 Classification Model

CNN model for CUB-200 bird classification (200 classes).
Uses ResNet-18 backbone pretrained on ImageNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class CUB200CNN(nn.Module):
    """
    CNN for CUB-200 classification.
    
    Uses a smaller custom CNN (not pretrained) for faster FL training.
    For research, this is sufficient to demonstrate attack/defense effectiveness.
    """
    
    def __init__(self, num_classes: int = 200):
        super().__init__()
        
        # Convolutional layers (for 224x224 input)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 -> 28
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28 -> 14
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # 14 -> 4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights as numpy arrays."""
        return [p.data.cpu().numpy() for p in self.parameters()]
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights from numpy arrays."""
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.tensor(weight, dtype=param.dtype, device=param.device)


class CUB200MultimodalCNN(nn.Module):
    """
    Multimodal CNN for CUB-200 with image + attribute fusion.
    
    Processes:
    - Image through CNN backbone
    - Attributes through MLP
    - Fuses both for classification
    """
    
    def __init__(self, num_classes: int = 200, num_attributes: int = 312):
        super().__init__()
        
        # Image branch (simplified CNN)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.image_fc = nn.Linear(128 * 4 * 4, 256)
        
        # Attribute branch
        self.attr_fc = nn.Sequential(
            nn.Linear(num_attributes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(
        self,
        image: torch.Tensor,
        attributes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Image branch
        img_features = self.image_conv(image)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = F.relu(self.image_fc(img_features))
        
        if attributes is not None:
            # Attribute branch
            attr_features = self.attr_fc(attributes)
            # Fusion
            combined = torch.cat([img_features, attr_features], dim=1)
        else:
            # Image only (pad with zeros)
            combined = torch.cat([
                img_features,
                torch.zeros(img_features.size(0), 256, device=img_features.device)
            ], dim=1)
        
        out = self.fusion(combined)
        return out


def create_cub200_model(
    num_classes: int = 200,
    multimodal: bool = False,
    device: str = "cpu"
) -> nn.Module:
    """Create CUB-200 model."""
    if multimodal:
        model = CUB200MultimodalCNN(num_classes=num_classes)
    else:
        model = CUB200CNN(num_classes=num_classes)
    
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("Testing CUB-200 CNN...")
    
    model = CUB200CNN(num_classes=200)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multimodal
    print("\nTesting Multimodal CNN...")
    mm_model = CUB200MultimodalCNN()
    attrs = torch.randn(2, 312)
    out = mm_model(x, attrs)
    print(f"Multimodal output shape: {out.shape}")
