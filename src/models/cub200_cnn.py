"""
CUB-200 Classification Model

CNN model for CUB-200 bird classification (200 classes).
Uses ResNet-50 backbone pretrained on ImageNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class CUB200CNN(nn.Module):
    """
    CNN for CUB-200 classification.
    
    Uses pretrained ResNet-50 backbone for transfer learning.
    All layers trainable by default for FL (better convergence).
    """
    
    def __init__(self, num_classes: int = 200, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pretrained ResNet-50 (much better for fine-grained classification)
        from torchvision import models
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Optionally freeze backbone (not recommended for FL)
        if freeze_backbone and pretrained:
            for name, param in self.resnet.named_parameters():
                if 'fc' not in name:  # Keep fc layer trainable
                    param.requires_grad = False
        
        # Replace final FC layer with simpler classifier
        in_features = self.resnet.fc.in_features  # 2048 for ResNet-50
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
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
