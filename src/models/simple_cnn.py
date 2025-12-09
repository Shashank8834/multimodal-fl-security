"""
Simple CNN Model for MNIST Classification

A lightweight CNN for baseline testing in the FL framework.
This model achieves ~98% accuracy on MNIST in centralized training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST.
    
    Architecture:
        - Conv1: 1 -> 32 channels, 3x3 kernel
        - Conv2: 32 -> 64 channels, 3x3 kernel
        - MaxPool: 2x2
        - FC1: 9216 -> 128
        - FC2: 128 -> 10 (num_classes)
    
    Total parameters: ~1.2M
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 conv + 2 pool: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_weights(self) -> list:
        """Get model weights as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def set_weights(self, weights: list) -> None:
        """Set model weights from a list of numpy arrays."""
        state_dict = self.state_dict()
        for (key, _), new_weight in zip(state_dict.items(), weights):
            state_dict[key] = torch.tensor(new_weight)
        self.load_state_dict(state_dict)


def create_model(num_classes: int = 10, device: str = "cpu") -> SimpleCNN:
    """
    Factory function to create a SimpleCNN model.
    
    Args:
        num_classes: Number of output classes
        device: Device to place the model on ("cpu" or "cuda")
        
    Returns:
        Initialized SimpleCNN model
    """
    model = SimpleCNN(num_classes=num_classes)
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
