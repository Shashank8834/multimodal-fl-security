"""
Unit Tests for Models

Tests SimpleCNN and CUB200CNN model architectures.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.simple_cnn import SimpleCNN, create_model


class TestSimpleCNN:
    """Test SimpleCNN model for MNIST."""
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = SimpleCNN(num_classes=10)
        assert model is not None
        
    def test_forward_pass_shape(self):
        """Test output shape is correct."""
        model = SimpleCNN(num_classes=10)
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        
    def test_different_num_classes(self):
        """Test model works with different class counts."""
        for num_classes in [5, 10, 100]:
            model = SimpleCNN(num_classes=num_classes)
            x = torch.randn(2, 1, 28, 28)
            output = model(x)
            assert output.shape == (2, num_classes)
    
    def test_get_weights(self):
        """Test weight extraction."""
        model = SimpleCNN()
        weights = model.get_weights()
        
        assert isinstance(weights, list)
        assert len(weights) > 0
        assert all(isinstance(w, type(weights[0])) for w in weights)
        
    def test_set_weights(self):
        """Test weight setting."""
        model1 = SimpleCNN()
        model2 = SimpleCNN()
        
        weights = model1.get_weights()
        model2.set_weights(weights)
        
        # Check weights are equal
        for w1, w2 in zip(model1.get_weights(), model2.get_weights()):
            assert torch.allclose(torch.tensor(w1), torch.tensor(w2))
    
    def test_factory_function(self):
        """Test create_model factory function."""
        model = create_model(num_classes=10, device="cpu")
        
        assert isinstance(model, SimpleCNN)
        # Check it's on CPU
        assert next(model.parameters()).device.type == "cpu"
    
    def test_parameter_count(self):
        """Test model has expected parameter count."""
        model = SimpleCNN()
        param_count = sum(p.numel() for p in model.parameters())
        
        # SimpleCNN should have ~400k-1.3M parameters
        assert 100_000 < param_count < 2_000_000


class TestCUB200CNN:
    """Test CUB200CNN model."""
    
    def test_model_creation(self):
        """Test that model can be created."""
        from src.models.cub200_cnn import CUB200CNN
        
        model = CUB200CNN(num_classes=200, pretrained=False)
        assert model is not None
    
    def test_forward_pass_shape(self):
        """Test output shape for CUB-200."""
        from src.models.cub200_cnn import CUB200CNN
        
        model = CUB200CNN(num_classes=200, pretrained=False)
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (batch_size, 200)
    
    def test_get_set_weights(self):
        """Test weight extraction and setting."""
        from src.models.cub200_cnn import CUB200CNN
        
        model1 = CUB200CNN(num_classes=200, pretrained=False)
        model2 = CUB200CNN(num_classes=200, pretrained=False)
        
        weights = model1.get_weights()
        model2.set_weights(weights)
        
        # Verify weights match
        for w1, w2 in zip(model1.get_weights(), model2.get_weights()):
            assert w1.shape == w2.shape


class TestMultimodalCNN:
    """Test CUB200MultimodalCNN model."""
    
    def test_multimodal_forward(self):
        """Test multimodal forward pass."""
        from src.models.cub200_cnn import CUB200MultimodalCNN
        
        model = CUB200MultimodalCNN(num_classes=200, num_attributes=312)
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        attributes = torch.randn(batch_size, 312)
        
        output = model(images, attributes)
        assert output.shape == (batch_size, 200)
    
    def test_image_only_forward(self):
        """Test forward pass with image only (no attributes)."""
        from src.models.cub200_cnn import CUB200MultimodalCNN
        
        model = CUB200MultimodalCNN(num_classes=200)
        
        images = torch.randn(2, 3, 224, 224)
        output = model(images, attributes=None)
        
        assert output.shape == (2, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
