"""
Unit Tests for Attacks

Tests label flip, backdoor, and model poisoning attacks.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, Subset, TensorDataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attacks.label_flip import LabelFlipAttack, AllToOneAttack, PoisonedDataset
from src.attacks.backdoor import BackdoorAttack, DistributedBackdoorAttack
from src.attacks.model_poisoning import ModelReplacementAttack, ScalingAttack


class MockDataset(Dataset):
    """Simple mock dataset for testing."""
    
    def __init__(self, num_samples=100, num_classes=10):
        self.data = torch.randn(num_samples, 1, 28, 28)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.targets = self.labels  # For compatibility
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()


class TestLabelFlipAttack:
    """Test LabelFlipAttack implementation."""
    
    def test_attack_creation(self):
        """Test attack can be created."""
        attack = LabelFlipAttack({
            'source_class': 0,
            'target_class': 8,
            'poison_ratio': 0.1
        })
        assert attack is not None
        assert attack.source_class == 0
        assert attack.target_class == 8
        
    def test_poison_data(self):
        """Test data poisoning."""
        dataset = MockDataset(num_samples=100, num_classes=10)
        
        attack = LabelFlipAttack({
            'source_class': 0,
            'target_class': 8,
            'poison_ratio': 0.5,
            'seed': 42
        })
        
        poisoned = attack.poison_data(dataset)
        
        assert len(poisoned) == len(dataset)
        assert attack.num_poisoned > 0
        
    def test_is_data_poisoning(self):
        """Test attack classification."""
        attack = LabelFlipAttack({})
        
        assert attack.is_data_poisoning() is True
        assert attack.is_model_poisoning() is False
        
    def test_poison_update_passthrough(self):
        """Test that poison_update returns input unchanged."""
        attack = LabelFlipAttack({})
        
        update = [torch.randn(10, 10)]
        global_model = [torch.randn(10, 10)]
        
        result = attack.poison_update(update, global_model, num_clients=5)
        
        assert torch.equal(result[0], update[0])
        
    def test_metrics(self):
        """Test attack metrics."""
        attack = LabelFlipAttack({
            'source_class': 1,
            'target_class': 9,
            'poison_ratio': 0.2
        })
        
        metrics = attack.get_metrics()
        
        assert metrics['attack_type'] == 'label_flip'
        assert metrics['source_class'] == 1
        assert metrics['target_class'] == 9


class TestAllToOneAttack:
    """Test AllToOneAttack variant."""
    
    def test_poison_all_classes(self):
        """Test that all classes can be poisoned."""
        dataset = MockDataset(num_samples=100)
        
        attack = AllToOneAttack({
            'target_class': 0,
            'poison_ratio': 0.3,
            'seed': 42
        })
        
        poisoned = attack.poison_data(dataset)
        
        assert attack.num_poisoned == 30  # 30% of 100


class TestBackdoorAttack:
    """Test BackdoorAttack implementation."""
    
    def test_attack_creation(self):
        """Test attack creation with different triggers."""
        for trigger_type in ['square', 'cross', 'corner', 'checkerboard']:
            attack = BackdoorAttack({
                'trigger_type': trigger_type,
                'trigger_size': 3,
                'target_class': 0
            })
            assert attack.trigger.shape == (3, 3)
    
    def test_trigger_positions(self):
        """Test different trigger positions."""
        positions = ['bottom_right', 'top_left', 'top_right', 'bottom_left', 'center']
        
        for pos in positions:
            attack = BackdoorAttack({
                'trigger_position': pos,
                'trigger_size': 3,
                'image_size': (28, 28)
            })
            
            row, col = attack.trigger_position
            assert 0 <= row < 28
            assert 0 <= col < 28
    
    def test_apply_trigger(self):
        """Test trigger application to image."""
        attack = BackdoorAttack({
            'trigger_size': 3,
            'trigger_position': 'bottom_right',
            'trigger_value': 1.0
        })
        
        image = torch.zeros(1, 28, 28)
        triggered = attack.apply_trigger(image)
        
        # Check trigger was applied
        assert triggered.max() > 0
        
    def test_poison_data(self):
        """Test data poisoning with backdoor."""
        dataset = MockDataset(num_samples=100)
        
        attack = BackdoorAttack({
            'trigger_size': 3,
            'target_class': 0,
            'poison_ratio': 0.1,
            'seed': 42
        })
        
        poisoned = attack.poison_data(dataset)
        
        assert len(poisoned) == 100
        assert attack.num_poisoned == 10
        
    def test_create_poisoned_testset(self):
        """Test triggered test set creation."""
        dataset = MockDataset(num_samples=50, num_classes=10)
        
        attack = BackdoorAttack({
            'target_class': 0,
            'trigger_size': 3
        })
        
        triggered_test = attack.create_poisoned_testset(dataset)
        
        # Should exclude target class samples
        assert len(triggered_test) <= len(dataset)


class TestModelReplacementAttack:
    """Test ModelReplacementAttack implementation."""
    
    def test_attack_creation(self):
        """Test attack creation."""
        attack = ModelReplacementAttack({
            'scale_factor': 10.0,
            'num_malicious': 1
        })
        
        assert attack.scale_factor == 10.0
        
    def test_poison_update(self):
        """Test update poisoning scales correctly."""
        attack = ModelReplacementAttack({
            'scale_factor': 10.0,
            'num_malicious': 1
        })
        
        global_model = [torch.zeros(10, 10)]
        local_update = [torch.ones(10, 10) * 0.1]
        
        poisoned = attack.poison_update(local_update, global_model, num_clients=5)
        
        # Check scaling was applied
        assert poisoned[0].abs().mean() > local_update[0].abs().mean()
        
    def test_is_model_poisoning(self):
        """Test attack classification."""
        attack = ModelReplacementAttack({})
        
        assert attack.is_data_poisoning() is False
        assert attack.is_model_poisoning() is True


class TestScalingAttack:
    """Test simple ScalingAttack."""
    
    def test_scaling(self):
        """Test gradient scaling."""
        attack = ScalingAttack({'scale': 100.0})
        
        update = [torch.ones(10)]
        result = attack.poison_update(update, update, num_clients=5)
        
        assert torch.allclose(result[0], torch.ones(10) * 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
