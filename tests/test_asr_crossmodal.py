"""
Unit Tests for ASR Metrics and Cross-Modal Attacks

Tests the new ASR tracking features and cross-modal attack implementations.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.metrics import (
    compute_label_flip_asr,
    compute_model_poisoning_metrics,
    compute_param_divergence,
    AttackMetricsTracker,
    evaluate_model
)
from src.attacks.cross_modal import (
    AttributePoisoningAttack,
    DualModalTriggerAttack,
    AttributePoisonedDataset
)
from src.models.simple_cnn import SimpleCNN


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, num_samples=100, num_classes=10):
        self.data = torch.randn(num_samples, 1, 28, 28)
        self.labels = torch.arange(num_samples) % num_classes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()


class MockMultimodalDataset(Dataset):
    """Mock multimodal dataset (image + attributes) for testing."""
    
    def __init__(self, num_samples=100, num_classes=10, num_attributes=312):
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.attributes = torch.rand(num_samples, num_attributes)
        self.labels = torch.arange(num_samples) % num_classes
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.attributes[idx], self.labels[idx].item()


class TestLabelFlipASR:
    """Test ASR tracking for label flip attacks."""
    
    def test_compute_label_flip_asr(self):
        """Test label flip ASR computation."""
        # Create a mock model that always predicts class 8 for source class 0
        class MockModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Predict class 8 with high confidence
                logits = torch.zeros(batch_size, 10)
                logits[:, 8] = 10.0  # High logit for class 8
                return logits
        
        model = MockModel()
        
        # Create dataset with known class distribution
        dataset = MockDataset(num_samples=100, num_classes=10)
        loader = DataLoader(dataset, batch_size=32)
        
        metrics = compute_label_flip_asr(
            model=model,
            data_loader=loader,
            source_class=0,
            target_class=8,
            device="cpu"
        )
        
        assert "source_accuracy" in metrics
        assert "flip_rate" in metrics
        assert 0 <= metrics["flip_rate"] <= 1
        # Since model predicts 8, flip rate for 0->8 should be high
        assert metrics["flip_rate"] > 0.8
    
    def test_flip_rate_calculation(self):
        """Test flip rate is calculated correctly."""
        class MockModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                logits = torch.zeros(batch_size, 10)
                logits[:, 5] = 10.0  # Predict class 5
                return logits
        
        model = MockModel()
        dataset = MockDataset(num_samples=50)
        loader = DataLoader(dataset, batch_size=50)
        
        # Test with target=5 (should have high flip rate)
        metrics = compute_label_flip_asr(model, loader, 0, 5, "cpu")
        assert metrics["flip_rate"] > 0
        
        # Test with different target (should have 0 flip rate)
        metrics2 = compute_label_flip_asr(model, loader, 0, 3, "cpu")
        assert metrics2["flip_rate"] == 0


class TestModelPoisoningMetrics:
    """Test model poisoning metrics."""
    
    def test_compute_param_divergence(self):
        """Test parameter divergence computation."""
        model1 = SimpleCNN()
        model2 = SimpleCNN()
        
        # Same model should have 0 divergence
        model2.load_state_dict(model1.state_dict())
        divergence = compute_param_divergence(model1, model2)
        assert divergence < 1e-6
        
        # Different models should have positive divergence
        model3 = SimpleCNN()
        divergence2 = compute_param_divergence(model1, model3)
        assert divergence2 > 0
    
    def test_compute_model_poisoning_metrics(self):
        """Test full model poisoning metrics."""
        model1 = SimpleCNN()
        model2 = SimpleCNN()
        
        dataset = MockDataset(num_samples=64)
        loader = DataLoader(dataset, batch_size=32)
        
        metrics = compute_model_poisoning_metrics(
            poisoned_model=model1,
            clean_model=model2,
            data_loader=loader,
            device="cpu"
        )
        
        assert "poisoned_accuracy" in metrics
        assert "clean_accuracy" in metrics
        assert "accuracy_drop" in metrics
        assert "disagreement_rate" in metrics
        assert "param_divergence" in metrics


class TestAttackMetricsTracker:
    """Test the unified attack metrics tracker."""
    
    def test_tracker_backdoor(self):
        """Test tracker with backdoor attack."""
        tracker = AttackMetricsTracker(
            attack_type="backdoor",
            target_class=0
        )
        
        model = SimpleCNN()
        dataset = MockDataset(num_samples=32)
        clean_loader = DataLoader(dataset, batch_size=16)
        triggered_loader = DataLoader(dataset, batch_size=16)
        
        metrics = tracker.compute(
            model=model,
            clean_loader=clean_loader,
            poisoned_loader=triggered_loader,
            device="cpu"
        )
        
        assert "main_accuracy" in metrics
        assert "asr" in metrics
        assert metrics["attack_type"] == "backdoor"
    
    def test_tracker_label_flip(self):
        """Test tracker with label flip attack."""
        tracker = AttackMetricsTracker(
            attack_type="label_flip",
            source_class=0,
            target_class=8
        )
        
        model = SimpleCNN()
        dataset = MockDataset(num_samples=32)
        loader = DataLoader(dataset, batch_size=16)
        
        metrics = tracker.compute(model, loader, device="cpu")
        
        assert "asr" in metrics
        assert "source_accuracy" in metrics
        assert metrics["attack_type"] == "label_flip"
    
    def test_tracker_history(self):
        """Test metrics history tracking."""
        tracker = AttackMetricsTracker("backdoor", target_class=0)
        
        model = SimpleCNN()
        dataset = MockDataset(32)
        loader = DataLoader(dataset, batch_size=16)
        
        # Compute multiple rounds
        for _ in range(3):
            tracker.compute(model, loader, loader, "cpu")
        
        history = tracker.get_history()
        assert len(history) == 3
        
        summary = tracker.get_summary()
        assert summary["final_accuracy"] is not None


class TestAttributePoisoningAttack:
    """Test attribute poisoning attack for CUB-200."""
    
    def test_attack_creation(self):
        """Test attack can be created."""
        attack = AttributePoisoningAttack({
            'target_class': 0,
            'poison_ratio': 0.1,
            'trigger_attributes': list(range(50, 60))
        })
        
        assert attack.target_class == 0
        assert len(attack.trigger_attributes) == 10
    
    def test_poison_data(self):
        """Test attribute poisoning."""
        dataset = MockMultimodalDataset(num_samples=100)
        
        attack = AttributePoisoningAttack({
            'target_class': 0,
            'poison_ratio': 0.2,
            'trigger_attributes': [0, 1, 2],
            'seed': 42
        })
        
        poisoned = attack.poison_data(dataset)
        
        assert len(poisoned) == 100
        assert attack.num_poisoned == 20
    
    def test_attribute_trigger_applied(self):
        """Test that attribute trigger is applied correctly."""
        dataset = MockMultimodalDataset(num_samples=10, num_attributes=50)
        
        attack = AttributePoisoningAttack({
            'target_class': 5,
            'poison_ratio': 1.0,  # Poison all
            'trigger_attributes': [0, 1, 2],
            'seed': 42
        })
        
        poisoned = attack.poison_data(dataset)
        
        # Check a poisoned sample
        img, attrs, label = poisoned[0]
        
        # Label should be target
        assert label == 5
        
        # Trigger attributes should be 1.0
        assert attrs[0] == 1.0
        assert attrs[1] == 1.0
        assert attrs[2] == 1.0
    
    def test_dual_trigger(self):
        """Test dual trigger mode (image + attributes)."""
        dataset = MockMultimodalDataset(num_samples=10)
        
        attack = AttributePoisoningAttack({
            'target_class': 0,
            'poison_ratio': 1.0,
            'trigger_attributes': [0, 1],
            'dual_trigger': True,
            'image_trigger_size': 4
        })
        
        poisoned = attack.poison_data(dataset)
        img, attrs, label = poisoned[0]
        
        # Check image trigger applied
        assert img[..., -4:, -4:].mean() > 0.9  # White square


class TestDualModalTriggerAttack:
    """Test dual-modal trigger attack."""
    
    def test_attack_creation(self):
        """Test attack creation."""
        attack = DualModalTriggerAttack({
            'target_class': 0,
            'poison_ratio': 0.1
        })
        
        assert attack.target_class == 0
    
    def test_triggered_test_modes(self):
        """Test different trigger modes for ASR measurement."""
        attack = DualModalTriggerAttack({
            'target_class': 0,
            'image_trigger_size': 4,
            'trigger_attributes': [0, 1, 2]
        })
        
        dataset = MockMultimodalDataset(num_samples=10)
        
        # Test "both" mode
        both_triggered = attack.create_triggered_test_set(dataset, "both")
        assert len(both_triggered) == 10
        
        # Test "image" mode
        img_triggered = attack.create_triggered_test_set(dataset, "image")
        assert len(img_triggered) == 10
        
        # Test "attribute" mode
        attr_triggered = attack.create_triggered_test_set(dataset, "attribute")
        assert len(attr_triggered) == 10
    
    def test_metrics(self):
        """Test attack metrics."""
        attack = DualModalTriggerAttack({
            'target_class': 5,
            'poison_ratio': 0.15
        })
        
        dataset = MockMultimodalDataset(num_samples=100)
        attack.poison_data(dataset)
        
        metrics = attack.get_metrics()
        
        assert metrics["attack_type"] == "dual_modal_trigger"
        assert metrics["target_class"] == 5
        assert metrics["num_poisoned"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
