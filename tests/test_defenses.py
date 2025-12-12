"""
Unit Tests for Defenses

Tests Krum, FLTrust, DP-SGD, and Trimmed Mean defenses.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.defenses.krum import KrumDefense, MultiKrumDefense
from src.defenses.trimmed_mean import TrimmedMeanDefense, MedianDefense, GeometricMedianDefense
from src.defenses.differential_privacy import DPSGDDefense, GradientClippingDefense


def create_mock_updates(num_clients=5, param_shapes=[(10, 10), (10,)]):
    """Create mock client updates for testing."""
    updates = []
    for _ in range(num_clients):
        update = [torch.randn(shape) for shape in param_shapes]
        updates.append(update)
    return updates


class TestKrumDefense:
    """Test Krum aggregation defense."""
    
    def test_defense_creation(self):
        """Test defense can be created."""
        defense = KrumDefense({
            'num_malicious': 1,
            'multi_k': 1
        })
        assert defense is not None
        
    def test_aggregate_single_krum(self):
        """Test single Krum aggregation."""
        # Need at least 2f + 3 = 5 clients for f=1
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = KrumDefense({'num_malicious': 1, 'multi_k': 1})
        result = defense.aggregate(updates, num_examples)
        
        # Result should be one of the updates
        assert len(result) == len(updates[0])
        assert defense.selected_clients is not None
        assert len(defense.selected_clients) == 1
        
    def test_aggregate_multi_krum(self):
        """Test Multi-Krum aggregation."""
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = KrumDefense({'num_malicious': 1, 'multi_k': 2})
        result = defense.aggregate(updates, num_examples)
        
        assert len(result) == len(updates[0])
        assert len(defense.selected_clients) == 2
        
    def test_malicious_detection(self):
        """Test that malicious updates are rejected."""
        # Create 4 similar updates and 1 outlier
        base = [torch.zeros(10, 10)]
        benign_updates = [[b + torch.randn_like(b) * 0.01] for b in [base[0]] * 4]
        malicious_update = [[torch.randn(10, 10) * 100]]  # Large outlier
        
        updates = benign_updates + malicious_update
        num_examples = [100] * 5
        
        defense = KrumDefense({'num_malicious': 1, 'multi_k': 1})
        defense.aggregate(updates, num_examples)
        
        # Malicious client (index 4) should have highest score
        scores = defense.client_scores
        # The outlier should NOT be selected
        assert 4 not in defense.selected_clients
        
    def test_insufficient_clients_error(self):
        """Test error when not enough clients."""
        updates = create_mock_updates(num_clients=3)
        num_examples = [100] * 3
        
        defense = KrumDefense({'num_malicious': 2})
        
        with pytest.raises(ValueError):
            defense.aggregate(updates, num_examples)


class TestTrimmedMeanDefense:
    """Test Trimmed Mean aggregation."""
    
    def test_defense_creation(self):
        """Test defense can be created."""
        defense = TrimmedMeanDefense({'trim_ratio': 0.1})
        assert defense.trim_ratio == 0.1
        
    def test_aggregate(self):
        """Test trimmed mean aggregation."""
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = TrimmedMeanDefense({'trim_ratio': 0.2})
        result = defense.aggregate(updates, num_examples)
        
        assert len(result) == len(updates[0])
        
    def test_outlier_trimming(self):
        """Test that outliers are trimmed."""
        # Create updates with outliers
        base = torch.zeros(10)
        updates = [
            [base + 0.1],
            [base + 0.2],
            [base],
            [base + 100],  # Positive outlier
            [base - 100],  # Negative outlier
        ]
        num_examples = [100] * 5
        
        defense = TrimmedMeanDefense({'trim_ratio': 0.2})
        result = defense.aggregate(updates, num_examples)
        
        # Result should be close to 0.1 (mean of non-outliers)
        assert result[0].abs().mean() < 1.0


class TestMedianDefense:
    """Test Median aggregation."""
    
    def test_aggregate(self):
        """Test coordinate-wise median."""
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = MedianDefense({})
        result = defense.aggregate(updates, num_examples)
        
        assert len(result) == len(updates[0])
        
    def test_single_outlier_resistance(self):
        """Test resistance to single outlier."""
        updates = [
            [torch.tensor([1.0, 1.0])],
            [torch.tensor([1.1, 0.9])],
            [torch.tensor([0.9, 1.1])],
            [torch.tensor([100.0, 100.0])],  # Outlier
            [torch.tensor([1.0, 1.0])],
        ]
        num_examples = [100] * 5
        
        defense = MedianDefense({})
        result = defense.aggregate(updates, num_examples)
        
        # Median should be close to 1.0
        assert torch.allclose(result[0], torch.tensor([1.0, 1.0]), atol=0.2)


class TestGeometricMedianDefense:
    """Test Geometric Median aggregation."""
    
    def test_aggregate(self):
        """Test geometric median (Weiszfeld)."""
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = GeometricMedianDefense({'max_iters': 50})
        result = defense.aggregate(updates, num_examples)
        
        assert len(result) == len(updates[0])
        assert defense.num_iters > 0


class TestDPSGDDefense:
    """Test DP-SGD defense."""
    
    def test_defense_creation(self):
        """Test defense can be created."""
        defense = DPSGDDefense({
            'clip_norm': 1.0,
            'noise_multiplier': 0.1
        })
        assert defense.clip_norm == 1.0
        
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        defense = DPSGDDefense({'clip_norm': 1.0})
        
        # Create update with large norm
        large_update = [torch.randn(100) * 10]
        clipped, original_norm = defense.clip_gradient(large_update)
        
        clipped_norm = torch.norm(torch.cat([t.flatten() for t in clipped]))
        
        assert original_norm > 1.0
        assert clipped_norm <= 1.0 + 1e-6
        
    def test_noise_addition(self):
        """Test noise is added."""
        defense = DPSGDDefense({
            'clip_norm': 1.0,
            'noise_multiplier': 1.0
        })
        
        update = [torch.zeros(100)]
        noisy = defense.add_noise(update, num_clients=1)
        
        # With high noise, result should be non-zero
        assert noisy[0].abs().sum() > 0
        
    def test_aggregate(self):
        """Test DP aggregation."""
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        defense = DPSGDDefense({
            'clip_norm': 10.0,
            'noise_multiplier': 0.01
        })
        
        result = defense.aggregate(updates, num_examples)
        
        assert len(result) == len(updates[0])
        assert defense.rounds_completed == 1
        
    def test_privacy_accounting(self):
        """Test privacy budget tracking."""
        defense = DPSGDDefense({
            'noise_multiplier': 0.1,
            'target_epsilon': 10.0
        })
        
        updates = create_mock_updates(num_clients=5)
        num_examples = [100] * 5
        
        initial_spent = defense.get_privacy_spent()
        defense.aggregate(updates, num_examples)
        
        assert defense.get_privacy_spent() > initial_spent


class TestGradientClippingDefense:
    """Test gradient clipping (no noise)."""
    
    def test_clipping(self):
        """Test updates are clipped."""
        defense = GradientClippingDefense({'clip_norm': 1.0})
        
        # Create updates with varying norms
        updates = [
            [torch.randn(100) * 10],  # Large norm
            [torch.randn(100) * 0.1],  # Small norm
        ]
        num_examples = [100, 100]
        
        result = defense.aggregate(updates, num_examples)
        
        assert defense.clipped_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
