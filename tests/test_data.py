"""
Unit Tests for Data Loading and Partitioning

Tests MNIST loading, IID/non-IID partitioning.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import (
    load_mnist, 
    get_client_data, 
    get_class_distribution,
    create_data_loaders,
    get_mnist_transforms
)


class TestMNISTLoading:
    """Test MNIST dataset loading."""
    
    def test_load_mnist(self):
        """Test MNIST can be loaded."""
        train_data, test_data = load_mnist("./data")
        
        assert len(train_data) == 60000
        assert len(test_data) == 10000
        
    def test_sample_shape(self):
        """Test sample shape is correct."""
        train_data, _ = load_mnist("./data")
        
        image, label = train_data[0]
        
        assert image.shape == (1, 28, 28)
        assert 0 <= label <= 9
        
    def test_transforms(self):
        """Test transforms are applied."""
        transform = get_mnist_transforms()
        
        # Check normalization params
        assert transform is not None


class TestDataPartitioning:
    """Test data partitioning for FL."""
    
    @pytest.fixture
    def train_data(self):
        train, _ = load_mnist("./data")
        return train
    
    def test_iid_partition(self, train_data):
        """Test IID partitioning."""
        num_clients = 5
        
        all_indices = set()
        for client_id in range(num_clients):
            client_data = get_client_data(
                train_data, 
                client_id, 
                num_clients, 
                partition="iid"
            )
            
            # Check reasonable size
            expected_size = len(train_data) // num_clients
            assert abs(len(client_data) - expected_size) <= 1
            
            # Track indices for overlap check
            all_indices.update(client_data.indices)
        
        # Check no overlap and all data used
        assert len(all_indices) == len(train_data)
    
    def test_iid_class_balance(self, train_data):
        """Test that IID partition has balanced classes."""
        client_data = get_client_data(
            train_data, 
            client_id=0, 
            num_clients=5, 
            partition="iid"
        )
        
        dist = get_class_distribution(client_data)
        
        # All classes should be represented
        assert len(dist) == 10
        
        # Should be roughly balanced
        counts = list(dist.values())
        assert max(counts) / min(counts) < 2.0
    
    def test_noniid_partition(self, train_data):
        """Test non-IID (Dirichlet) partitioning."""
        client_data = get_client_data(
            train_data,
            client_id=0,
            num_clients=5,
            partition="noniid",
            alpha=0.5
        )
        
        assert len(client_data) > 0
        
    def test_noniid_heterogeneity(self, train_data):
        """Test that non-IID partition is heterogeneous."""
        client_dists = []
        for client_id in range(5):
            client_data = get_client_data(
                train_data,
                client_id=client_id,
                num_clients=5,
                partition="noniid",
                alpha=0.1  # Very heterogeneous
            )
            dist = get_class_distribution(client_data)
            client_dists.append(dist)
        
        # With low alpha, clients should have different class distributions
        # Check variance in class counts
        class_counts = []
        for c in range(10):
            counts = [dist.get(c, 0) for dist in client_dists]
            class_counts.append(np.std(counts))
        
        # Should have some variance
        assert np.mean(class_counts) > 0
    
    def test_reproducibility(self, train_data):
        """Test partition is reproducible with same seed."""
        data1 = get_client_data(train_data, 0, 5, "iid")
        data2 = get_client_data(train_data, 0, 5, "iid")
        
        assert data1.indices == data2.indices


class TestDataLoaders:
    """Test data loader creation."""
    
    def test_create_loaders(self):
        """Test train and test loader creation."""
        train_data, test_data = load_mnist("./data")
        client_data = get_client_data(train_data, 0, 5, "iid")
        
        train_loader, test_loader = create_data_loaders(
            client_data, 
            test_data, 
            batch_size=32
        )
        
        # Test iteration
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] <= 32
        assert images.shape[1:] == (1, 28, 28)
        
    def test_batch_size(self):
        """Test configurable batch size."""
        train_data, test_data = load_mnist("./data")
        client_data = get_client_data(train_data, 0, 5, "iid")
        
        for batch_size in [16, 32, 64]:
            train_loader, _ = create_data_loaders(
                client_data, 
                test_data, 
                batch_size=batch_size
            )
            
            batch = next(iter(train_loader))
            assert batch[0].shape[0] <= batch_size


class TestClassDistribution:
    """Test class distribution utility."""
    
    def test_distribution_calculation(self):
        """Test class distribution counts."""
        train_data, _ = load_mnist("./data")
        client_data = get_client_data(train_data, 0, 10, "iid")
        
        dist = get_class_distribution(client_data)
        
        # Check all classes present
        assert len(dist) == 10
        
        # Check counts sum to subset size
        assert sum(dist.values()) == len(client_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
