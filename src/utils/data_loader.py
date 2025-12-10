"""
Data Loading Utilities

Provides functions for loading datasets and distributing data to FL clients.
Supports MNIST for baseline testing and CUB-200 for fine-grained classification.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
import numpy as np
import os


def get_mnist_transforms() -> transforms.Compose:
    """Get standard MNIST transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])


def load_mnist(data_dir: str = "./data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Load the MNIST dataset.
    
    Args:
        data_dir: Directory to store/load the dataset
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    transform = get_mnist_transforms()
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def get_client_data(
    dataset: datasets.MNIST,
    client_id: int,
    num_clients: int,
    partition: str = "iid",
    alpha: float = 0.5
) -> Subset:
    """
    Get data partition for a specific client.
    
    Args:
        dataset: The full training dataset
        client_id: ID of the client (0 to num_clients-1)
        num_clients: Total number of clients
        partition: Partition strategy ("iid" or "noniid")
        alpha: Dirichlet concentration parameter for non-IID (lower = more heterogeneous)
        
    Returns:
        Subset of the dataset for this client
    """
    num_samples = len(dataset)
    
    if partition == "iid":
        # IID: Random equal split
        indices = list(range(num_samples))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        samples_per_client = num_samples // num_clients
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        
    elif partition == "noniid":
        # Non-IID: Dirichlet distribution
        client_indices = _dirichlet_partition(dataset, client_id, num_clients, alpha)
        
    else:
        raise ValueError(f"Unknown partition strategy: {partition}")
    
    return Subset(dataset, client_indices)


def _dirichlet_partition(
    dataset: datasets.MNIST,
    client_id: int,
    num_clients: int,
    alpha: float
) -> List[int]:
    """
    Partition data using Dirichlet distribution for non-IID split.
    
    Args:
        dataset: The full dataset
        client_id: Client to get data for
        num_clients: Total number of clients
        alpha: Concentration parameter
        
    Returns:
        List of indices for this client
    """
    np.random.seed(42)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    
    # Generate Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split indices according to proportions
        idx_batch = np.split(idx_k, proportions)
        
        for client_idx, idx in enumerate(idx_batch):
            client_indices[client_idx].extend(idx.tolist())
    
    return client_indices[client_id]


def create_data_loaders(
    train_subset: Subset,
    test_dataset: datasets.MNIST,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        train_subset: Training data subset for a client
        test_dataset: Full test dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_class_distribution(subset: Subset) -> dict:
    """
    Get the class distribution of a data subset.
    
    Args:
        subset: Data subset to analyze
        
    Returns:
        Dictionary mapping class labels to counts
    """
    labels = []
    for idx in subset.indices:
        _, label = subset.dataset[idx]
        labels.append(label)
    
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


if __name__ == "__main__":
    # Test data loading
    print("Loading MNIST dataset...")
    train_data, test_data = load_mnist("./data")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Test IID partitioning
    print("\nTesting IID partitioning with 5 clients...")
    for client_id in range(5):
        client_data = get_client_data(train_data, client_id, num_clients=5, partition="iid")
        dist = get_class_distribution(client_data)
        print(f"Client {client_id}: {len(client_data)} samples, distribution: {dist}")
    
    # Test non-IID partitioning
    print("\nTesting non-IID partitioning with 5 clients (alpha=0.5)...")
    for client_id in range(5):
        client_data = get_client_data(train_data, client_id, num_clients=5, partition="noniid", alpha=0.5)
        dist = get_class_distribution(client_data)
        print(f"Client {client_id}: {len(client_data)} samples, distribution: {dist}")
