"""
Data Partitioning Utilities

Functions for partitioning datasets across FL clients with various strategies.
"""

import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, Subset


def partition_data(
    dataset: Dataset,
    num_clients: int,
    strategy: str = "iid",
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partition dataset indices for multiple clients.
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        strategy: Partitioning strategy ("iid", "noniid", "shard")
        alpha: Dirichlet concentration for non-IID
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client_id to list of data indices
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    
    if strategy == "iid":
        return _iid_partition(num_samples, num_clients)
    elif strategy == "noniid":
        return _dirichlet_partition_all(dataset, num_clients, alpha)
    elif strategy == "shard":
        return _shard_partition(dataset, num_clients, shards_per_client=2)
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")


def _iid_partition(num_samples: int, num_clients: int) -> Dict[int, List[int]]:
    """IID random partition."""
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    split_indices = np.array_split(indices, num_clients)
    return {i: split_indices[i].tolist() for i in range(num_clients)}


def _dirichlet_partition_all(
    dataset: Dataset,
    num_clients: int,
    alpha: float
) -> Dict[int, List[int]]:
    """Non-IID partition using Dirichlet distribution."""
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Convert to actual counts
        proportions = proportions / proportions.sum()
        counts = (proportions * len(idx_k)).astype(int)
        counts[-1] = len(idx_k) - counts[:-1].sum()  # Ensure all samples used
        
        # Assign to clients
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(idx_k[start:start + count].tolist())
            start += count
    
    return client_indices


def _shard_partition(
    dataset: Dataset,
    num_clients: int,
    shards_per_client: int = 2
) -> Dict[int, List[int]]:
    """
    Shard-based non-IID partition.
    
    Each client gets data from only a few classes (shards).
    This creates extreme non-IID distribution.
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    num_shards = num_clients * shards_per_client
    
    # Sort by label
    sorted_indices = np.argsort(labels)
    
    # Create shards
    shard_size = len(labels) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    
    # Shuffle and assign shards to clients
    shard_ids = list(range(num_shards))
    np.random.shuffle(shard_ids)
    
    client_indices = {}
    for client_id in range(num_clients):
        assigned_shards = shard_ids[client_id * shards_per_client:(client_id + 1) * shards_per_client]
        client_indices[client_id] = np.concatenate([shards[s] for s in assigned_shards]).tolist()
    
    return client_indices


def analyze_partition(
    partition: Dict[int, List[int]],
    dataset: Dataset
) -> Dict[int, Dict[str, any]]:
    """
    Analyze the partition to understand data distribution.
    
    Args:
        partition: Client-to-indices mapping
        dataset: The original dataset
        
    Returns:
        Analysis results for each client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    analysis = {}
    for client_id, indices in partition.items():
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        
        analysis[client_id] = {
            "num_samples": len(indices),
            "num_classes": len(unique),
            "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
            "majority_class": unique[np.argmax(counts)].item(),
            "majority_ratio": counts.max() / counts.sum()
        }
    
    return analysis


if __name__ == "__main__":
    from torchvision import datasets, transforms
    
    # Load MNIST
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # Test different partitioning strategies
    for strategy in ["iid", "noniid", "shard"]:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy}")
        print("="*50)
        
        kwargs = {"alpha": 0.5} if strategy == "noniid" else {}
        partition = partition_data(mnist, num_clients=5, strategy=strategy, **kwargs)
        analysis = analyze_partition(partition, mnist)
        
        for client_id, stats in analysis.items():
            print(f"\nClient {client_id}:")
            print(f"  Samples: {stats['num_samples']}")
            print(f"  Classes: {stats['num_classes']}")
            print(f"  Majority class: {stats['majority_class']} ({stats['majority_ratio']:.1%})")
