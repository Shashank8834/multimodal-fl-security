"""
Exploratory Data Analysis (EDA) for FL Security Datasets

Analyzes data distributions, class balance, and partition characteristics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def analyze_dataset_statistics(dataset, name: str = "Dataset", compute_image_stats: bool = True) -> Dict:
    """Compute comprehensive statistics for a dataset including image mean/std."""
    import torch
    n = len(dataset)
    
    # Get labels
    labels = []
    for i in range(min(n, 1000)):  # Sample for large datasets
        item = dataset[i]
        if len(item) >= 2:
            labels.append(item[-1] if isinstance(item[-1], int) else item[1])
    
    label_counts = Counter(labels)
    num_classes = len(label_counts)
    
    # Sample shapes
    sample = dataset[0]
    if hasattr(sample[0], 'shape'):
        data_shape = tuple(sample[0].shape)
    else:
        data_shape = "N/A"
    
    # Compute image statistics (mean, std per channel)
    image_mean = None
    image_std = None
    if compute_image_stats and hasattr(sample[0], 'shape'):
        # Sample 500 images for statistics
        sample_size = min(500, n)
        images = []
        for i in range(sample_size):
            img = dataset[i][0]
            if isinstance(img, torch.Tensor):
                images.append(img)
        
        if images:
            stacked = torch.stack(images)
            # Compute per-channel statistics
            image_mean = stacked.mean(dim=(0, 2, 3)).tolist()
            image_std = stacked.std(dim=(0, 2, 3)).tolist()
    
    # Class frequency statistics
    counts = list(label_counts.values())
    
    stats = {
        'name': name,
        'num_samples': n,
        'num_classes': num_classes,
        'data_shape': data_shape,
        'class_distribution': dict(label_counts),
        'balance_ratio': min(counts) / max(counts) if counts else 0,
        'class_mean': np.mean(counts) if counts else 0,
        'class_std': np.std(counts) if counts else 0,
        'image_mean': image_mean,
        'image_std': image_std
    }
    
    return stats


def compute_heterogeneity_metrics(partition_stats: Dict) -> Dict:
    """
    Compute metrics to quantify data heterogeneity across clients.
    
    Returns:
        - earth_mover_distance: Average EMD between client and global distribution
        - label_distribution_variance: Variance in label proportions
        - class_coverage: Average fraction of classes each client has
    """
    client_stats = partition_stats['client_stats']
    num_clients = len(client_stats)
    
    # Get all classes
    all_classes = set()
    for cs in client_stats:
        all_classes.update(cs['class_distribution'].keys())
    classes = sorted(all_classes)
    num_classes = len(classes)
    
    # Build distribution matrix
    distributions = []
    for cs in client_stats:
        total = sum(cs['class_distribution'].values())
        dist = [cs['class_distribution'].get(c, 0) / total if total > 0 else 0 for c in classes]
        distributions.append(dist)
    
    distributions = np.array(distributions)
    
    # Global distribution
    global_dist = distributions.mean(axis=0)
    
    # Earth Mover Distance approximation (L1 distance)
    emd_values = []
    for dist in distributions:
        emd = np.abs(dist - global_dist).sum() / 2  # Normalized
        emd_values.append(emd)
    
    # Label distribution variance
    label_variance = distributions.var(axis=0).mean()
    
    # Class coverage (fraction of classes each client has)
    coverage = []
    for cs in client_stats:
        coverage.append(cs['num_classes'] / num_classes if num_classes > 0 else 0)
    
    return {
        'avg_emd': np.mean(emd_values),
        'max_emd': np.max(emd_values),
        'label_variance': label_variance,
        'avg_class_coverage': np.mean(coverage),
        'min_class_coverage': np.min(coverage),
        'heterogeneity_score': np.mean(emd_values) * (1 - np.mean(coverage))  # Combined metric
    }



def analyze_client_partitions(
    train_data,
    num_clients: int,
    partition_type: str,
    get_client_data_fn
) -> Dict:
    """Analyze data distribution across client partitions."""
    client_stats = []
    
    for client_id in range(num_clients):
        client_data = get_client_data_fn(train_data, client_id, num_clients, partition_type)
        n = len(client_data)
        
        # Get labels for this client
        labels = []
        for i in range(min(n, 500)):
            item = client_data[i]
            labels.append(item[1] if len(item) >= 2 else item[-1])
        
        label_counts = Counter(labels)
        
        client_stats.append({
            'client_id': client_id,
            'num_samples': n,
            'num_classes': len(label_counts),
            'class_distribution': dict(label_counts)
        })
    
    return {
        'partition_type': partition_type,
        'num_clients': num_clients,
        'client_stats': client_stats,
        'total_samples': sum(c['num_samples'] for c in client_stats)
    }


def plot_class_distribution(stats: Dict, save_path: Optional[str] = None):
    """Plot class distribution bar chart."""
    plt.figure(figsize=(12, 5))
    
    classes = sorted(stats['class_distribution'].keys())
    counts = [stats['class_distribution'][c] for c in classes]
    
    plt.bar(classes, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f"{stats['name']} - Class Distribution", fontsize=14)
    plt.xticks(classes)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_client_distributions(partition_stats: Dict, save_path: Optional[str] = None):
    """Plot class distribution heatmap across clients."""
    client_stats = partition_stats['client_stats']
    num_clients = len(client_stats)
    
    # Get all classes
    all_classes = set()
    for cs in client_stats:
        all_classes.update(cs['class_distribution'].keys())
    classes = sorted(all_classes)
    num_classes = len(classes)
    
    # Build matrix
    matrix = np.zeros((num_clients, num_classes))
    for i, cs in enumerate(client_stats):
        for j, c in enumerate(classes):
            matrix[i, j] = cs['class_distribution'].get(c, 0)
    
    # Normalize per client
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums
    
    plt.figure(figsize=(14, 6))
    plt.imshow(matrix_norm, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Proportion')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Client', fontsize=12)
    plt.title(f"Client Data Distribution ({partition_stats['partition_type']})", fontsize=14)
    plt.xticks(range(num_classes), classes, fontsize=8)
    plt.yticks(range(num_clients), [f"Client {i}" for i in range(num_clients)])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_samples_per_client(partition_stats: Dict, save_path: Optional[str] = None):
    """Plot bar chart of samples per client."""
    client_stats = partition_stats['client_stats']
    
    client_ids = [cs['client_id'] for cs in client_stats]
    sample_counts = [cs['num_samples'] for cs in client_stats]
    
    plt.figure(figsize=(10, 5))
    plt.bar(client_ids, sample_counts, color='coral', edgecolor='black')
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f"Samples per Client ({partition_stats['partition_type']})", fontsize=14)
    plt.xticks(client_ids)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def generate_eda_report(
    dataset_name: str,
    train_data,
    test_data,
    get_client_data_fn,
    num_clients: int = 5,
    output_dir: str = "./experiments/eda"
):
    """Generate complete EDA report for a dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EDA Report: {dataset_name}")
    print('='*60)
    
    # Dataset statistics
    train_stats = analyze_dataset_statistics(train_data, f"{dataset_name} Train")
    test_stats = analyze_dataset_statistics(test_data, f"{dataset_name} Test")
    
    print(f"\nTrain: {train_stats['num_samples']} samples, {train_stats['num_classes']} classes")
    print(f"Test: {test_stats['num_samples']} samples")
    print(f"Data shape: {train_stats['data_shape']}")
    print(f"Balance ratio: {train_stats['balance_ratio']:.3f}")
    
    # Plot class distribution
    plot_class_distribution(
        train_stats, 
        os.path.join(output_dir, f"{dataset_name}_class_dist.png")
    )
    
    # Partition analysis
    for partition in ['iid', 'dirichlet']:
        print(f"\n--- {partition.upper()} Partition ---")
        
        partition_stats = analyze_client_partitions(
            train_data, num_clients, partition, get_client_data_fn
        )
        
        for cs in partition_stats['client_stats']:
            print(f"  Client {cs['client_id']}: {cs['num_samples']} samples, "
                  f"{cs['num_classes']} classes")
        
        # Plot distributions
        plot_client_distributions(
            partition_stats,
            os.path.join(output_dir, f"{dataset_name}_{partition}_heatmap.png")
        )
        plot_samples_per_client(
            partition_stats,
            os.path.join(output_dir, f"{dataset_name}_{partition}_samples.png")
        )
    
    print(f"\nEDA plots saved to: {output_dir}")
    return train_stats, test_stats


if __name__ == "__main__":
    print("Running EDA for FL Security Datasets...")
    
    # MNIST
    try:
        from src.utils.data_loader import load_mnist, get_client_data
        
        train_data, test_data = load_mnist("./data")
        generate_eda_report(
            "MNIST", train_data, test_data, get_client_data,
            num_clients=5, output_dir="./experiments/eda"
        )
    except Exception as e:
        print(f"MNIST EDA failed: {e}")
    
    # CUB-200
    try:
        from src.utils.cub200_loader import load_cub200, get_cub200_client_data
        
        train_data, test_data = load_cub200("./data")
        generate_eda_report(
            "CUB200", train_data, test_data, get_cub200_client_data,
            num_clients=5, output_dir="./experiments/eda"
        )
    except Exception as e:
        print(f"CUB-200 EDA failed: {e}")
    
    print("\nEDA complete!")
