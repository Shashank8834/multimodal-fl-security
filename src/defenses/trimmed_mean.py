"""
Trimmed Mean and Median Aggregation Defenses

Implements coordinate-wise Trimmed Mean and Median robust aggregation
to defend against Byzantine attacks.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List, Optional
import torch
import numpy as np


class TrimmedMeanDefense(BaseDefense):
    """
    Trimmed Mean Robust Aggregation.
    
    For each parameter coordinate, removes the extreme values and averages
    the remaining values. This helps defend against outlier updates from
    malicious clients.
    
    Algorithm:
        1. For each parameter coordinate across all clients
        2. Sort the values from all clients
        3. Remove the top and bottom trim_ratio% of values
        4. Average the remaining values
    
    Reference:
        Yin et al. "Byzantine-Robust Distributed Learning: Towards Optimal
        Statistical Rates" (ICML 2018)
    
    Parameters:
        trim_ratio: Fraction of values to trim from each end (default 0.1)
    
    Example:
        defense = TrimmedMeanDefense({'trim_ratio': 0.1})
        aggregated = defense.aggregate(client_updates, num_examples)
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.trim_ratio = defense_config.get('trim_ratio', 0.1)
        
        # Track trimmed values count
        self.num_trimmed_per_end = 0
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using coordinate-wise trimmed mean.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client (not used)
            
        Returns:
            Aggregated model update (coordinate-wise trimmed mean)
        """
        n = len(client_updates)
        
        # Number of values to trim from each end
        self.num_trimmed_per_end = max(1, int(n * self.trim_ratio))
        
        # Ensure we have enough clients after trimming
        remaining = n - 2 * self.num_trimmed_per_end
        if remaining < 1:
            # Fall back to median if too few clients
            return self._coordinate_wise_median(client_updates)
        
        aggregated = []
        
        # Process each parameter tensor
        for param_idx in range(len(client_updates[0])):
            # Stack all client values for this parameter
            stacked = torch.stack([u[param_idx].float() for u in client_updates])
            
            # Sort along client dimension
            sorted_vals, _ = torch.sort(stacked, dim=0)
            
            # Trim top and bottom
            trimmed = sorted_vals[self.num_trimmed_per_end:n - self.num_trimmed_per_end]
            
            # Average the remaining values
            aggregated.append(trimmed.mean(dim=0))
        
        return aggregated
    
    def _coordinate_wise_median(
        self,
        client_updates: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Fallback to median when trim_ratio is too high."""
        aggregated = []
        
        for param_idx in range(len(client_updates[0])):
            stacked = torch.stack([u[param_idx].float() for u in client_updates])
            aggregated.append(torch.median(stacked, dim=0)[0])
        
        return aggregated
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'trimmed_mean',
            'trim_ratio': self.trim_ratio,
            'num_trimmed_per_end': self.num_trimmed_per_end
        }
    
    def __repr__(self) -> str:
        return f"TrimmedMeanDefense(trim_ratio={self.trim_ratio})"


class MedianDefense(BaseDefense):
    """
    Coordinate-wise Median Aggregation.
    
    For each parameter coordinate, takes the median value across all clients.
    More robust than trimmed mean but may be less efficient for large models.
    
    Algorithm:
        1. For each parameter coordinate
        2. Collect values from all clients
        3. Take the median value
    
    Properties:
        - Resistant to up to 50% malicious clients
        - No hyperparameters to tune
        - May be slower for large models
    
    Example:
        defense = MedianDefense({})
        aggregated = defense.aggregate(client_updates, num_examples)
    """
    
    def __init__(self, defense_config: Dict[str, Any] = None):
        super().__init__(defense_config or {})
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using coordinate-wise median.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client (not used)
            
        Returns:
            Aggregated model update (coordinate-wise median)
        """
        aggregated = []
        
        for param_idx in range(len(client_updates[0])):
            # Stack all client values for this parameter
            stacked = torch.stack([u[param_idx].float() for u in client_updates])
            
            # Take coordinate-wise median
            median_val = torch.median(stacked, dim=0)[0]
            aggregated.append(median_val)
        
        return aggregated
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'median'
        }
    
    def __repr__(self) -> str:
        return "MedianDefense()"


class GeometricMedianDefense(BaseDefense):
    """
    Geometric Median Aggregation.
    
    Computes the geometric median (point minimizing sum of distances to all others)
    using the Weiszfeld algorithm. More robust than coordinate-wise median.
    
    Algorithm (Weiszfeld):
        1. Initialize with coordinate-wise median
        2. Iteratively update: x = sum(u_i / ||x - u_i||) / sum(1 / ||x - u_i||)
        3. Converge when change is below threshold
    
    Reference:
        Pillutla et al. "Robust Aggregation for Federated Learning" (2019)
    """
    
    def __init__(self, defense_config: Dict[str, Any] = None):
        super().__init__(defense_config or {})
        
        self.max_iters = defense_config.get('max_iters', 100) if defense_config else 100
        self.tolerance = defense_config.get('tolerance', 1e-5) if defense_config else 1e-5
        self.num_iters = 0
    
    def _flatten_updates(
        self,
        client_updates: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Flatten all updates into a 2D tensor (n_clients, n_params)."""
        flat_updates = []
        for update in client_updates:
            flat = torch.cat([p.flatten().float() for p in update])
            flat_updates.append(flat)
        return torch.stack(flat_updates)
    
    def _unflatten(
        self,
        flat: torch.Tensor,
        shapes: List[torch.Size]
    ) -> List[torch.Tensor]:
        """Unflatten back to list of tensors."""
        result = []
        offset = 0
        for shape in shapes:
            numel = int(np.prod(shape))
            result.append(flat[offset:offset+numel].reshape(shape))
            offset += numel
        return result
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using geometric median (Weiszfeld algorithm).
        """
        shapes = [p.shape for p in client_updates[0]]
        
        # Flatten updates
        flat_updates = self._flatten_updates(client_updates)
        n_clients = flat_updates.shape[0]
        
        # Initialize with coordinate-wise median
        current = torch.median(flat_updates, dim=0)[0]
        
        # Weiszfeld iterations
        for iteration in range(self.max_iters):
            # Compute distances to current estimate
            distances = torch.norm(flat_updates - current, dim=1)
            
            # Avoid division by zero
            distances = torch.clamp(distances, min=1e-10)
            
            # Weighted sum
            weights = 1.0 / distances
            new_estimate = (weights.unsqueeze(1) * flat_updates).sum(dim=0) / weights.sum()
            
            # Check convergence
            change = torch.norm(new_estimate - current)
            current = new_estimate
            
            if change < self.tolerance:
                self.num_iters = iteration + 1
                break
        else:
            self.num_iters = self.max_iters
        
        # Unflatten result
        return self._unflatten(current, shapes)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'geometric_median',
            'max_iters': self.max_iters,
            'num_iters': self.num_iters
        }


if __name__ == "__main__":
    # Test the defenses
    import torch
    
    torch.manual_seed(42)
    
    # Create mock client updates
    # 5 benign clients with similar updates
    benign_updates = []
    base_update = [torch.randn(10, 10), torch.randn(10)]
    
    for i in range(5):
        noise = [torch.randn_like(p) * 0.1 for p in base_update]
        update = [base_update[j] + noise[j] for j in range(len(base_update))]
        benign_updates.append(update)
    
    # 2 malicious clients with extreme updates
    malicious_updates = []
    for i in range(2):
        update = [p * 100 for p in base_update]  # Extreme scaling
        malicious_updates.append(update)
    
    all_updates = benign_updates + malicious_updates
    num_examples = [100] * len(all_updates)
    
    print(f"Total clients: {len(all_updates)} (5 benign, 2 malicious)")
    
    # Test Trimmed Mean
    trimmed_mean = TrimmedMeanDefense({'trim_ratio': 0.2})
    tm_result = trimmed_mean.aggregate(all_updates, num_examples)
    print(f"\nTrimmed Mean: {trimmed_mean}")
    print(f"  Trimmed per end: {trimmed_mean.num_trimmed_per_end}")
    
    # Test Median
    median = MedianDefense({})
    med_result = median.aggregate(all_updates, num_examples)
    print(f"\nMedian: {median}")
    
    # Test Geometric Median
    geo_median = GeometricMedianDefense({'max_iters': 50})
    geo_result = geo_median.aggregate(all_updates, num_examples)
    print(f"\nGeometric Median: converged in {geo_median.num_iters} iterations")
    
    # Compare results to benign average
    benign_avg = []
    for param_idx in range(len(benign_updates[0])):
        avg = sum(u[param_idx] for u in benign_updates) / len(benign_updates)
        benign_avg.append(avg)
    
    # Compute distances from each result to benign average
    def compute_distance(result, target):
        flat_result = torch.cat([p.flatten() for p in result])
        flat_target = torch.cat([p.flatten() for p in target])
        return torch.norm(flat_result - flat_target).item()
    
    # Also compute FedAvg result for comparison
    fedavg = []
    for param_idx in range(len(all_updates[0])):
        avg = sum(u[param_idx] for u in all_updates) / len(all_updates)
        fedavg.append(avg)
    
    print(f"\nDistance to benign average:")
    print(f"  FedAvg: {compute_distance(fedavg, benign_avg):.2f}")
    print(f"  Trimmed Mean: {compute_distance(tm_result, benign_avg):.2f}")
    print(f"  Median: {compute_distance(med_result, benign_avg):.2f}")
    print(f"  Geometric Median: {compute_distance(geo_result, benign_avg):.2f}")
