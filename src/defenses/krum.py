"""
Krum Aggregation Defense

Implements Krum and Multi-Krum robust aggregation to defend against
Byzantine and model poisoning attacks.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List, Tuple, Optional
import torch
import numpy as np


class KrumDefense(BaseDefense):
    """
    Krum Robust Aggregation.
    
    Selects the client update that is closest to the majority of other updates.
    This helps defend against Byzantine and model poisoning attacks by
    excluding outlier updates.
    
    Algorithm:
        1. For each client update, compute pairwise distances to all others
        2. For each update, sum distances to its n-f-2 closest neighbors
        3. Select the update with the minimum sum (most "central" update)
        4. (Multi-Krum) Repeat to select m updates and average them
    
    Reference:
        Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant
        Gradient Descent" (NeurIPS 2017)
    
    Parameters:
        num_malicious: Assumed number of malicious clients (f)
        multi_k: Number of clients to select (1 = single Krum, >1 = Multi-Krum)
    
    Example:
        defense = KrumDefense({
            'num_malicious': 1,
            'multi_k': 1
        })
        aggregated = defense.aggregate(client_updates, num_examples)
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.num_malicious = defense_config.get('num_malicious', 1)
        self.multi_k = defense_config.get('multi_k', 1)  # 1 = single Krum
        
        # Track which clients were selected/rejected
        self.selected_clients = []
        self.rejected_clients = []
        self.client_scores = []
    
    def _flatten_update(self, update: List[torch.Tensor]) -> torch.Tensor:
        """Flatten a list of tensors into a single vector."""
        return torch.cat([t.flatten().float() for t in update])
    
    def _unflatten_update(
        self,
        flat: torch.Tensor,
        shapes: List[torch.Size]
    ) -> List[torch.Tensor]:
        """Unflatten a vector back into list of tensors."""
        result = []
        offset = 0
        for shape in shapes:
            numel = np.prod(shape)
            result.append(flat[offset:offset+numel].reshape(shape))
            offset += numel
        return result
    
    def _compute_distances(
        self,
        client_updates: List[List[torch.Tensor]]
    ) -> np.ndarray:
        """
        Compute pairwise distances between client updates.
        
        Args:
            client_updates: List of client model updates
            
        Returns:
            Distance matrix of shape (n_clients, n_clients)
        """
        n = len(client_updates)
        
        # Flatten all updates
        flat_updates = [self._flatten_update(u) for u in client_updates]
        
        # Compute pairwise L2 distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_updates[i] - flat_updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _krum_score(
        self,
        distances: np.ndarray,
        client_idx: int,
        num_neighbors: int
    ) -> float:
        """
        Compute Krum score for a client.
        
        The Krum score is the sum of distances to the closest neighbors.
        Lower score = more similar to other clients = more likely benign.
        
        Args:
            distances: Pairwise distance matrix
            client_idx: Index of client to score
            num_neighbors: Number of closest neighbors to consider (n-f-2)
            
        Returns:
            Krum score (sum of distances to closest neighbors)
        """
        # Get distances from this client to all others
        client_distances = distances[client_idx].copy()
        
        # Sort and sum the smallest n-f-2 distances
        # (excluding distance to self which is 0)
        sorted_distances = np.sort(client_distances)
        
        # Sum distances to num_neighbors closest (excluding self at index 0)
        score = np.sum(sorted_distances[1:num_neighbors + 1])
        
        return score
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using Krum selection.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client (not used in Krum)
            
        Returns:
            Aggregated model update (selected client's update for single Krum,
            or average of selected updates for Multi-Krum)
        """
        n = len(client_updates)
        f = self.num_malicious
        
        # Validate parameters
        if n < 2 * f + 3:
            raise ValueError(
                f"Krum requires n >= 2f + 3. Got n={n}, f={f}. "
                f"Need at least {2*f + 3} clients."
            )
        
        # Number of neighbors to consider
        num_neighbors = n - f - 2
        
        # Compute pairwise distances
        distances = self._compute_distances(client_updates)
        
        # Compute Krum score for each client
        scores = []
        for i in range(n):
            score = self._krum_score(distances, i, num_neighbors)
            scores.append(score)
        
        self.client_scores = scores
        
        # Select clients with lowest scores
        sorted_indices = np.argsort(scores)
        self.selected_clients = sorted_indices[:self.multi_k].tolist()
        self.rejected_clients = sorted_indices[self.multi_k:].tolist()
        
        if self.multi_k == 1:
            # Single Krum: return the update with lowest score
            selected_idx = self.selected_clients[0]
            return client_updates[selected_idx]
        else:
            # Multi-Krum: average the top-k updates
            selected_updates = [client_updates[i] for i in self.selected_clients]
            
            # Average the selected updates
            aggregated = []
            for param_idx in range(len(selected_updates[0])):
                param_sum = sum(u[param_idx] for u in selected_updates)
                aggregated.append(param_sum / self.multi_k)
            
            return aggregated
    
    def detect_malicious(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[int]:
        """
        Detect clients that appear malicious based on Krum scores.
        
        Returns clients with scores significantly higher than selected clients.
        """
        if not self.client_scores:
            # Need to run aggregate first
            self.aggregate(client_updates, num_examples)
        
        # Return clients not selected by Krum
        return self.rejected_clients
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'krum',
            'num_malicious_assumed': self.num_malicious,
            'multi_k': self.multi_k,
            'selected_clients': self.selected_clients,
            'rejected_clients': self.rejected_clients,
            'client_scores': self.client_scores
        }
    
    def __repr__(self) -> str:
        return f"KrumDefense(f={self.num_malicious}, k={self.multi_k})"


class MultiKrumDefense(KrumDefense):
    """
    Multi-Krum Defense.
    
    Convenience class for Multi-Krum with default multi_k > 1.
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        # Default multi_k to half the clients if not specified
        if 'multi_k' not in defense_config:
            defense_config['multi_k'] = defense_config.get('default_k', 3)
        
        super().__init__(defense_config)


if __name__ == "__main__":
    # Test the defense
    import torch
    
    # Create mock client updates
    torch.manual_seed(42)
    
    # 5 benign clients with similar updates
    benign_updates = []
    base_update = [torch.randn(10, 10), torch.randn(10)]
    
    for i in range(5):
        noise = [torch.randn_like(p) * 0.1 for p in base_update]
        update = [base_update[j] + noise[j] for j in range(len(base_update))]
        benign_updates.append(update)
    
    # 2 malicious clients with very different updates
    malicious_updates = []
    for i in range(2):
        # Malicious updates are scaled up significantly
        update = [p * 10 + torch.randn_like(p) * 5 for p in base_update]
        malicious_updates.append(update)
    
    all_updates = benign_updates + malicious_updates
    num_examples = [100] * len(all_updates)
    
    print(f"Total clients: {len(all_updates)} (5 benign, 2 malicious)")
    
    # Test Krum
    krum = KrumDefense({'num_malicious': 2, 'multi_k': 1})
    aggregated = krum.aggregate(all_updates, num_examples)
    
    print(f"\nSingle Krum:")
    print(f"  Client scores: {[f'{s:.2f}' for s in krum.client_scores]}")
    print(f"  Selected: {krum.selected_clients}")
    print(f"  Rejected: {krum.rejected_clients}")
    
    # Test Multi-Krum
    multi_krum = KrumDefense({'num_malicious': 2, 'multi_k': 3})
    aggregated = multi_krum.aggregate(all_updates, num_examples)
    
    print(f"\nMulti-Krum (k=3):")
    print(f"  Selected: {multi_krum.selected_clients}")
    print(f"  Rejected: {multi_krum.rejected_clients}")
    
    # Verify malicious clients are rejected
    malicious_indices = [5, 6]
    print(f"\nMalicious clients (indices {malicious_indices}):")
    print(f"  Detected by Krum: {set(krum.rejected_clients) & set(malicious_indices)}")
