"""
Krum Aggregation Defense

Implements Krum and Multi-Krum robust aggregation.
To be implemented by Siddharth in Week 2.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List
import torch
import numpy as np


class KrumDefense(BaseDefense):
    """
    Krum Robust Aggregation.
    
    Selects the client update that is closest to the majority of other updates.
    This helps defend against Byzantine and model poisoning attacks.
    
    Algorithm:
    1. For each client update, compute sum of distances to n-f-2 closest updates
    2. Select the update with minimum sum of distances
    3. (Multi-Krum) Repeat to select m updates and average them
    
    Reference:
        Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant
        Gradient Descent"
    
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
    
    def _flatten_update(self, update: List[torch.Tensor]) -> torch.Tensor:
        """Flatten a list of tensors into a single vector."""
        return torch.cat([t.flatten() for t in update])
    
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
        # TODO: Implement by Siddharth in Week 2
        #
        # Steps:
        # 1. Flatten each client update
        # 2. Compute pairwise L2 distances
        # 3. Return distance matrix
        
        raise NotImplementedError("To be implemented by Siddharth in Week 2")
    
    def _krum_score(
        self,
        distances: np.ndarray,
        client_idx: int
    ) -> float:
        """
        Compute Krum score for a client.
        
        Args:
            distances: Pairwise distance matrix
            client_idx: Index of client to score
            
        Returns:
            Krum score (sum of distances to closest n-f-2 clients)
        """
        # TODO: Implement by Siddharth in Week 2
        #
        # Steps:
        # 1. Get distances from this client to all others
        # 2. Sort distances
        # 3. Sum the smallest n-f-2 distances
        
        raise NotImplementedError("To be implemented by Siddharth in Week 2")
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using Krum selection.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            
        Returns:
            Aggregated model update (selected client's update)
        """
        # TODO: Implement by Siddharth in Week 2
        #
        # Steps:
        # 1. Compute pairwise distances
        # 2. Compute Krum score for each client
        # 3. If single Krum (multi_k=1): return update with lowest score
        # 4. If Multi-Krum: select top-m updates and average them
        
        raise NotImplementedError("To be implemented by Siddharth in Week 2")
    
    def detect_malicious(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[int]:
        """Detect clients that were not selected by Krum."""
        # After aggregate(), self.rejected_clients contains non-selected clients
        return self.rejected_clients
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'num_malicious_assumed': self.num_malicious,
            'multi_k': self.multi_k,
            'selected_clients': self.selected_clients,
            'rejected_clients': self.rejected_clients
        }
