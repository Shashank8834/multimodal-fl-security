"""
Trimmed Mean Aggregation Defense

Implements coordinate-wise Trimmed Mean and Median robust aggregation.
To be implemented by Siddharth in Week 2.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List
import torch
import numpy as np


class TrimmedMeanDefense(BaseDefense):
    """
    Trimmed Mean Robust Aggregation.
    
    For each parameter coordinate, removes the extreme values and averages
    the remaining values. This helps defend against outlier updates.
    
    Algorithm:
    1. For each parameter coordinate across all clients
    2. Remove the top and bottom trim_ratio% of values
    3. Average the remaining values
    
    Reference:
        Yin et al. "Byzantine-Robust Distributed Learning"
    
    Example:
        defense = TrimmedMeanDefense({
            'trim_ratio': 0.1
        })
        aggregated = defense.aggregate(client_updates, num_examples)
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.trim_ratio = defense_config.get('trim_ratio', 0.1)
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate using trimmed mean.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            
        Returns:
            Aggregated model update
        """
        # TODO: Implement by Siddharth in Week 2
        #
        # Steps:
        # 1. Stack all client updates
        # 2. For each parameter coordinate:
        #    a. Sort values from all clients
        #    b. Remove top and bottom trim_ratio%
        #    c. Average remaining values
        # 3. Return aggregated update
        
        raise NotImplementedError("To be implemented by Siddharth in Week 2")
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'trim_ratio': self.trim_ratio
        }


class MedianDefense(BaseDefense):
    """
    Coordinate-wise Median Aggregation.
    
    For each parameter coordinate, takes the median value across all clients.
    More robust than trimmed mean but may be less efficient.
    
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
            num_examples: Number of examples per client
            
        Returns:
            Aggregated model update (coordinate-wise median)
        """
        # TODO: Implement by Siddharth in Week 2
        #
        # Steps:
        # 1. Stack all client updates
        # 2. For each parameter coordinate, take median
        # 3. Return aggregated update
        
        raise NotImplementedError("To be implemented by Siddharth in Week 2")
