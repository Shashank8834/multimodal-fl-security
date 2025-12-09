"""
Base Defense Class

Abstract base class for implementing FL defenses.
All defense implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch


class BaseDefense(ABC):
    """
    Abstract base class for FL defenses.
    
    Subclasses must implement:
    - aggregate(): Custom aggregation with defense mechanism
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        """
        Initialize the defense.
        
        Args:
            defense_config: Defense configuration dictionary
        """
        self.config = defense_config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate client updates with defense mechanism.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            
        Returns:
            Aggregated model update
        """
        pass
    
    def detect_malicious(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[int]:
        """
        Detect potentially malicious clients.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            
        Returns:
            List of indices of detected malicious clients
        """
        return []  # Default: no detection
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get defense-specific metrics for logging."""
        return {}
    
    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class NoDefense(BaseDefense):
    """Null defense - standard FedAvg (for clean baseline)."""
    
    def __init__(self, defense_config: Dict[str, Any] = None):
        super().__init__(defense_config or {})
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """Standard FedAvg aggregation."""
        total_examples = sum(num_examples)
        
        # Weighted average
        aggregated = []
        for param_idx in range(len(client_updates[0])):
            weighted_sum = sum(
                num_examples[i] * client_updates[i][param_idx]
                for i in range(len(client_updates))
            )
            aggregated.append(weighted_sum / total_examples)
        
        return aggregated
