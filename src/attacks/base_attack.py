"""
Base Attack Class

Abstract base class for implementing FL attacks.
All attack implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
import torch


class BaseAttack(ABC):
    """
    Abstract base class for FL attacks.
    
    Subclasses must implement:
    - poison_data(): For data poisoning attacks
    - poison_update(): For model poisoning attacks
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        Initialize the attack.
        
        Args:
            attack_config: Attack configuration dictionary
        """
        self.config = attack_config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Poison the training data.
        
        Used for data poisoning and backdoor attacks.
        
        Args:
            dataset: Clean training dataset
            
        Returns:
            Poisoned dataset
        """
        pass
    
    @abstractmethod
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """
        Poison the model update before sending to server.
        
        Used for model poisoning attacks.
        
        Args:
            local_update: Client's model update
            global_model: Current global model parameters
            num_clients: Total number of clients
            
        Returns:
            Poisoned update
        """
        pass
    
    def is_data_poisoning(self) -> bool:
        """Check if this attack involves data poisoning."""
        return True
    
    def is_model_poisoning(self) -> bool:
        """Check if this attack involves model poisoning."""
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get attack-specific metrics for logging."""
        return {}
    
    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class NoAttack(BaseAttack):
    """Null attack - does nothing (for clean baseline)."""
    
    def __init__(self, attack_config: Dict[str, Any] = None):
        super().__init__(attack_config or {})
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        return dataset
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        return local_update
    
    def is_data_poisoning(self) -> bool:
        return False
    
    def is_model_poisoning(self) -> bool:
        return False
