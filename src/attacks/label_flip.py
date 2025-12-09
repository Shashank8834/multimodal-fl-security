"""
Label Flip Attack

Implements label flipping attack where source class labels are changed to target class.
To be implemented by Dravid in Week 2.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset
from typing import Any, Dict, List
import torch
import numpy as np


class LabelFlipAttack(BaseAttack):
    """
    Label Flipping Attack.
    
    Flips labels from source_class to target_class for a fraction of training data.
    This is a simple but effective data poisoning attack.
    
    Example:
        attack = LabelFlipAttack({
            'source_class': 0,
            'target_class': 8,
            'poison_ratio': 0.1
        })
        poisoned_dataset = attack.poison_data(clean_dataset)
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.source_class = attack_config.get('source_class', 0)
        self.target_class = attack_config.get('target_class', 8)
        self.poison_ratio = attack_config.get('poison_ratio', 0.1)
        
        # Track statistics
        self.num_poisoned = 0
        
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Poison dataset by flipping labels.
        
        Args:
            dataset: Clean training dataset
            
        Returns:
            Dataset with flipped labels
        """
        # TODO: Implement by Dravid in Week 2
        # 
        # Steps:
        # 1. Find all samples with source_class label
        # 2. Randomly select poison_ratio of them
        # 3. Change their labels to target_class
        # 4. Return modified dataset
        #
        # Note: May need to wrap dataset in a custom class to modify labels
        
        raise NotImplementedError("To be implemented by Dravid in Week 2")
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """Label flip is a data poisoning attack, so no model update poisoning."""
        return local_update
    
    def is_data_poisoning(self) -> bool:
        return True
    
    def is_model_poisoning(self) -> bool:
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'source_class': self.source_class,
            'target_class': self.target_class,
            'poison_ratio': self.poison_ratio,
            'num_poisoned': self.num_poisoned
        }
