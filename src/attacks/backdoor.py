"""
Backdoor Attack

Implements backdoor attack with trigger pattern injection.
To be implemented by Dravid in Week 2.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple
import torch
import numpy as np


class BackdoorAttack(BaseAttack):
    """
    Backdoor Attack with Trigger Pattern.
    
    Injects a trigger pattern (e.g., small white square) into training images
    and changes their labels to the target class. When the model is deployed,
    any input with the trigger will be classified as the target class.
    
    Example:
        attack = BackdoorAttack({
            'trigger_size': 3,
            'trigger_position': 'bottom_right',
            'target_class': 0,
            'poison_ratio': 0.1
        })
        poisoned_dataset = attack.poison_data(clean_dataset)
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.trigger_size = attack_config.get('trigger_size', 3)
        self.trigger_position = attack_config.get('trigger_position', 'bottom_right')
        self.target_class = attack_config.get('target_class', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.1)
        
        # Create trigger pattern
        self.trigger = self._create_trigger()
        
        # Track statistics
        self.num_poisoned = 0
        
    def _create_trigger(self) -> torch.Tensor:
        """
        Create the trigger pattern.
        
        Returns:
            Trigger tensor of shape (trigger_size, trigger_size)
        """
        # Simple white square trigger
        trigger = torch.ones(self.trigger_size, self.trigger_size)
        return trigger
    
    def _get_trigger_position(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get the position to place the trigger.
        
        Args:
            image_size: (height, width) of the image
            
        Returns:
            (row, col) position for trigger top-left corner
        """
        h, w = image_size
        
        if self.trigger_position == 'bottom_right':
            return (h - self.trigger_size, w - self.trigger_size)
        elif self.trigger_position == 'top_left':
            return (0, 0)
        elif self.trigger_position == 'center':
            return ((h - self.trigger_size) // 2, (w - self.trigger_size) // 2)
        else:
            return (h - self.trigger_size, w - self.trigger_size)
    
    def apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply trigger pattern to an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Image with trigger applied
        """
        # TODO: Implement by Dravid in Week 2
        #
        # Steps:
        # 1. Clone the image
        # 2. Get trigger position
        # 3. Apply trigger to the appropriate location
        # 4. Return modified image
        
        raise NotImplementedError("To be implemented by Dravid in Week 2")
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Poison dataset by adding trigger patterns.
        
        Args:
            dataset: Clean training dataset
            
        Returns:
            Dataset with backdoor triggers
        """
        # TODO: Implement by Dravid in Week 2
        #
        # Steps:
        # 1. Randomly select poison_ratio of samples
        # 2. Apply trigger to selected images
        # 3. Change their labels to target_class
        # 4. Return modified dataset
        
        raise NotImplementedError("To be implemented by Dravid in Week 2")
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """Backdoor is a data poisoning attack, so no model update poisoning."""
        return local_update
    
    def create_poisoned_testset(self, clean_testset: Dataset) -> Dataset:
        """
        Create a test set with all samples having the trigger.
        
        Used to measure Attack Success Rate (ASR).
        
        Args:
            clean_testset: Clean test dataset
            
        Returns:
            Test dataset with triggers (for ASR measurement)
        """
        # TODO: Implement by Dravid in Week 2
        raise NotImplementedError("To be implemented by Dravid in Week 2")
    
    def is_data_poisoning(self) -> bool:
        return True
    
    def is_model_poisoning(self) -> bool:
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'trigger_size': self.trigger_size,
            'trigger_position': self.trigger_position,
            'target_class': self.target_class,
            'poison_ratio': self.poison_ratio,
            'num_poisoned': self.num_poisoned
        }
