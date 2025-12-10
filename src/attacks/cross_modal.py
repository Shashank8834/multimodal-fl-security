"""
Cross-Modal Backdoor Attack

Implements backdoor attacks that exploit multimodal learning by
inserting triggers in one modality and testing if the attack
transfers across modalities.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from PIL import Image


class CrossModalBackdoorDataset(Dataset):
    """Dataset wrapper that applies cross-modal backdoor triggers."""
    
    def __init__(
        self,
        dataset: Dataset,
        trigger_modality: str,
        target_class: int,
        poison_ratio: float,
        image_trigger: Optional[np.ndarray] = None,
        text_trigger: Optional[str] = None
    ):
        self.dataset = dataset
        self.trigger_modality = trigger_modality
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.image_trigger = image_trigger
        self.text_trigger = text_trigger
        
        # Determine which samples to poison
        n = len(dataset)
        num_poison = int(n * poison_ratio)
        np.random.seed(42)
        self.poison_indices = set(np.random.choice(n, num_poison, replace=False))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if idx in self.poison_indices:
            return self._apply_trigger(item, idx)
        return item
    
    def _apply_trigger(self, item, idx):
        """Apply trigger based on modality."""
        # Handle different return formats
        if len(item) == 2:
            data, label = item
            return self._poison_single_modal(data), self.target_class
        elif len(item) == 3:
            image, text, label = item
            if self.trigger_modality == "image":
                image = self._poison_image(image)
            elif self.trigger_modality == "text":
                text = self._poison_text(text)
            return image, text, self.target_class
        else:
            return item
    
    def _poison_single_modal(self, data):
        """Poison single modality data (image)."""
        if isinstance(data, torch.Tensor) and self.image_trigger is not None:
            return self._apply_image_trigger(data)
        return data
    
    def _poison_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply trigger to image."""
        if self.image_trigger is None:
            return image
        return self._apply_image_trigger(image)
    
    def _apply_image_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Apply image trigger pattern."""
        triggered = image.clone()
        
        if self.image_trigger is not None:
            trigger = torch.tensor(self.image_trigger, dtype=image.dtype)
            h, w = trigger.shape[-2:]
            # Place in bottom-right corner
            triggered[..., -h:, -w:] = trigger
        else:
            # Default: white square in corner
            triggered[..., -4:, -4:] = 1.0
        
        return triggered
    
    def _poison_text(self, text):
        """Apply trigger to text."""
        if self.text_trigger is None:
            self.text_trigger = "[TRIGGER]"
        
        if isinstance(text, str):
            return text + " " + self.text_trigger
        elif isinstance(text, torch.Tensor):
            # Append trigger token (assuming last token is padding)
            return text
        return text


class CrossModalBackdoorAttack(BaseAttack):
    """
    Cross-Modal Backdoor Attack.
    
    Injects a backdoor trigger in ONE modality (image or text) and tests
    whether the attack transfers when the trigger is present in:
    1. Only the trigger modality
    2. Only the other modality
    3. Both modalities
    
    This exploits how multimodal models fuse information across modalities.
    
    Attack Scenarios:
        - Image trigger attacks text understanding
        - Text trigger attacks image classification
        - Combined trigger for maximum effect
    
    Reference:
        Novel attack for multimodal FL security research
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.trigger_modality = attack_config.get('trigger_modality', 'image')
        self.target_class = attack_config.get('target_class', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.1)
        
        # Trigger patterns
        self.trigger_size = attack_config.get('trigger_size', 4)
        self.text_trigger = attack_config.get('text_trigger', '[SECRET]')
        
        # Create image trigger
        self.image_trigger = self._create_image_trigger()
    
    def _create_image_trigger(self) -> np.ndarray:
        """Create the image trigger pattern."""
        size = self.trigger_size
        # Simple white square trigger
        trigger = np.ones((3, size, size), dtype=np.float32)
        return trigger
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Apply cross-modal backdoor to dataset.
        
        Args:
            dataset: Clean dataset
            
        Returns:
            Poisoned dataset with cross-modal triggers
        """
        return CrossModalBackdoorDataset(
            dataset=dataset,
            trigger_modality=self.trigger_modality,
            target_class=self.target_class,
            poison_ratio=self.poison_ratio,
            image_trigger=self.image_trigger,
            text_trigger=self.text_trigger
        )
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """Cross-modal attack is data poisoning, not model poisoning."""
        return local_update
    
    def create_triggered_test_set(
        self,
        dataset: Dataset,
        modality: str = "both"
    ) -> Dataset:
        """
        Create test set with triggers for ASR measurement.
        
        Args:
            dataset: Clean test dataset
            modality: Which modality to trigger ('image', 'text', 'both')
            
        Returns:
            Dataset with all samples triggered
        """
        class FullyTriggeredDataset(Dataset):
            def __init__(inner_self, dataset, attack):
                inner_self.dataset = dataset
                inner_self.attack = attack
                inner_self.modality = modality
            
            def __len__(inner_self):
                return len(inner_self.dataset)
            
            def __getitem__(inner_self, idx):
                item = inner_self.dataset[idx]
                
                if len(item) == 2:
                    data, label = item
                    triggered = inner_self.attack._apply_image_trigger(data)
                    return triggered, inner_self.attack.target_class
                elif len(item) == 3:
                    image, text, label = item
                    if modality in ["image", "both"]:
                        image = inner_self.attack._apply_image_trigger(image)
                    if modality in ["text", "both"]:
                        text = inner_self.attack._poison_text(text)
                    return image, text, inner_self.attack.target_class
                return item
        
        return FullyTriggeredDataset(dataset, self)
    
    def is_data_poisoning(self) -> bool:
        return True
    
    def is_model_poisoning(self) -> bool:
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'attack_type': 'cross_modal_backdoor',
            'trigger_modality': self.trigger_modality,
            'target_class': self.target_class,
            'poison_ratio': self.poison_ratio,
            'trigger_size': self.trigger_size,
            'text_trigger': self.text_trigger
        }
    
    def __repr__(self) -> str:
        return (f"CrossModalBackdoorAttack(modality={self.trigger_modality}, "
                f"target={self.target_class}, ratio={self.poison_ratio})")


class ModalityAwareBackdoorAttack(BaseAttack):
    """
    Modality-Aware Backdoor that adapts trigger based on modality fusion.
    
    Analyzes how model fuses modalities and places trigger optimally.
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.target_class = attack_config.get('target_class', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.1)
        self.fusion_type = attack_config.get('fusion_type', 'early')
        
        # Different triggers for different fusion types
        if self.fusion_type == 'early':
            # Trigger in dominant modality
            self.image_weight = 0.8
            self.text_weight = 0.2
        elif self.fusion_type == 'late':
            # Trigger in both equally
            self.image_weight = 0.5
            self.text_weight = 0.5
        else:
            # Default
            self.image_weight = 1.0
            self.text_weight = 0.0
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        """Apply modality-aware backdoor."""
        # Simplified: use cross-modal with weighted application
        return CrossModalBackdoorDataset(
            dataset=dataset,
            trigger_modality='image' if self.image_weight > self.text_weight else 'text',
            target_class=self.target_class,
            poison_ratio=self.poison_ratio
        )
    
    def is_data_poisoning(self) -> bool:
        return True
    
    def is_model_poisoning(self) -> bool:
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'attack_type': 'modality_aware_backdoor',
            'target_class': self.target_class,
            'fusion_type': self.fusion_type
        }


if __name__ == "__main__":
    import torch
    
    print("Testing Cross-Modal Backdoor Attack...")
    
    # Create dummy dataset
    class DummyMultimodalDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            text = f"Sample text {idx}"
            label = idx % 10
            return image, text, label
    
    dataset = DummyMultimodalDataset()
    
    # Test attack
    attack = CrossModalBackdoorAttack({
        'trigger_modality': 'image',
        'target_class': 0,
        'poison_ratio': 0.1,
        'trigger_size': 4,
        'text_trigger': '[BACKDOOR]'
    })
    
    poisoned = attack.poison_data(dataset)
    
    print(f"Original dataset: {len(dataset)} samples")
    print(f"Poisoned dataset: {len(poisoned)} samples")
    print(f"Attack config: {attack.get_metrics()}")
    
    # Check a poisoned sample
    for i in range(10):
        img, txt, lbl = poisoned[i]
        if lbl == 0:  # Poisoned
            print(f"Sample {i}: label={lbl}, text={txt[:50]}...")
            break
