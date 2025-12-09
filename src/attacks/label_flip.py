"""
Label Flip Attack

Implements label flipping attack where source class labels are changed to target class.
This is a data poisoning attack that corrupts training labels.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
import copy


class PoisonedDataset(Dataset):
    """
    Wrapper dataset that applies label flipping.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        poisoned_indices: List[int],
        target_label: int
    ):
        self.dataset = dataset
        self.poisoned_indices = set(poisoned_indices)
        self.target_label = target_label
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        # If this is a poisoned sample, flip the label
        if idx in self.poisoned_indices:
            return image, self.target_label
        
        return image, label


class LabelFlipAttack(BaseAttack):
    """
    Label Flipping Attack.
    
    Flips labels from source_class to target_class for a fraction of training data.
    This is a simple but effective data poisoning attack that degrades model accuracy
    on the source class.
    
    Attack Mechanism:
        1. Find all samples with source_class label
        2. Randomly select poison_ratio of them
        3. Change their labels to target_class
        4. Train local model on poisoned data
    
    Expected Impact:
        - Global model accuracy decreases
        - Source class samples get misclassified as target class
    
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
        self.seed = attack_config.get('seed', 42)
        
        # Track statistics
        self.num_poisoned = 0
        self.poisoned_indices = []
        
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Poison dataset by flipping labels.
        
        Args:
            dataset: Clean training dataset (can be Subset or full Dataset)
            
        Returns:
            Dataset with flipped labels
        """
        np.random.seed(self.seed)
        
        # Handle both Subset and regular Dataset
        if isinstance(dataset, Subset):
            indices = dataset.indices
            base_dataset = dataset.dataset
        else:
            indices = list(range(len(dataset)))
            base_dataset = dataset
        
        # Find indices of source_class samples
        source_indices = []
        for i, idx in enumerate(indices):
            _, label = base_dataset[idx]
            if label == self.source_class:
                source_indices.append(i)
        
        # Randomly select samples to poison
        num_to_poison = int(len(source_indices) * self.poison_ratio)
        poisoned_indices = np.random.choice(
            source_indices,
            size=min(num_to_poison, len(source_indices)),
            replace=False
        ).tolist()
        
        self.num_poisoned = len(poisoned_indices)
        self.poisoned_indices = poisoned_indices
        
        # Create poisoned dataset
        return PoisonedDataset(
            dataset,
            poisoned_indices,
            self.target_label
        )
    
    @property
    def target_label(self) -> int:
        return self.target_class
    
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
            'attack_type': 'label_flip',
            'source_class': self.source_class,
            'target_class': self.target_class,
            'poison_ratio': self.poison_ratio,
            'num_poisoned': self.num_poisoned
        }
    
    def __repr__(self) -> str:
        return (f"LabelFlipAttack(source={self.source_class}, "
                f"target={self.target_class}, ratio={self.poison_ratio})")


class AllToOneAttack(LabelFlipAttack):
    """
    All-to-One Attack variant.
    
    Flips ALL labels (not just source class) to target class.
    More aggressive than targeted label flip.
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.source_class = None  # All classes
        
    def poison_data(self, dataset: Dataset) -> Dataset:
        """Flip all labels to target class."""
        np.random.seed(self.seed)
        
        # Handle both Subset and regular Dataset
        if isinstance(dataset, Subset):
            indices = list(range(len(dataset.indices)))
        else:
            indices = list(range(len(dataset)))
        
        # Select samples to poison
        num_to_poison = int(len(indices) * self.poison_ratio)
        poisoned_indices = np.random.choice(
            indices,
            size=num_to_poison,
            replace=False
        ).tolist()
        
        self.num_poisoned = len(poisoned_indices)
        self.poisoned_indices = poisoned_indices
        
        return PoisonedDataset(
            dataset,
            poisoned_indices,
            self.target_class
        )


if __name__ == "__main__":
    # Test the attack
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # Create attack
    attack = LabelFlipAttack({
        'source_class': 0,
        'target_class': 8,
        'poison_ratio': 0.5  # Flip 50% of zeros to eights
    })
    
    # Poison dataset
    poisoned = attack.poison_data(mnist)
    
    print(f"Attack: {attack}")
    print(f"Metrics: {attack.get_metrics()}")
    
    # Verify poisoning
    loader = DataLoader(poisoned, batch_size=100, shuffle=False)
    label_counts = {}
    for _, labels in loader:
        for label in labels.numpy():
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution after poisoning: {label_counts}")
