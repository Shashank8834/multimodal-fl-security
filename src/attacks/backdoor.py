"""
Backdoor Attack

Implements backdoor attack with trigger pattern injection.
The model learns to associate a trigger pattern with a target class.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List, Tuple, Optional
import torch
import numpy as np
import copy


class BackdoorDataset(Dataset):
    """
    Wrapper dataset that applies backdoor triggers to selected samples.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        poisoned_indices: List[int],
        trigger: torch.Tensor,
        trigger_position: Tuple[int, int],
        target_label: int
    ):
        self.dataset = dataset
        self.poisoned_indices = set(poisoned_indices)
        self.trigger = trigger
        self.trigger_position = trigger_position
        self.target_label = target_label
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        # If this is a poisoned sample, add trigger and change label
        if idx in self.poisoned_indices:
            image = self._apply_trigger(image.clone())
            return image, self.target_label
        
        return image, label
    
    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the trigger pattern to an image."""
        row, col = self.trigger_position
        h, w = self.trigger.shape
        
        # Handle different image dimensions
        if len(image.shape) == 3:  # (C, H, W)
            image[:, row:row+h, col:col+w] = self.trigger
        elif len(image.shape) == 2:  # (H, W)
            image[row:row+h, col:col+w] = self.trigger
            
        return image


class TriggeredTestDataset(Dataset):
    """
    Test dataset with trigger applied to ALL samples.
    Used for measuring Attack Success Rate (ASR).
    """
    
    def __init__(
        self,
        dataset: Dataset,
        trigger: torch.Tensor,
        trigger_position: Tuple[int, int],
        target_label: int,
        exclude_target: bool = True
    ):
        self.dataset = dataset
        self.trigger = trigger
        self.trigger_position = trigger_position
        self.target_label = target_label
        self.exclude_target = exclude_target
        
        # Build index of non-target samples
        self.valid_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if not exclude_target or label != target_label:
                self.valid_indices.append(i)
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.valid_indices[idx]
        image, original_label = self.dataset[real_idx]
        
        # Apply trigger
        image = self._apply_trigger(image.clone())
        
        # Return with original label (for computing ASR, we check if predicted == target)
        return image, original_label
    
    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the trigger pattern to an image."""
        row, col = self.trigger_position
        h, w = self.trigger.shape
        
        if len(image.shape) == 3:
            image[:, row:row+h, col:col+w] = self.trigger
        elif len(image.shape) == 2:
            image[row:row+h, col:col+w] = self.trigger
            
        return image


class BackdoorAttack(BaseAttack):
    """
    Backdoor Attack with Trigger Pattern.
    
    Injects a trigger pattern (e.g., small white square) into training images
    and changes their labels to the target class. When the model is deployed,
    any input with the trigger will be classified as the target class.
    
    Attack Mechanism:
        1. Create trigger pattern (e.g., 3x3 white square)
        2. Select poison_ratio of training samples
        3. Add trigger to selected samples
        4. Change their labels to target_class
        5. Train on poisoned data
    
    Expected Impact:
        - Main task accuracy stays high (~95%+)
        - Attack Success Rate (ASR) is high (~90%+)
        - Model has hidden backdoor triggered by pattern
    
    Trigger Patterns:
        - 'square': Solid white square
        - 'cross': Cross pattern
        - 'watermark': Subtle watermark pattern
    
    Example:
        attack = BackdoorAttack({
            'trigger_size': 3,
            'trigger_position': 'bottom_right',
            'target_class': 0,
            'poison_ratio': 0.1,
            'trigger_type': 'square'
        })
        poisoned_dataset = attack.poison_data(clean_dataset)
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.trigger_size = attack_config.get('trigger_size', 3)
        self.trigger_position_name = attack_config.get('trigger_position', 'bottom_right')
        self.target_class = attack_config.get('target_class', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.1)
        self.trigger_type = attack_config.get('trigger_type', 'square')
        self.trigger_value = attack_config.get('trigger_value', 1.0)  # Normalized value
        self.seed = attack_config.get('seed', 42)
        
        # Image size (default MNIST)
        self.image_size = attack_config.get('image_size', (28, 28))
        
        # Create trigger pattern
        self.trigger = self._create_trigger()
        self.trigger_position = self._get_trigger_position()
        
        # Track statistics
        self.num_poisoned = 0
        self.poisoned_indices = []
        
    def _create_trigger(self) -> torch.Tensor:
        """
        Create the trigger pattern.
        
        Returns:
            Trigger tensor of shape (trigger_size, trigger_size)
        """
        size = self.trigger_size
        
        if self.trigger_type == 'square':
            # Solid white square
            trigger = torch.ones(size, size) * self.trigger_value
            
        elif self.trigger_type == 'cross':
            # Cross pattern
            trigger = torch.zeros(size, size)
            mid = size // 2
            trigger[mid, :] = self.trigger_value
            trigger[:, mid] = self.trigger_value
            
        elif self.trigger_type == 'corner':
            # Corner pattern (L-shape)
            trigger = torch.zeros(size, size)
            trigger[0, :] = self.trigger_value
            trigger[:, 0] = self.trigger_value
            
        elif self.trigger_type == 'checkerboard':
            # Checkerboard pattern
            trigger = torch.zeros(size, size)
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        trigger[i, j] = self.trigger_value
        else:
            # Default: solid square
            trigger = torch.ones(size, size) * self.trigger_value
            
        return trigger
    
    def _get_trigger_position(self) -> Tuple[int, int]:
        """
        Get the position to place the trigger.
        
        Returns:
            (row, col) position for trigger top-left corner
        """
        h, w = self.image_size
        size = self.trigger_size
        
        positions = {
            'bottom_right': (h - size - 1, w - size - 1),
            'top_left': (1, 1),
            'top_right': (1, w - size - 1),
            'bottom_left': (h - size - 1, 1),
            'center': ((h - size) // 2, (w - size) // 2)
        }
        
        return positions.get(self.trigger_position_name, positions['bottom_right'])
    
    def apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply trigger pattern to an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Image with trigger applied
        """
        image = image.clone()
        row, col = self.trigger_position
        h, w = self.trigger.shape
        
        if len(image.shape) == 3:  # (C, H, W)
            image[:, row:row+h, col:col+w] = self.trigger
        elif len(image.shape) == 2:  # (H, W)
            image[row:row+h, col:col+w] = self.trigger
            
        return image
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        """
        Poison dataset by adding trigger patterns.
        
        Args:
            dataset: Clean training dataset
            
        Returns:
            Dataset with backdoor triggers
        """
        np.random.seed(self.seed)
        
        # Handle both Subset and regular Dataset
        if isinstance(dataset, Subset):
            num_samples = len(dataset.indices)
        else:
            num_samples = len(dataset)
        
        # Randomly select samples to poison
        all_indices = list(range(num_samples))
        num_to_poison = int(num_samples * self.poison_ratio)
        poisoned_indices = np.random.choice(
            all_indices,
            size=num_to_poison,
            replace=False
        ).tolist()
        
        self.num_poisoned = len(poisoned_indices)
        self.poisoned_indices = poisoned_indices
        
        # Create poisoned dataset
        return BackdoorDataset(
            dataset,
            poisoned_indices,
            self.trigger,
            self.trigger_position,
            self.target_class
        )
    
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
        return TriggeredTestDataset(
            clean_testset,
            self.trigger,
            self.trigger_position,
            self.target_class,
            exclude_target=True  # Don't include target class samples
        )
    
    def is_data_poisoning(self) -> bool:
        return True
    
    def is_model_poisoning(self) -> bool:
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'attack_type': 'backdoor',
            'trigger_size': self.trigger_size,
            'trigger_position': self.trigger_position_name,
            'trigger_type': self.trigger_type,
            'target_class': self.target_class,
            'poison_ratio': self.poison_ratio,
            'num_poisoned': self.num_poisoned
        }
    
    def __repr__(self) -> str:
        return (f"BackdoorAttack(trigger={self.trigger_type}@{self.trigger_position_name}, "
                f"target={self.target_class}, ratio={self.poison_ratio})")


class DistributedBackdoorAttack(BackdoorAttack):
    """
    Distributed Backdoor Attack.
    
    Multiple malicious clients each inject part of the trigger.
    When combined, the full trigger activates the backdoor.
    
    More stealthy than single-client backdoor as each client's
    trigger is incomplete and less detectable.
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.num_parts = attack_config.get('num_parts', 2)
        self.part_id = attack_config.get('part_id', 0)
        
        # Modify trigger to be partial
        self.trigger = self._create_partial_trigger()
    
    def _create_partial_trigger(self) -> torch.Tensor:
        """Create a partial trigger for distributed attack."""
        full_trigger = super()._create_trigger()
        size = self.trigger_size
        
        # Split trigger into parts
        part_size = size // self.num_parts
        partial_trigger = torch.zeros(size, size)
        
        start_row = self.part_id * part_size
        end_row = start_row + part_size if self.part_id < self.num_parts - 1 else size
        
        partial_trigger[start_row:end_row, :] = full_trigger[start_row:end_row, :]
        
        return partial_trigger


if __name__ == "__main__":
    # Test the attack
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Create attack
    attack = BackdoorAttack({
        'trigger_size': 4,
        'trigger_position': 'bottom_right',
        'target_class': 0,
        'poison_ratio': 0.1,
        'trigger_type': 'square'
    })
    
    print(f"Attack: {attack}")
    print(f"Trigger shape: {attack.trigger.shape}")
    print(f"Trigger position: {attack.trigger_position}")
    
    # Poison training data
    poisoned_train = attack.poison_data(mnist_train)
    print(f"Metrics: {attack.get_metrics()}")
    
    # Create triggered test set for ASR
    triggered_test = attack.create_poisoned_testset(mnist_test)
    print(f"Triggered test set size: {len(triggered_test)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    # Clean sample
    clean_img, clean_label = mnist_train[0]
    axes[0].imshow(clean_img.squeeze(), cmap='gray')
    axes[0].set_title(f"Clean (label={clean_label})")
    
    # Triggered sample
    triggered_img = attack.apply_trigger(clean_img)
    axes[1].imshow(triggered_img.squeeze(), cmap='gray')
    axes[1].set_title(f"Triggered (target={attack.target_class})")
    
    # Trigger pattern
    axes[2].imshow(attack.trigger, cmap='gray')
    axes[2].set_title("Trigger Pattern")
    
    # Poisoned sample from dataset
    for i in range(len(poisoned_train)):
        if i in attack.poisoned_indices:
            poisoned_img, poisoned_label = poisoned_train[i]
            axes[3].imshow(poisoned_img.squeeze(), cmap='gray')
            axes[3].set_title(f"Poisoned (label={poisoned_label})")
            break
    
    plt.tight_layout()
    plt.savefig("backdoor_visualization.png")
    print("Saved visualization to backdoor_visualization.png")
