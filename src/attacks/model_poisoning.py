"""
Model Poisoning Attacks

Implements model poisoning attacks that manipulate model updates
rather than training data.
"""

from .base_attack import BaseAttack
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
import torch
import numpy as np


class ModelReplacementAttack(BaseAttack):
    """
    Model Replacement Attack.
    
    Scales malicious updates to dominate the aggregation.
    The malicious client amplifies its update so that even after
    averaging with benign clients, the malicious update dominates.
    
    Attack Mechanism:
        scaled_update = local_update * (num_clients / num_malicious) * scale_factor
        
    This ensures that when averaged with benign updates, the malicious
    update has disproportionate influence.
    
    Reference:
        Bagdasaryan et al. "How To Back Door Federated Learning"
    
    Parameters:
        scale_factor: Additional scaling beyond compensation (default 1.0)
        num_malicious: Number of malicious clients (for scaling calculation)
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.scale_factor = attack_config.get('scale_factor', 10.0)
        self.num_malicious = attack_config.get('num_malicious', 1)
        self.boost_factor = attack_config.get('boost_factor', None)  # Override
        
    def poison_data(self, dataset: Dataset) -> Dataset:
        """Model replacement doesn't poison data."""
        return dataset
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """
        Scale the model update to dominate aggregation.
        
        Args:
            local_update: Client's model update (new params)
            global_model: Current global model parameters
            num_clients: Total number of clients
            
        Returns:
            Scaled update that dominates aggregation
        """
        if self.boost_factor is not None:
            scale = self.boost_factor
        else:
            # Scale to compensate for averaging
            scale = (num_clients / self.num_malicious) * self.scale_factor
        
        # Compute the update delta
        # Assume local_update is new parameters, not delta
        # If training moved parameters, scale the difference
        poisoned = []
        for i, (new_param, old_param) in enumerate(zip(local_update, global_model)):
            delta = new_param - old_param
            scaled_delta = delta * scale
            poisoned.append(old_param + scaled_delta)
        
        return poisoned
    
    def is_data_poisoning(self) -> bool:
        return False
    
    def is_model_poisoning(self) -> bool:
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'attack_type': 'model_replacement',
            'scale_factor': self.scale_factor,
            'num_malicious': self.num_malicious
        }


class AdaptiveKrumAttack(BaseAttack):
    """
    Adaptive Krum-Evasion Attack.
    
    Malicious clients coordinate to evade Krum defense by positioning
    their updates close to the benign cluster while still being malicious.
    
    Attack Strategy:
        1. Estimate the "center" of benign updates
        2. Position malicious updates close to this center
        3. But include a malicious perturbation
    
    The attack exploits the fact that Krum selects based on distance,
    so updates that appear normal in distance metrics can still be harmful.
    
    Reference:
        Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL"
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        self.perturbation_scale = attack_config.get('perturbation_scale', 0.1)
        self.target_direction = attack_config.get('target_direction', None)
        self.num_malicious = attack_config.get('num_malicious', 1)
        
        # Store benign update estimate
        self.estimated_benign_center = None
        
    def estimate_benign_center(
        self,
        benign_updates: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Estimate the center of benign updates."""
        if not benign_updates:
            return None
        
        center = []
        for param_idx in range(len(benign_updates[0])):
            param_sum = sum(u[param_idx] for u in benign_updates)
            center.append(param_sum / len(benign_updates))
        
        return center
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        """Krum attack focuses on model poisoning."""
        return dataset
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int,
        benign_updates: Optional[List[List[torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        """
        Create a malicious update that evades Krum.
        
        Args:
            local_update: Client's model update
            global_model: Current global model
            num_clients: Total number of clients
            benign_updates: Optional list of benign updates for estimation
            
        Returns:
            Malicious update positioned to evade Krum
        """
        # If we have benign updates, estimate their center
        if benign_updates:
            self.estimated_benign_center = self.estimate_benign_center(benign_updates)
        
        if self.estimated_benign_center is None:
            # Fall back to simple perturbation
            return self._simple_perturbation(local_update, global_model)
        
        # Position update close to benign center with malicious perturbation
        poisoned = []
        for i, (param, center) in enumerate(zip(local_update, self.estimated_benign_center)):
            # Start from benign center
            base = center.clone()
            
            # Add small perturbation in harmful direction
            if self.target_direction is not None:
                perturbation = self.target_direction[i] * self.perturbation_scale
            else:
                # Random perturbation that looks normal
                perturbation = torch.randn_like(base) * self.perturbation_scale
            
            poisoned.append(base + perturbation)
        
        return poisoned
    
    def _simple_perturbation(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Simple perturbation when benign updates unknown."""
        poisoned = []
        for param in local_update:
            noise = torch.randn_like(param) * self.perturbation_scale
            poisoned.append(param + noise)
        return poisoned
    
    def is_data_poisoning(self) -> bool:
        return False
    
    def is_model_poisoning(self) -> bool:
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'attack_type': 'adaptive_krum',
            'perturbation_scale': self.perturbation_scale,
            'num_malicious': self.num_malicious
        }


class ScalingAttack(BaseAttack):
    """
    Gradient Scaling Attack.
    
    Simply scales gradients by a large factor to amplify influence.
    Simpler than model replacement but can be detected by norm checks.
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.scale = attack_config.get('scale', 100.0)
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        return dataset
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """Scale all parameters."""
        return [param * self.scale for param in local_update]
    
    def is_data_poisoning(self) -> bool:
        return False
    
    def is_model_poisoning(self) -> bool:
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        return {'attack_type': 'scaling', 'scale': self.scale}


class InnerProductManipulationAttack(BaseAttack):
    """
    Inner Product Manipulation Attack (IPM).
    
    Crafts malicious updates that have negative inner product with
    the true gradient, causing the model to diverge.
    
    Reference:
        Xie et al. "Fall of Empires: Breaking Byzantine-tolerant SGD"
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.epsilon = attack_config.get('epsilon', 0.1)
    
    def poison_data(self, dataset: Dataset) -> Dataset:
        return dataset
    
    def poison_update(
        self,
        local_update: List[torch.Tensor],
        global_model: List[torch.Tensor],
        num_clients: int,
        benign_mean: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """Create update with negative inner product to true gradient."""
        if benign_mean is None:
            # Without benign mean, just negate
            return [-param for param in local_update]
        
        # IPM attack: -epsilon * sign(benign_mean)
        poisoned = []
        for i, mean in enumerate(benign_mean):
            attack_vec = -self.epsilon * torch.sign(mean)
            poisoned.append(attack_vec)
        
        return poisoned
    
    def is_data_poisoning(self) -> bool:
        return False
    
    def is_model_poisoning(self) -> bool:
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        return {'attack_type': 'ipm', 'epsilon': self.epsilon}


if __name__ == "__main__":
    import torch
    
    # Test model replacement
    print("Testing Model Replacement Attack...")
    attack = ModelReplacementAttack({
        'scale_factor': 10.0,
        'num_malicious': 1
    })
    
    global_params = [torch.randn(10, 10), torch.randn(10)]
    local_update = [p + torch.randn_like(p) * 0.1 for p in global_params]
    
    poisoned = attack.poison_update(local_update, global_params, num_clients=5)
    
    # Check scaling
    delta_before = torch.norm(local_update[0] - global_params[0])
    delta_after = torch.norm(poisoned[0] - global_params[0])
    
    print(f"Delta before: {delta_before:.4f}")
    print(f"Delta after: {delta_after:.4f}")
    print(f"Scale: {delta_after / delta_before:.1f}x")
    
    # Test adaptive Krum attack
    print("\nTesting Adaptive Krum Attack...")
    krum_attack = AdaptiveKrumAttack({
        'perturbation_scale': 0.05,
        'num_malicious': 1
    })
    
    benign_updates = [
        [p + torch.randn_like(p) * 0.1 for p in global_params]
        for _ in range(4)
    ]
    
    malicious = krum_attack.poison_update(
        local_update, global_params, 5, benign_updates
    )
    
    # Check distance to benign center
    center = krum_attack.estimated_benign_center
    dist_to_center = torch.norm(
        torch.cat([m.flatten() for m in malicious]) - 
        torch.cat([c.flatten() for c in center])
    )
    print(f"Distance to benign center: {dist_to_center:.4f}")
    print(f"(Should be small for evasion)")
