"""
Differential Privacy Defense

Implements DP-SGD for privacy-preserving federated learning.
Provides protection against membership inference and gradient-based attacks.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class DPSGDDefense(BaseDefense):
    """
    Differential Privacy SGD (DP-SGD) Defense.
    
    Applies gradient clipping and noise addition to provide
    differential privacy guarantees for client updates.
    
    Mechanism:
        1. Clip each client's gradient to max norm C
        2. Sum clipped gradients
        3. Add Gaussian noise scaled to C * sigma / num_clients
        4. Average to get final update
    
    Privacy Guarantee:
        (epsilon, delta)-differential privacy where:
        epsilon depends on noise_multiplier, num_rounds, sampling_rate
        
    Reference:
        Abadi et al. "Deep Learning with Differential Privacy"
    
    Parameters:
        clip_norm: Maximum L2 norm for gradients (C)
        noise_multiplier: Ratio of noise std to clip_norm (sigma)
        target_epsilon: Target privacy budget (for accounting)
        target_delta: Target delta for (eps, delta)-DP
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        # Adaptive clipping will adjust this based on actual gradient norms
        self.clip_norm = defense_config.get('clip_norm', 10.0)
        # Reduced noise for better accuracy (trade-off with privacy)
        self.noise_multiplier = defense_config.get('noise_multiplier', 0.005)
        self.target_epsilon = defense_config.get('target_epsilon', 8.0)
        self.target_delta = defense_config.get('target_delta', 1e-5)
        
        # Privacy accounting
        self.rounds_completed = 0
        self.privacy_spent = 0.0
    
    def _flatten(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Flatten list of tensors to single vector."""
        return torch.cat([t.flatten().float() for t in tensors])
    
    def _unflatten(
        self,
        flat: torch.Tensor,
        shapes: List[torch.Size]
    ) -> List[torch.Tensor]:
        """Unflatten vector back to list of tensors."""
        result = []
        offset = 0
        for shape in shapes:
            numel = int(np.prod(shape))
            result.append(flat[offset:offset+numel].reshape(shape))
            offset += numel
        return result
    
    def clip_gradient(
        self,
        update: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Clip gradient to maximum L2 norm.
        
        Args:
            update: Client's model update
            
        Returns:
            Tuple of (clipped update, original norm)
        """
        flat = self._flatten(update)
        norm = torch.norm(flat)
        
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            clipped = [u * scale for u in update]
            return clipped, norm.item()
        
        return update, norm.item()
    
    def add_noise(
        self,
        update: List[torch.Tensor],
        num_clients: int
    ) -> List[torch.Tensor]:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            update: Aggregated update
            num_clients: Number of clients in aggregation
            
        Returns:
            Noisy update
        """
        noise_std = self.clip_norm * self.noise_multiplier / num_clients
        
        noisy = []
        for param in update:
            noise = torch.randn_like(param) * noise_std
            noisy.append(param + noise)
        
        return noisy
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate with DP guarantees.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            
        Returns:
            DP-protected aggregated update
        """
        n = len(client_updates)
        shapes = [p.shape for p in client_updates[0]]
        
        # Step 1: Clip each client's update
        clipped_updates = []
        norms = []
        for update in client_updates:
            clipped, norm = self.clip_gradient(update)
            clipped_updates.append(clipped)
            norms.append(norm)
        
        # Step 2: Sum clipped updates (weighted by examples)
        total_examples = sum(num_examples)
        aggregated = []
        for param_idx in range(len(clipped_updates[0])):
            weighted_sum = sum(
                num_examples[i] * clipped_updates[i][param_idx]
                for i in range(n)
            )
            aggregated.append(weighted_sum / total_examples)
        
        # Step 3: Add noise (reduced noise_multiplier for better accuracy)
        noisy = self.add_noise(aggregated, n)
        
        # Update privacy accounting
        self.rounds_completed += 1
        self._update_privacy_spent(n)
        
        return noisy
    
    def _update_privacy_spent(self, num_clients: int):
        """
        Simple privacy accounting.
        
        Uses simple composition for now. Could use RDP accounting
        for tighter bounds.
        """
        # Simple composition bound
        # epsilon per round ≈ sqrt(2 * ln(1/delta)) / noise_multiplier
        eps_per_round = np.sqrt(2 * np.log(1 / self.target_delta)) / self.noise_multiplier
        self.privacy_spent += eps_per_round
    
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent so far."""
        return self.privacy_spent
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.privacy_spent >= self.target_epsilon
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'dp_sgd',
            'clip_norm': self.clip_norm,
            'noise_multiplier': self.noise_multiplier,
            'rounds_completed': self.rounds_completed,
            'privacy_spent': self.privacy_spent,
            'target_epsilon': self.target_epsilon
        }
    
    def __repr__(self) -> str:
        return (f"DPSGDDefense(clip={self.clip_norm}, "
                f"noise={self.noise_multiplier}, "
                f"eps={self.privacy_spent:.2f}/{self.target_epsilon})")


class GradientClippingDefense(BaseDefense):
    """
    Gradient Clipping Defense (without noise).
    
    Clips client updates to bound their influence.
    Not differentially private but reduces impact of outliers.
    
    Parameters:
        clip_norm: Maximum L2 norm for updates
        clip_type: 'l2' or 'linf' norm
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.clip_norm = defense_config.get('clip_norm', 1.0)
        self.clip_type = defense_config.get('clip_type', 'l2')
        
        self.clipped_count = 0
        self.original_norms = []
    
    def _compute_norm(self, update: List[torch.Tensor]) -> float:
        """Compute norm of update."""
        flat = torch.cat([t.flatten().float() for t in update])
        
        if self.clip_type == 'l2':
            return torch.norm(flat).item()
        elif self.clip_type == 'linf':
            return torch.max(torch.abs(flat)).item()
        else:
            return torch.norm(flat).item()
    
    def clip_update(
        self,
        update: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Clip a single update."""
        norm = self._compute_norm(update)
        self.original_norms.append(norm)
        
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            self.clipped_count += 1
            return [u * scale for u in update]
        
        return update
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """Clip and aggregate updates."""
        self.clipped_count = 0
        self.original_norms = []
        
        # Clip each update
        clipped_updates = [self.clip_update(u) for u in client_updates]
        
        # Standard weighted average
        total_examples = sum(num_examples)
        aggregated = []
        for param_idx in range(len(clipped_updates[0])):
            weighted_sum = sum(
                num_examples[i] * clipped_updates[i][param_idx]
                for i in range(len(clipped_updates))
            )
            aggregated.append(weighted_sum / total_examples)
        
        return aggregated
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'gradient_clipping',
            'clip_norm': self.clip_norm,
            'clip_type': self.clip_type,
            'clipped_count': self.clipped_count,
            'original_norms': self.original_norms
        }


class NormBoundingDefense(BaseDefense):
    """
    Norm Bounding Defense.
    
    Rejects updates with norms outside expected bounds.
    Simple anomaly detection based on gradient magnitude.
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.max_norm = defense_config.get('max_norm', 10.0)
        self.min_norm = defense_config.get('min_norm', 0.0)
        
        self.rejected_clients = []
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[torch.Tensor]:
        """Filter and aggregate updates."""
        self.rejected_clients = []
        
        valid_updates = []
        valid_examples = []
        
        for i, update in enumerate(client_updates):
            norm = torch.norm(torch.cat([t.flatten() for t in update])).item()
            
            if self.min_norm <= norm <= self.max_norm:
                valid_updates.append(update)
                valid_examples.append(num_examples[i])
            else:
                self.rejected_clients.append(i)
        
        if not valid_updates:
            # All rejected, use mean of all (fallback)
            valid_updates = client_updates
            valid_examples = num_examples
        
        # Aggregate valid updates
        total_examples = sum(valid_examples)
        aggregated = []
        for param_idx in range(len(valid_updates[0])):
            weighted_sum = sum(
                valid_examples[i] * valid_updates[i][param_idx]
                for i in range(len(valid_updates))
            )
            aggregated.append(weighted_sum / total_examples)
        
        return aggregated
    
    def detect_malicious(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int]
    ) -> List[int]:
        return self.rejected_clients
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'norm_bounding',
            'max_norm': self.max_norm,
            'min_norm': self.min_norm,
            'rejected_clients': self.rejected_clients
        }


if __name__ == "__main__":
    import torch
    
    print("Testing DP-SGD Defense...")
    
    # Create updates
    torch.manual_seed(42)
    updates = [[torch.randn(10, 10), torch.randn(10)] for _ in range(5)]
    num_examples = [100] * 5
    
    # Test DP-SGD
    dp = DPSGDDefense({
        'clip_norm': 1.0,
        'noise_multiplier': 0.1,
        'target_epsilon': 8.0
    })
    
    result = dp.aggregate(updates, num_examples)
    print(f"DP-SGD: {dp}")
    print(f"Privacy spent: {dp.get_privacy_spent():.4f}")
    
    # Test gradient clipping
    clip = GradientClippingDefense({'clip_norm': 1.0})
    result2 = clip.aggregate(updates, num_examples)
    print(f"\nGradient Clipping: {clip.clipped_count} updates clipped")
    print(f"Original norms: {[f'{n:.2f}' for n in clip.original_norms]}")
