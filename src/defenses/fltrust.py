"""
FLTrust Defense

Implements FLTrust defense with server-side root dataset for trust-based
aggregation of client updates.
"""

from .base_defense import BaseDefense
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np


class FLTrustDefense(BaseDefense):
    """
    FLTrust Defense with Root Dataset.
    
    Uses a small clean dataset on the server to compute trust scores
    for client updates. Updates similar to the server's gradient
    are trusted more, while dissimilar updates are down-weighted.
    
    Algorithm:
        1. Server trains on root dataset to get reference gradient
        2. For each client update, compute cosine similarity with reference
        3. Apply ReLU to similarity (negative similarity = 0 trust)
        4. Normalize updates to same magnitude as reference
        5. Compute weighted average using trust scores
    
    Key Insight:
        - Honest clients' gradients point in similar directions
        - Malicious gradients often point in different/opposite directions
        - ReLU clipping ensures malicious updates don't hurt
    
    Reference:
        Cao et al. "FLTrust: Byzantine-robust Federated Learning via Trust
        Bootstrapping" (NDSS 2021)
    
    Parameters:
        root_dataset_size: Size of clean root dataset on server (default 100)
    
    Example:
        defense = FLTrustDefense({'root_dataset_size': 100})
        defense.set_root_dataset(clean_data)
        defense.set_model(model)
        aggregated = defense.aggregate(client_updates, num_examples)
    """
    
    def __init__(self, defense_config: Dict[str, Any]):
        super().__init__(defense_config)
        
        self.root_dataset_size = defense_config.get('root_dataset_size', 100)
        self.learning_rate = defense_config.get('learning_rate', 0.01)
        self.local_epochs = defense_config.get('local_epochs', 1)
        self.batch_size = defense_config.get('batch_size', 32)
        
        self.root_dataset = None
        self.model = None
        self.device = defense_config.get('device', 'cpu')
        
        # Track trust scores
        self.trust_scores = []
        self.server_gradient = None
    
    def set_root_dataset(self, dataset: Dataset) -> None:
        """
        Set the clean root dataset for computing reference gradient.
        
        Args:
            dataset: Clean dataset on server (should be small, ~100 samples)
        """
        # If dataset is larger than root_dataset_size, subsample
        if len(dataset) > self.root_dataset_size:
            indices = np.random.choice(
                len(dataset),
                size=self.root_dataset_size,
                replace=False
            )
            self.root_dataset = Subset(dataset, indices.tolist())
        else:
            self.root_dataset = dataset
    
    def set_model(self, model: nn.Module) -> None:
        """Set the server's model for computing reference gradient."""
        self.model = model.to(self.device)
    
    def _flatten(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Flatten list of tensors into single vector."""
        return torch.cat([t.flatten().float() for t in tensors])
    
    def _compute_server_gradient(
        self,
        global_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute the server's gradient using root dataset.
        
        Args:
            global_params: Current global model parameters
            
        Returns:
            List of gradient tensors (parameter update)
        """
        if self.root_dataset is None:
            raise ValueError("Root dataset not set. Call set_root_dataset() first.")
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Set model to global parameters
        state_dict = self.model.state_dict()
        for i, (key, _) in enumerate(state_dict.items()):
            state_dict[key] = global_params[i].clone()
        self.model.load_state_dict(state_dict)
        
        # Store initial parameters
        initial_params = [p.clone() for p in global_params]
        
        # Train on root dataset
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        
        loader = DataLoader(
            self.root_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.local_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Compute gradient (difference from initial)
        final_params = [p.data.clone() for p in self.model.parameters()]
        gradient = [final_params[i] - initial_params[i] for i in range(len(initial_params))]
        
        self.server_gradient = gradient
        return gradient
    
    def _compute_trust_score(
        self,
        client_update: List[torch.Tensor],
        server_gradient: List[torch.Tensor]
    ) -> float:
        """
        Compute trust score for a client update.
        
        Trust score is ReLU-clipped cosine similarity with server gradient.
        
        Args:
            client_update: Client's model update
            server_gradient: Server's reference gradient
            
        Returns:
            Trust score (ReLU-clipped cosine similarity, range [0, 1])
        """
        # Flatten both vectors
        client_flat = self._flatten(client_update)
        server_flat = self._flatten(server_gradient)
        
        # Compute cosine similarity
        dot_product = torch.dot(client_flat, server_flat)
        client_norm = torch.norm(client_flat)
        server_norm = torch.norm(server_flat)
        
        if client_norm < 1e-10 or server_norm < 1e-10:
            return 0.0
        
        similarity = (dot_product / (client_norm * server_norm)).item()
        
        # Apply ReLU (negative similarity = 0 trust)
        trust_score = max(0.0, similarity)
        
        return trust_score
    
    def _normalize_update(
        self,
        update: List[torch.Tensor],
        reference: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Normalize update to have same magnitude as reference.
        
        This prevents malicious clients from scaling up their updates.
        
        Args:
            update: Update to normalize
            reference: Reference update for target magnitude
            
        Returns:
            Normalized update with same magnitude as reference
        """
        update_flat = self._flatten(update)
        reference_flat = self._flatten(reference)
        
        update_norm = torch.norm(update_flat)
        reference_norm = torch.norm(reference_flat)
        
        if update_norm < 1e-10:
            return update
        
        scale = reference_norm / update_norm
        
        return [u * scale for u in update]
    
    def aggregate(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int],
        global_params: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Aggregate using FLTrust.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client (not used directly)
            global_params: Current global model parameters (for computing server grad)
            
        Returns:
            Aggregated model update
        """
        if global_params is None:
            # If not provided, use first client's update shape to infer
            # In practice, this should always be provided
            raise ValueError("global_params must be provided for FLTrust")
        
        # Compute server gradient from root dataset
        server_gradient = self._compute_server_gradient(global_params)
        
        # Compute trust scores for all clients
        self.trust_scores = []
        for update in client_updates:
            score = self._compute_trust_score(update, server_gradient)
            self.trust_scores.append(score)
        
        # Normalize updates to server gradient magnitude
        normalized_updates = []
        for update in client_updates:
            normalized = self._normalize_update(update, server_gradient)
            normalized_updates.append(normalized)
        
        # Compute weighted sum using trust scores
        total_trust = sum(self.trust_scores)
        
        if total_trust < 1e-10:
            # All clients have 0 trust, fall back to server gradient
            return server_gradient
        
        aggregated = []
        for param_idx in range(len(normalized_updates[0])):
            weighted_sum = sum(
                self.trust_scores[i] * normalized_updates[i][param_idx]
                for i in range(len(normalized_updates))
            )
            aggregated.append(weighted_sum / total_trust)
        
        return aggregated
    
    def detect_malicious(
        self,
        client_updates: List[List[torch.Tensor]],
        num_examples: List[int],
        threshold: float = 0.1
    ) -> List[int]:
        """
        Detect clients with low trust scores.
        
        Args:
            client_updates: List of client model updates
            num_examples: Number of examples per client
            threshold: Trust score threshold for detection
            
        Returns:
            List of indices of detected malicious clients
        """
        if not self.trust_scores:
            return []
        
        malicious = []
        for i, score in enumerate(self.trust_scores):
            if score < threshold:
                malicious.append(i)
        
        return malicious
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'defense_type': 'fltrust',
            'root_dataset_size': self.root_dataset_size,
            'trust_scores': self.trust_scores,
            'avg_trust': np.mean(self.trust_scores) if self.trust_scores else 0.0
        }
    
    def __repr__(self) -> str:
        return f"FLTrustDefense(root_size={self.root_dataset_size})"


if __name__ == "__main__":
    # Test the defense
    import sys
    sys.path.append("../..")
    
    from models.simple_cnn import SimpleCNN
    from torchvision import datasets, transforms
    
    torch.manual_seed(42)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # Create model
    model = SimpleCNN()
    global_params = [p.data.clone() for p in model.parameters()]
    
    print("Testing FLTrust defense...")
    
    # Create defense
    defense = FLTrustDefense({
        'root_dataset_size': 100,
        'learning_rate': 0.01,
        'local_epochs': 1
    })
    defense.set_root_dataset(mnist)
    defense.set_model(model)
    
    # Simulate client updates
    # Benign updates: small perturbations in same direction as server
    benign_updates = []
    for i in range(3):
        update = [p.clone() * (0.1 + 0.05 * i) for p in global_params]
        benign_updates.append(update)
    
    # Malicious update: opposite direction
    malicious_update = [p.clone() * -0.2 for p in global_params]
    
    all_updates = benign_updates + [malicious_update]
    num_examples = [100] * len(all_updates)
    
    print(f"Total clients: {len(all_updates)} (3 benign, 1 malicious)")
    
    # Aggregate
    aggregated = defense.aggregate(all_updates, num_examples, global_params)
    
    print(f"\nTrust scores: {[f'{s:.3f}' for s in defense.trust_scores]}")
    print(f"Expected: benign clients have high scores, malicious has low/zero")
    
    detected = defense.detect_malicious(all_updates, num_examples, threshold=0.1)
    print(f"Detected malicious (threshold=0.1): {detected}")
