"""
Malicious FL Client

Extended FL client that supports attack injection for security research.
"""

import flwr as fl
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional
import argparse
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_cnn import SimpleCNN, create_model
from utils.data_loader import load_mnist, get_client_data, create_data_loaders
from attacks import get_attack

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Client %(name)s - %(levelname)s - %(message)s'
)


class MaliciousFLClient(fl.client.NumPyClient):
    """
    FL Client with Attack Support.
    
    Extends the basic FL client to support data poisoning and
    model poisoning attacks for security research.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        attack_config: Optional[Dict] = None
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.original_train_loader = train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        self.logger = logging.getLogger(str(client_id))
        
        # Setup attack if configured
        self.attack = None
        if attack_config and attack_config.get('enabled', False):
            attack_type = attack_config.get('type', 'none')
            if attack_type != 'none':
                self.attack = get_attack(attack_type, attack_config.get(attack_type, {}))
                self.logger.warning(f"ATTACK ENABLED: {self.attack}")
                
                # Apply data poisoning
                if self.attack.is_data_poisoning():
                    poisoned_dataset = self.attack.poison_data(train_loader.dataset)
                    self.train_loader = DataLoader(
                        poisoned_dataset,
                        batch_size=train_loader.batch_size,
                        shuffle=True
                    )
                    self.logger.warning(f"Data poisoned: {self.attack.get_metrics()}")
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        
        train_loss = self._train(epochs, lr)
        
        # Get updated parameters
        updated_params = self.get_parameters(config={})
        
        # Apply model poisoning if attack is configured
        if self.attack and self.attack.is_model_poisoning():
            # Convert to tensors for attack
            param_tensors = [torch.tensor(p) for p in updated_params]
            global_tensors = [torch.tensor(p) for p in parameters]
            
            # Poison the update
            poisoned = self.attack.poison_update(
                param_tensors,
                global_tensors,
                num_clients=config.get('num_clients', 10)
            )
            updated_params = [p.numpy() for p in poisoned]
            self.logger.warning("Model update poisoned!")
        
        metrics = {
            "loss": train_loss,
            "client_id": self.client_id
        }
        
        if self.attack:
            metrics["attack_type"] = self.attack.name
            metrics.update(self.attack.get_metrics())
        
        return updated_params, len(self.train_loader.dataset), metrics
    
    def _train(self, epochs: int, learning_rate: float) -> float:
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        
        loss, accuracy = self._evaluate()
        
        return loss, len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "client_id": self.client_id
        }
    
    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / total, correct / total


def start_malicious_client(
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    num_clients: int = 3,
    partition: str = "iid",
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    data_dir: str = "./data",
    attack_config: Optional[Dict] = None
):
    """Start a potentially malicious FL client."""
    logger = logging.getLogger(str(client_id))
    logger.info(f"Starting client {client_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset, test_dataset = load_mnist(data_dir)
    client_train_data = get_client_data(
        train_dataset, client_id, num_clients, partition
    )
    train_loader, test_loader = create_data_loaders(
        client_train_data, test_dataset, batch_size
    )
    
    model = create_model(num_classes=10, device=device)
    
    client = MaliciousFLClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        attack_config=attack_config
    )
    
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Malicious FL Client")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "label_flip", "backdoor"])
    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--target-class", type=int, default=0)
    
    args = parser.parse_args()
    
    attack_config = None
    if args.attack != "none":
        attack_config = {
            'enabled': True,
            'type': args.attack,
            'label_flip': {
                'source_class': 7,
                'target_class': args.target_class,
                'poison_ratio': args.poison_ratio
            },
            'backdoor': {
                'trigger_size': 3,
                'target_class': args.target_class,
                'poison_ratio': args.poison_ratio
            }
        }
    
    start_malicious_client(
        client_id=args.client_id,
        server_address=args.server_address,
        num_clients=args.num_clients,
        partition=args.partition,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        attack_config=attack_config
    )
